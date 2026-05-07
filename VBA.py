import math
import time
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorBundleProjection(nn.Module):
    """
    Projects input token features x -> base coords b_i and M fiber candidates {f_i^(m)}.
    """
    def __init__(self, d_model: int, base_dim: int, fiber_dim: int, num_bundles: int):
        super().__init__()
        self.base_proj  = nn.Linear(d_model, base_dim)
        self.base_norm  = nn.LayerNorm(base_dim)

        self.num_bundles = num_bundles
        self.fiber_dim   = fiber_dim

        self.fiber_proj  = nn.Linear(d_model, num_bundles * fiber_dim)
        self.fiber_norm  = nn.LayerNorm(num_bundles * fiber_dim)

      
        self.bundle_heads = nn.ModuleList(
            [nn.Linear(fiber_dim, fiber_dim, bias=True) for _ in range(num_bundles)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, d_model)
        returns:
          b: (B, N, Db)
          f_candidates: (B, N, M, df)
        """
        b = self.base_norm(self.base_proj(x))               
        f_all = self.fiber_norm(self.fiber_proj(x))        

        B, N, _ = f_all.shape
        M, df = self.num_bundles, self.fiber_dim
        f_candidates = f_all.view(B, N, M, df)               

  
        f_out = []
        for m in range(M):
            f_out.append(self.bundle_heads[m](f_candidates[:, :, m, :]))  
        f_candidates = torch.stack(f_out, dim=2)            
        return b, f_candidates


class BundleSelector(nn.Module):
    """
    Produces mixing weights alpha_i over M bundles, per token.
    """
    def __init__(self, d_model: int, num_bundles: int, temperature: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(d_model, num_bundles)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,N,d_model)
        return: alpha (B,N,M), sum_M alpha = 1
        """
        logits = self.fc(x) / self.temperature
        alpha = F.softmax(logits, dim=-1)
        return alpha



class TransportNetExp(nn.Module):
    """
    Learns an endpoint-dependent, orthogonal transport operator on a d-by-d space:
       S_skew = 0.5*(S - S^T);  T = exp(alpha * S_skew) \in SO(d).
    We use it on the fiber space (dim = fiber_dim) to follow the transport→project design.
    """
    def __init__(self, base_dim: int, mat_dim: int, hidden_dim: int = 64, scale_init: float = 0.1):
        super().__init__()
        self.mat_dim = mat_dim
        
        self.alpha = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(base_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mat_dim * mat_dim)
        )
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _skew(A: torch.Tensor) -> torch.Tensor:
        return 0.5 * (A - A.transpose(-1, -2))

    def _exp_map(self, S_skew: torch.Tensor) -> torch.Tensor:
        """
        Batched matrix exponential for (..., d, d) skew-symmetric matrices.
        """
        T = torch.matrix_exp(self.alpha * S_skew)
        return T  

    def _pairwise_features(self, bq: torch.Tensor, bk: torch.Tensor) -> torch.Tensor:
    
        pair = torch.cat([bq, bk], dim=-1)              
        flat = pair.reshape(-1, pair.shape[-1])
        d = self.mat_dim
        S = self.mlp(flat).view(*pair.shape[:-1], d, d)  
        return self._skew(S)

    def forward(self, b_query: torch.Tensor, b_key: torch.Tensor) -> torch.Tensor:
        """
        Dense: b_query (B,Nq,Db), b_key (B,Nk,Db) -> (B,Nq,Nk,d,d).
        """
        B, Nq, Db = b_query.shape
        _, Nk, _  = b_key.shape
        bq = b_query.unsqueeze(2).expand(-1, -1, Nk, -1)
        bk = b_key.unsqueeze(1).expand(-1, Nq, -1, -1)
        S_skew = self._pairwise_features(bq, bk)
        return self._exp_map(S_skew)

    def forward_local(self, b_query: torch.Tensor, b_key_neighbors: torch.Tensor) -> torch.Tensor:
        """
        Local KNN: b_query (B,N,Db), b_key_neighbors (B,N,K,Db) -> (B,N,K,d,d)
        """
        bq = b_query.unsqueeze(2).expand(-1, -1, b_key_neighbors.shape[2], -1) 
        bk = b_key_neighbors                                                     
        S_skew = self._pairwise_features(bq, bk)                                 
        return self._exp_map(S_skew)                                            



class ConnectionNet(nn.Module):
    """
    ConnectionNet: b -> {Gamma_k(b)}_{k=1..Db}, each Gamma_k in gl(df).
    Returns shape (B,N,Db,df,df).
    """
    def __init__(self, base_dim: int, fiber_dim: int, hidden: int = 64):
        super().__init__()
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        self.net = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, base_dim * fiber_dim * fiber_dim)
        )
    
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        B, N, Db = b.shape
        df = self.fiber_dim
        out = self.net(b)                         
        Gamma = out.view(B, N, Db, df, df)       
        return Gamma


def _commutator(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B - B @ A


class CurvatureAdapter(nn.Module):
    """
    Maps invariant scalars -> scalar gates (alpha, beta, gamma, delta).
    By default we use [tr(S), tr(S^2), logdet(I+eta S)] as inputs (dim_in=3).
    """
    def __init__(self, dim_in: int = 3, hidden: int = 32, use_delta: bool = False):
        super().__init__()
        dim_out = 3 + int(use_delta)  
        self.use_delta = use_delta
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim_out)
        )
      
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, kappa: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        kappa: (B,N,dim_in)
        returns tuple of scalars per token: (alpha, beta, gamma[, delta]) with shape (B,N)
        """
        gates = self.mlp(kappa) 
        if self.use_delta:
            alpha, beta, gamma, delta = gates.unbind(dim=-1)
            return alpha, beta, gamma, delta
        else:
            alpha, beta, gamma = gates.unbind(dim=-1)
            return alpha, beta, gamma


class CurvatureModuleFromConnection(nn.Module):
    """
    Paper-aligned curvature pipeline:
      - Gamma(b) from ConnectionNet(b)
      - Omega_ij = ∂Γ_j/∂x_i - ∂Γ_i/∂x_j + [Γ_i, Γ_j]
      - S = Σ_{i<j} Omega_ij^T Omega_ij (PSD)
      - Invariants kappa = [tr(S), tr(S^2), logdet(I+eta S)]
      - R_eff = alpha I + beta S_tilde + gamma S_tilde^2 (+ delta R_dir)
      - Return R_eff (B,N,df,df)
    Notes:
      * Uses autograd.functional.jacobian per token (slow for large Db/df). Vectorization is possible.
      * By default create_graph=False to avoid 2nd-order costs; set flag if needed.
    """
    def __init__(self, base_dim: int, fiber_dim: int,
                 learn_pair_weights: bool = True,
                 backprop_through_jacobian: bool = False,
                 eta_logdet: float = 1e-2,
                 eps_norm: float = 1e-6,
                 use_directional: bool = False):
        super().__init__()
        self.conn = ConnectionNet(base_dim, fiber_dim)
        self.base_dim = base_dim
        self.fiber_dim = fiber_dim
        self.learn_pair_weights = learn_pair_weights
        self.backprop_through_jacobian = backprop_through_jacobian
        self.eta_logdet = eta_logdet
        self.eps_norm = eps_norm
        self.use_directional = use_directional

        if learn_pair_weights:
            # map b -> weights over i<j pairs
            num_pairs = base_dim * (base_dim - 1) // 2
            self.w_head = nn.Sequential(
                nn.Linear(base_dim, 32),
                nn.GELU(),
                nn.Linear(32, num_pairs)
            )

      
        self.adapter = CurvatureAdapter(dim_in=3, hidden=32, use_delta=use_directional)

      
        if use_directional:
            num_pairs = base_dim * (base_dim - 1) // 2
            self.sigma_head = nn.Sequential(
                nn.Linear(base_dim, 32),
                nn.GELU(),
                nn.Linear(32, num_pairs)
            )

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        """
        b: (B,N,Db)
        return: R_eff: (B,N,df,df)
        """
        B, N, Db = b.shape
        df = self.fiber_dim
        device = b.device
        dtype = b.dtype

     
        Gamma = self.conn(b) 

 
        pairs: List[Tuple[int, int]] = [(i, j) for i in range(Db) for j in range(i + 1, Db)]
        num_pairs = len(pairs)
        if self.learn_pair_weights:
            logits = self.w_head(b)             
            w_pairs = F.softmax(logits, dim=-1)   
        else:
            w_pairs = None

   
        if self.use_directional:
            sigma_logits = self.sigma_head(b)            
     
            sigma = sigma_logits / (sigma_logits.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            sigma = None

        R_eff_all = torch.zeros(B, N, df, df, device=device, dtype=dtype)

  
        I = torch.eye(df, device=device, dtype=dtype).view(1, 1, df, df)

        for bi in range(B):
            for ni in range(N):
                x = b[bi, ni].detach().requires_grad_(True) 

        
                def conn_single(x_vec: torch.Tensor) -> torch.Tensor:
                    x_in = x_vec.view(1, 1, Db)
                    G = self.conn(x_in)                     
                    return G.view(Db, df, df)

         
                G0 = conn_single(x)                          

                def flat_conn(x_vec):
                    return conn_single(x_vec).reshape(-1)

                J = torch.autograd.functional.jacobian(
                    flat_conn, x,
                    create_graph=self.backprop_through_jacobian,
                    vectorize=True
                )                                            
                J = J.view(Db, Db, df, df)                    

  
                S_sum = torch.zeros(df, df, device=device, dtype=dtype)
                R_dir = torch.zeros(df, df, device=device, dtype=dtype)
                for p_idx, (i, j) in enumerate(pairs):
                    dGamma = J[i, j] - J[j, i]               
                    comm   = _commutator(G0[i], G0[j])       
                    Oij    = dGamma + comm                   

                    S_sum = S_sum + Oij.transpose(-1, -2) @ Oij  

                    if sigma is not None:
                        R_dir = R_dir + sigma[bi, ni, p_idx] * Oij

     
                trS  = torch.einsum('ii->', S_sum)  
                trS2 = torch.einsum('ij,ij->', S_sum, S_sum) 
                PD = I[0,0] + self.eta_logdet * S_sum
                sign, logabsdet = torch.linalg.slogdet(PD)
                logdetI = logabsdet 

                kappa = torch.stack([trS, trS2, logdetI], dim=0).view(1, 1, 3) 
                gates = self.adapter(kappa) 

                if self.use_directional:
                    alpha, beta, gamma, delta = [gates[i].view(1, 1, 1, 1) for i in range(4)]
                else:
                    alpha, beta, gamma = [g.view(1, 1, 1, 1) for g in gates]
                    delta = torch.zeros_like(alpha)

            
                S_norm = S_sum / (trS + self.eps_norm)

            
                R_eff = alpha * I + beta * S_norm + gamma * (S_norm @ S_norm) + delta * R_dir
                R_eff_all[bi, ni] = R_eff

        return R_eff_all  



def knn_indices(base: torch.Tensor, k: int) -> torch.Tensor:
    """
    base: (B,N,Db) -> indices (B,N,k), smallest distances.
    """
    with torch.no_grad():
        dist = torch.cdist(base, base)                 
        knn = dist.topk(k, largest=False).indices      
    return knn


def gather_neighbors(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather neighbors along N-dimension.
    tensor: (B,N,D) -> (B,N,k,D)
    """
    B, N, D = tensor.shape
    k = idx.shape[-1]
    idx_exp = idx.unsqueeze(-1).expand(B, N, k, D)     
    tensor_exp = tensor.unsqueeze(2).expand(B, N, k, D)
    out = torch.gather(tensor_exp, dim=1, index=idx_exp)
    return out


class VectorBundleAttention(nn.Module):
    """
    Geometric attention with orthogonal transport in F (fiber), then linear projection to Q/K/V heads.
    This enforces the "transport → project" order for better geometric consistency.
    """
    def __init__(self, d_model: int, n_heads: int, base_dim: int, fiber_dim: int,
                 dropout: float = 0.1, k_neighbors: Optional[int] = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.fiber_dim = fiber_dim
    
        self.to_q = nn.Linear(fiber_dim, d_model)
        self.to_k = nn.Linear(fiber_dim, d_model)
        self.to_v = nn.Linear(fiber_dim, d_model)

  
        self.transport_net = TransportNetExp(base_dim, mat_dim=fiber_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.k_neighbors = k_neighbors  

    def forward_dense(self, base: torch.Tensor, fiber: torch.Tensor) -> torch.Tensor:
        """
        base:  (B,N,Db)
        fiber: (B,N,df)
        """
        B, N, df = fiber.shape
        H, Dh = self.n_heads, self.head_dim

 
        q = self.to_q(fiber).view(B, N, H, Dh).transpose(1, 2)  

   
        T = self.transport_net(base, base)                      

     
        f_trans = torch.einsum('bijnm,bjm->bijn', T, fiber)     

     
        k_trans = self.to_k(f_trans).view(B, N, N, H, Dh).permute(0, 3, 1, 2, 4) 
        v_trans = self.to_v(f_trans).view(B, N, N, H, Dh).permute(0, 3, 1, 2, 4)  

 
        attn_scores = torch.einsum('bhid,bhij d->bhij', q, k_trans) * self.scale  
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

      
        out = torch.einsum('bhij,bhij d->bhid', attn_weights, v_trans)            
        out = out.transpose(1, 2).reshape(B, N, H * Dh)                           
        return self.out_proj(out)

    def forward_knn(self, base: torch.Tensor, fiber: torch.Tensor, kN: int) -> torch.Tensor:
        """
        base:  (B,N,Db)
        fiber: (B,N,df)
        """
        B, N, df = fiber.shape
        H, Dh = self.n_heads, self.head_dim

     
        q = self.to_q(fiber).view(B, N, H, Dh).transpose(1, 2)  

        idx = knn_indices(base, k=kN)                            
        base_nb  = gather_neighbors(base, idx)                   
        fiber_nb = gather_neighbors(fiber, idx)                 

        T_local = self.transport_net.forward_local(base, base_nb)  


        f_trans_nb = torch.einsum('bnkij,bnkj->bnki', T_local, fiber_nb)  

    
        k_trans_nb = self.to_k(f_trans_nb).view(B, N, kN, H, Dh).permute(0, 3, 1, 2, 4)  
        v_trans_nb = self.to_v(f_trans_nb).view(B, N, kN, H, Dh).permute(0, 3, 1, 2, 4)  


        q_exp = q.unsqueeze(3)                                  
        attn_scores = torch.matmul(q_exp, k_trans_nb.transpose(-1, -2)).squeeze(3) 
        attn_scores = attn_scores * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.einsum('bhnk,bhnkd->bhnd', attn_weights, v_trans_nb)            
        out = out.transpose(1, 2).reshape(B, N, H * Dh)                           
        return self.out_proj(out)

    def forward(self, base: torch.Tensor, fiber: torch.Tensor) -> torch.Tensor:
        if self.k_neighbors is None:
            return self.forward_dense(base, fiber)
        else:
            return self.forward_knn(base, fiber, self.k_neighbors)



class VBABlock(nn.Module):
    """
    VBA Transformer Block:
      - Pre-LN
      - Projection to base + M fiber candidates
      - Bundle selection & mixing
      - Curvature correction via R_eff(b): f <- f + lambda * R_eff * f
      - Geometric attention (transport in fiber, then project to Q/K/V)
      - FFN
    """
    def __init__(self, dim: int, heads: int, base_dim: int, fiber_dim: int, num_bundles: int,
                 dropout: float = 0.1, k_neighbors: Optional[int] = None,
                 learn_pair_weights: bool = True, curvature_scale: float = 1.0):
        super().__init__()
        self.num_bundles = num_bundles
        self.fiber_dim   = fiber_dim
        self.lambda_curv = curvature_scale

        self.projection = VectorBundleProjection(dim, base_dim, fiber_dim, num_bundles)
        self.bundle_selector = BundleSelector(dim, num_bundles)

        self.curvature = CurvatureModuleFromConnection(
            base_dim, fiber_dim,
            learn_pair_weights=learn_pair_weights,
            backprop_through_jacobian=False,    
            use_directional=False               
        )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = VectorBundleAttention(dim, heads, base_dim, fiber_dim, dropout=dropout,
                                          k_neighbors=k_neighbors)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B,N,dim)
        """

        x1 = self.norm1(x)


        b, f_candidates = self.projection(x1)           


        alpha = self.bundle_selector(x1).unsqueeze(-1)  
        f_mixed = (alpha * f_candidates).sum(dim=2)     

   
        R_eff = self.curvature(b)                        
        f_corrected = f_mixed + self.lambda_curv * torch.einsum('bnij,bnj->bni', R_eff, f_mixed)

     
        x = x + self.attn(b, f_corrected)


        x = x + self.ffn(self.norm2(x))
        return x


class VBAEncoder(nn.Module):
    """A simple stack of VBA blocks acting as an encoder."""
    def __init__(self, dim, depth, heads, base_dim, fiber_dim, num_bundles,
                 dropout=0.1, k_neighbors=8, learn_pair_weights=True, curvature_scale=1.0):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(
                VBABlock(dim=dim, heads=heads, base_dim=base_dim, fiber_dim=fiber_dim,
                         num_bundles=num_bundles, dropout=dropout, k_neighbors=k_neighbors,
                         learn_pair_weights=learn_pair_weights, curvature_scale=curvature_scale)
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


def main():
    parser = argparse.ArgumentParser(description="VBA quick smoke test (transport→project version)")
    parser.add_argument("--batch", type=int, default=2, help="batch size B")
    parser.add_argument("--tokens", type=int, default=32, help="sequence length N")
    parser.add_argument("--dim", type=int, default=128, help="model dimension per token")
    parser.add_argument("--depth", type=int, default=2, help="number of VBABlocks")
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--base_dim", type=int, default=2, help="Db (start with 2)")
    parser.add_argument("--fiber_dim", type=int, default=16, help="df (fiber space dim)")
    parser.add_argument("--bundles", type=int, default=4, help="number of bundles M")
    parser.add_argument("--k", type=int, default=8, help="KNN neighbors; set 0 for dense attention")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--steps", type=int, default=5, help="training steps (tiny regression)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_pair_weights", action="store_true",
                        help="disable learned pair (i,j) weights; use uniform average")
    parser.add_argument("--curv_scale", type=float, default=1.0, help="lambda for curvature modulation")
    parser.add_argument("--compile", action="store_true",
                        help="use torch.compile if available (PyTorch 2.0+)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Device: {device}")

    k_neighbors = None if args.k <= 0 else args.k
    model = VBAEncoder(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        base_dim=args.base_dim,
        fiber_dim=args.fiber_dim,
        num_bundles=args.bundles,
        dropout=args.dropout,
        k_neighbors=k_neighbors,
        learn_pair_weights=not args.no_pair_weights,
        curvature_scale=args.curv_scale,
    ).to(device)
    print(model)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("[Info] Using torch.compile")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}")


    B, N, D = args.batch, args.tokens, args.dim
    x = torch.randn(B, N, D, device=device)
    target = torch.zeros(B, N, D, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    with torch.no_grad():
        y = model(x)
    print(f"[Shape] input: {tuple(x.shape)} -> output: {tuple(y.shape)}")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        optim.zero_grad(set_to_none=True)
        y = model(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        optim.step()

        if step == 1:
            t1 = time.time()
        print(f"[Step {step:02d}] loss={loss.item():.6f}")
    t2 = time.time()

    print(f"[Timing] first fwd/bwd step: {(t1 - t0):.3f}s  | "
          f"avg step over {args.steps} steps: {(t2 - t1)/args.steps:.3f}s")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Params] total parameters: {n_params/1e6:.3f} M")


if __name__ == "__main__":
    main()
