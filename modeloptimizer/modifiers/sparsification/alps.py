# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/sparsegpt/base.py
# licensed under the Apache License 2.0
#
# ALPS (Alternating Linearized Pruning and Smoothing): Second-order pruning using
# Hessian information. Combines ADMM for sparsity constraint with conjugate
# gradient for final refinement. Supports both unstructured and N:M structured sparsity.

import time
import numpy as np

import torch
from torch.nn import Module
from torchtitan.tools.logging import logger

from modeloptimizer.modifiers.sparsification.base import SparsityModifierBase
from modeloptimizer.observers.hessian import PRECISION, HessianObserver
from modeloptimizer.utils.pytorch.module import TransformerConv1D

__all__ = ["AlpsModifier"]


class AlpsModifier(SparsityModifierBase):
    """
    Modifier for applying the one-shot ALPS algorithm to a model
    from the paper: https://arxiv.org/abs/2406.07831

    Sample yaml:

    ```yaml
    sparsity_modifiers:
        AlpsModifier:
            sparsity: 0.5
            mask_structure: "2:4"
            dampening_frac: 0.01
            alps_rho: 1.0
            alps_iterations: 16
            alps_update_iter: 16
            alps_switch_iter: 16
            targets: ['Linear']
            ignore: ['re:.*lm_head']
    ```

    Lifecycle:

    - on_initialize
        - register_hook(module, calibrate_module, "forward")
    - on_sequential_batch_end
        - sparsify_weight
    - on_finalize
        - remove_hooks()

    :param sparsity: Sparsity to compress model
    :param mask_structure: String to define the structure of the mask to apply.
        Must be of the form N:M where N, M are integers that define a custom block
        shape. Defaults to 0:0 which represents an unstructured mask.
    :param dampening_frac: Amount of dampening to apply to H, as a fraction of the
        diagonal norm
    :param alps_rho: Regularization parameter
    :param alps_iterations: Number of iterations to run the ALPS algorithm
    :param alps_update_iter: Number of iterations to update the support
    :param alps_switch_iter: Number of iterations to switch the support
    :param targets: list of layer names to compress during ALPS, or '__ALL__'
        to compress every layer in the model. 
    :param ignore: optional list of module class names or submodule names to not
        sparsify even if they match a target. Defaults to empty list.
    """

    # modifier arguments
    dampening_frac: float | None = 0.01
    alps_rho: float = 1.0
    alps_iterations: int = 16
    alps_update_iter: int = 16
    alps_switch_iter: int = 16

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        # ALPS uses Hessian of activations; must use HessianObserver for calibration
        self._observer_name = "hessian"
        return super().on_initialize(model_parts, **kwargs)

    def compress_modules(self):
        """
        Sparsify modules which have been calibrated (Hessian stats collected).
        Each layer is compressed in-place via _sparsify_weight and weight.data.copy_.
        """
        for module, observer in self._layer_observers.items():
            name = self._module_names[module]
            sparsity = self._module_sparsities[module]
            num_samples = observer.num_samples

            logger.info(f"Sparsifying {name} using {num_samples.item()} samples")
            assert isinstance(
                observer, HessianObserver
            ), "AlpsModifier requires hessian observer"
            sparsified_weight, W_mask = self._sparsify_weight(
                module=module,
                hessian=observer.stats,
                sparsity=sparsity,
                prune_n=self._prune_n,
                prune_m=self._prune_m,
            )
            module.weight.data.copy_(sparsified_weight)

    def _sparsify_weight(
        self,
        module: Module,
        hessian: torch.Tensor,
        sparsity: float,
        prune_n: int,
        prune_m: int,
    ) -> tuple[float, torch.Tensor]:
        """
        Run pruning on the layer up to the target sparsity value.

        :param module: module with weight being sparsified
        :param hessian: Hessian matrix of the activations
        :param sparsity: target sparsity to reach for layer
        :param prune_n: N for N:M pruning
        :param prune_m: M for N:M pruning
        """
        # --- Phase 1: Weight and Hessian setup ---
        final_shape = module.weight.shape
        final_dtype = module.weight.dtype
        W = module.weight.clone()
        W = W.to(dtype=PRECISION)
        H = hessian
        if isinstance(module, torch.nn.Conv2d):
            W = W.flatten(1)
        elif TransformerConv1D and isinstance(module, TransformerConv1D):
            W.transpose_(0, 1)
        W_mask_T = torch.zeros_like(W, dtype=torch.bool).T
        # Dampening: add small diagonal to H for numerical stability
        damp1 = self.dampening_frac * torch.mean(torch.diag(H)).item()
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp1
        
        # Symmetric normalization: H' = H / (sqrt(diag(H)) * sqrt(diag(H))^T)
        # so that diag(H') = 1; keeps conditioning under control
        X_norm = torch.diag(H).sqrt() + 1e-8
        H = H / X_norm
        H = (H.T / X_norm).T    
        
        # Right-hand side for linear system: GT = (W * X_norm) @ H (used as G^T in paper)
        GT = torch.zeros_like(W)
        GT = torch.matmul(W * X_norm, H)

        # --- Phase 2: ADMM variables and eigendecomposition ---
        # B = primal (weight) variable; D = auxiliary with sparsity; V = dual
        # Work in normalized space: B = (W * X_norm)^T
        H_inv = torch.zeros_like(H).float()
        B = (W * X_norm).t().clone()
        W = None
        B_orig = B.clone()
        V = torch.zeros_like(B)
        D = torch.zeros_like(B)
        D_suppp = torch.zeros_like(B)  # previous support mask (for convergence check)
        D_supp = torch.zeros_like(B)

        totp, num_cout = B.shape
        # (H + rho*I)^{-1} via eigendecomposition: H = Q L Q^T => (H+rho*I)^{-1} = Q (L+rho)^{-1} Q^T
        try:
            L, Q = torch.linalg.eigh(H.double())
        except Exception:
            H_cpu = H.cpu()
            L, Q = torch.linalg.eigh(H_cpu.double())
            H = H_cpu.to(H.device)
        H_inv = (Q @ ((1 / (L + (self.alps_rho))) * Q).T).float().to(H.device)
        
        init_rho = False   # True after we have increased rho at least once
        fix_supp = False   # True when support is frozen (optional late phase)
        D_fix = torch.zeros_like(D)
        
        # Reference residual scale for relative error (Res0 = <B_orig, G^T>)
        Res0 = GT.T
        Res0 = torch.sum(B_orig * Res0)
        Res0 = torch.sum(Res0)

        params = B.shape[0]*B.shape[1]
        k_spar = int(np.round((1-sparsity)*params))  # number of nonzeros to keep
        
        # --- Phase 3: Initial sparsity pattern (projection onto constraint set) ---
        if prune_n == -1:
            D = B.clone().reshape(-1)
            _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
            D[loss_idx] = 0    
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(totp, num_cout)
        else:
            # N:M structured: reshape to blocks of size prune_m, zero out (prune_m - prune_n) smallest in each block
            new_dim = totp * num_cout / prune_m
            new_dim = int(new_dim)
            k_spar = totp * num_cout * prune_n/prune_m
            D = B.clone().t().reshape((new_dim, prune_m))
            _, loss_idx = torch.topk(-D**2,prune_m - prune_n, dim = 1)
            D = D.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to(H.device),dim=1,index=loss_idx)   
            D_suppp = (D == 0).to(torch.float)
            D = D.reshape(num_cout, totp).t()
    
        D_init = D.clone()
        # --- Phase 4: ADMM iterations ---
        # Minimize (1/2)*||H*B - G^T||^2 s.t. B has sparsity pattern; B = primal, D = auxiliary, V = dual
        for i_admm in range(self.alps_iterations):
            # B-update: (H + rho*I) B = G^T - V + rho*D  =>  B = H_inv @ (GT.T - V + rho*D)
            B = H_inv @ (GT.T-V+self.alps_rho*D)
            # D-update: project (V + rho*B)/rho onto sparsity constraint (keep k_spar largest, zero rest)
            if fix_supp:
                D = ((V + self.alps_rho * B) / self.alps_rho) * D_fix
            elif prune_n == -1:
                # Unstructured: zero out (totp*num_cout - k_spar) smallest entries by magnitude
                D = ((V + self.alps_rho * B) / self.alps_rho).reshape(-1)
                _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
                D[loss_idx] = 0    
                D = D.reshape(totp, num_cout)   
            else:
                # N:M: in each block of prune_m, zero (prune_m - prune_n) smallest
                D = ((V + self.alps_rho * B) / self.alps_rho).t().reshape((new_dim, prune_m))
                _, loss_idx = torch.topk(-D**2,prune_m - prune_n, dim = 1)
                D = D.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to(H.device),dim=1,index=loss_idx) 
                D_supp = (D == 0).to(torch.float)  
                D = D.reshape(num_cout, totp).t()  

            # V-update (dual)
            V = V + self.alps_rho * (B - D)
            
            # Periodically: check support change, adapt rho, and optionally recompute H_inv
            if (i_admm+1) % self.alps_update_iter == 0:
                if prune_n == -1:
                    D_supp = (D.reshape(-1) == 0).to(torch.float)
                supp_change = torch.sum((D_supp-D_suppp)**2)  # measure how much support changed
                
                # Rho adaptation: increase rho when support is unstable, decrease when stable
                if not fix_supp:
                    if supp_change / k_spar > 0.1:
                        init_rho = True
                        self.alps_rho *= 1.3
                    elif supp_change / k_spar > 0.005:
                        init_rho = True
                        self.alps_rho *= 1.2
                    elif supp_change > 0.5:
                        if init_rho:
                            self.alps_rho *= 1.1
                        else:
                            # Large change without prior rho increase: restart with smaller rho
                            self.alps_rho /= 5
                            B = B_orig.clone().to(H.device)
                            D = D_init.clone().to(H.device)
                            V = torch.zeros_like(B).to(H.device)     
                    else:
                        if init_rho:
                            break  # support stable and rho was increased before -> converged
                        else:
                            self.alps_rho /= 5
                
                D_suppp = (D_supp).clone()
                if self.alps_rho > 1e6:
                    self.alps_rho = 1e6
            
                # Recompute (H + rho*I)^{-1} after rho change
                H_inv = (Q @ ((1 / (L + self.alps_rho)) * Q).T).float().to(H.device)
                
                # Relative residual error for early termination check
                if prune_n == -1:
                    Btest = B.reshape(-1)
                    _, loss_idx = torch.topk(-Btest**2,totp * num_cout - k_spar)
                    Btest[loss_idx] = 0    
                    Btest = Btest.reshape(totp, num_cout)
                else:
                    Btest = B.t().reshape((new_dim, prune_m))
                    _, loss_idx = torch.topk(-Btest**2,prune_m - prune_n, dim = 1)
                    Btest = Btest.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to(H.device),dim=1,index=loss_idx)  
                    Btest = Btest.reshape(num_cout, totp).t()
            
                Resc = torch.matmul(H.to(H.device),Btest) - GT.T
                Resc = torch.diag(torch.matmul((Btest-B_orig.to(H.device)).t(), Resc))
        
                errorc = torch.sum(Resc)/Res0
                errorc = errorc.item()
                # Early exit: after switch_iter, if support barely changes
                if i_admm >= self.alps_switch_iter and supp_change / k_spar < 0.0003:
                    break

        # --- Phase 5: Final hard projection onto sparsity (same as D-update) ---
        # prune_n == 0: treat as unstructured (zero all but k_spar largest)
        if prune_n == 0:
            B = B.reshape(-1)
            _, loss_idx = torch.topk(-B**2,totp * num_cout - k_spar)
            B[loss_idx] = 0    
            B = B.reshape(totp, num_cout)
        else:
            B = B.t().reshape((new_dim, prune_m))
            _, loss_idx = torch.topk(-B**2,prune_m - prune_n, dim = 1)
            B = B.scatter(src=torch.zeros((new_dim,prune_m-prune_n)).to(H.device),dim=1,index=loss_idx)  
            B = B.reshape(num_cout, totp).t()
        
        W_mask_T = (B == 0).to(torch.bool)

        V = None
        D = None

        Res = torch.matmul(H, B) - GT.T
        Res = torch.diag(torch.matmul((B  -B_orig).t(), Res))
        
        error = torch.sum(Res)/Res0
        error = error.item()

        # --- Phase 6: Conjugate gradient refinement ---
        # Solve H*B = GT.T only on the support (B fixed to 0 elsewhere); improves objective on same mask
        B = self.cg_batch(H, GT.T, (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10)
        Res = torch.matmul(H,B) - GT.T
        Res = torch.diag(torch.matmul((B -B_orig).t(), Res))
        
        error = torch.sum(Res)/Res0
        error = error.item()
        
        torch.cuda.synchronize()

        # Denormalize and reshape back to original weight layout
        if TransformerConv1D and isinstance(module, TransformerConv1D):
            sparsified_weight = (B.t() / X_norm).t().reshape(final_shape).to(final_dtype)
        else:
            sparsified_weight = (B.t() / X_norm).reshape(final_shape).to(final_dtype)
        return sparsified_weight, W_mask_T.T.reshape(final_shape)

    def cg_batch(self, A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None):
        """
        Conjugate gradient (CG) solver for A @ X = B, with support constraint:
        only the entries where A_supp != 0 are updated; others stay 0 (masked residual).
        Solves one system per column of B (batch over columns).
        """
        n, m = B.shape
        if M_bmm is None:
            M_bmm = lambda x: x
        if X0 is None:
            X0 = M_bmm(B)
        if maxiter is None:
            maxiter = 5 * n
        assert B.shape == (n, m)
        assert X0.shape == (n, m)
        assert rtol > 0 or atol > 0
        assert isinstance(maxiter, int)
        X_k = X0
        R_k = B - A @ X_k
        R_k = R_k * A_supp   # residual only on support
        Z_k = M_bmm(R_k)
        P_k = torch.zeros_like(Z_k)
        P_k1, R_k1, R_k2, Z_k1, Z_k2 = P_k, R_k, R_k, Z_k, Z_k
        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))
        optimal = False
        start = time.perf_counter()
        for k in range(1, maxiter + 1):
            Z_k = M_bmm(R_k)
            if k == 1:
                P_k, R_k1, X_k1, Z_k1 = Z_k, R_k, X_k, Z_k
            else:
                # Polak-Ribiere beta for conjugate direction
                R_k2, Z_k2, P_k1, R_k1, Z_k1, X_k1 = R_k1, Z_k1, P_k, R_k, Z_k, X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1
            # Line search: alpha such that X_k minimizes residual along P_k
            denominator = (P_k * (A@P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A@P_k)
            R_k = R_k * A_supp   # keep residual zero outside support
            residual_norm = torch.norm(A@X_k - B, dim=1)
            logger.info("%03d | %8.4e" % (k, torch.max(residual_norm/B_norm)))
            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break
        end = time.perf_counter()
        if optimal:
            logger.info("Terminated in %d steps (optimal). Took %.3f ms." %(k, (end - start) * 1000))
        else:
            logger.info("Terminated in %d steps (reached maxiter). Took %.3f ms." %(k, (end - start) * 1000))
        return X_k