import time
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import transformers

from src.utils import ALGO_REGISTRY
from .blockwise_sparsification import BlockwiseSparsification


@ALGO_REGISTRY
class ALPS(BlockwiseSparsification):
    def __init__(self, model, sparsity_config, global_config, input):
        super().__init__(model, sparsity_config, global_config, input)
        self.optimization_method_name = 'ALPS'
        self.applicability_message = 'ALPS is only suitable for unstructured and N:M sparsity pattern.'
        assert self.block_sparsity_config == False, self.applicability_message
        self.percdamp = sparsity_config['method_kwargs'].get('percdamp', 0.01)
        self.max_iter = sparsity_config['method_kwargs'].get('max_iter', 300)
        self.update_iter = sparsity_config['method_kwargs'].get('update_iter', 3)
        self.switch_iter = sparsity_config['method_kwargs'].get('switch_iter', 30)
        self.rho = sparsity_config['method_kwargs'].get('rho', 0.1)

    @torch.no_grad()
    def add_batch(self, layer, inp, H, nsamples):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        minibatch_size = inp.shape[0]
        if isinstance(layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        nsamples += minibatch_size
        inp = inp.float().to(H.device)
        H += inp.matmul(inp.t())
        return H, nsamples

    def cg_batch(self, A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None):
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
        R_k = R_k * A_supp
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
                R_k2, Z_k2, P_k1, R_k1, Z_k1, X_k1 = R_k1, Z_k1, P_k, R_k, Z_k, X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1
            denominator = (P_k * (A@P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A@P_k)
            R_k = R_k * A_supp
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

    @torch.no_grad()
    def optimize_subset(
        self,
        layers_dict,
        input_feat,
        output_feat,
        prev_op,
        input_name,
        inspect_module,
        block_idx,
        subset_kwargs,
    ):
        for name, layer in layers_dict.items():
            global_layer_name = f'layers.{block_idx}.{name}'
            if self.sparsity_dict is not None:
                sparsity = self.sparsity_dict[global_layer_name]
            elif isinstance(self.sparsity, list):
                sparsity = self.sparsity[block_idx]
            else:
                sparsity = self.sparsity
            logger.info(f"Sparsity of {name} is {sparsity}.")

            W = layer.weight.data.clone()
            W_mask_T = torch.zeros_like(W, dtype=torch.bool).T
            device = W.device
            if isinstance(layer, transformers.Conv1D):
                W = W.t()
            rows, columns = W.shape[0], W.shape[1]
            W = W.float()
            H = torch.zeros((columns, columns), device=device, dtype=torch.float)
            nsamples = 0
            for batch_idx in range(len(input_feat[input_name])):
                H, nsamples = self.add_batch(layer, input_feat[input_name][batch_idx], H, nsamples)

            damp1 = self.percdamp * torch.mean(torch.diag(H)).item()
            diag = torch.arange(H.shape[0], device=H.device)
            H[diag,diag] += damp1
            
            # normalization 
            X_norm = torch.diag(H).sqrt() + 1e-8
            H = H / X_norm
            H = (H.T / X_norm).T    
            
            GT = torch.zeros_like(W)
            GT = torch.matmul(W * X_norm, H)

            # initialization
            H_inv = torch.zeros_like(H).float()
            B = (W * X_norm).t().clone()
            W = None
            B_orig = B.clone()
            V = torch.zeros_like(B)
            D = torch.zeros_like(B)
            D_suppp = torch.zeros_like(B)
            D_supp = torch.zeros_like(B)

            totp, num_cout = B.shape
            try:
                L, Q = torch.linalg.eigh(H.double())
            except Exception:
                H_cpu = H.cpu()
                L, Q = torch.linalg.eigh(H_cpu.double())
                H = H_cpu.to(device)
            H_inv = (Q @ ((1/(L+(self.rho))) * Q).T).float().to(device)
            
            init_rho = False
            fix_supp = False
            D_fix = torch.zeros_like(D)
            
            Res0 = GT.T
            Res0 = torch.sum(B_orig * Res0)
            Res0 = torch.sum(Res0)

            params = B.shape[0]*B.shape[1]
            k_spar = int(np.round((1-sparsity)*params))
            
            if self.N == -1:
                D = B.clone().reshape(-1)
                _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
                D[loss_idx] = 0    
                D_suppp = (D == 0).to(torch.float)
                D = D.reshape(totp, num_cout)
            else:
                new_dim = totp * num_cout / self.M
                new_dim = int(new_dim)
                k_spar = totp * num_cout * self.N/self.M
                D = B.clone().t().reshape((new_dim, self.M))
                _, loss_idx = torch.topk(-D**2,self.M - self.N, dim = 1)
                D = D.scatter(src=torch.zeros((new_dim,self.M-self.N)).to(device),dim=1,index=loss_idx)   
                D_suppp = (D == 0).to(torch.float)
                D = D.reshape(num_cout, totp).t()
        
            D_init = D.clone()
            for i_admm in range(self.max_iter):
                B = H_inv @ (GT.T-V+self.rho*D)
                if fix_supp:
                    D = ((V + self.rho * B) / self.rho) * D_fix
                elif self.N == -1:
                    D = ((V + self.rho * B) / self.rho).reshape(-1)
                    _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
                    D[loss_idx] = 0    
                    D = D.reshape(totp, num_cout)   
                else:
                    D = ((V + self.rho * B) / self.rho).t().reshape((new_dim, self.M))
                    _, loss_idx = torch.topk(-D**2,self.M - self.N, dim = 1)
                    D = D.scatter(src=torch.zeros((new_dim,self.M-self.N)).to(device),dim=1,index=loss_idx) 
                    D_supp = (D == 0).to(torch.float)  
                    D = D.reshape(num_cout, totp).t()  

                V = V + self.rho * (B - D)
                
                if (i_admm+1) % self.update_iter == 0:
                    if self.N == -1:
                        D_supp = (D.reshape(-1) == 0).to(torch.float)
                    supp_change = torch.sum((D_supp-D_suppp)**2)
                    
                    if not fix_supp:
                        if supp_change / k_spar > 0.1:
                            init_rho = True
                            self.rho *= 1.3
                        elif supp_change / k_spar > 0.005:
                            init_rho = True
                            self.rho *= 1.2
                        elif supp_change > 0.5:
                            if init_rho:
                                self.rho *= 1.1
                            else:
                                self.rho /= 5
                                B = B_orig.clone().to(device)
                                D = D_init.clone().to(device)
                                V = torch.zeros_like(B).to(device)     
                        else:
                            if init_rho:
                                break
                            else:
                                self.rho /= 5
                    
                    D_suppp = (D_supp).clone()
                    if self.rho > 1e6:
                        self.rho = 1e6
                
                    H_inv = (Q @ ((1/(L+(self.rho))) * Q).T).float().to(device)
                    
                    if self.N == -1:
                        Btest = B.reshape(-1)
                        _, loss_idx = torch.topk(-Btest**2,totp * num_cout - k_spar)
                        Btest[loss_idx] = 0    
                        Btest = Btest.reshape(totp, num_cout)
                    else:
                        Btest = B.t().reshape((new_dim, self.M))
                        _, loss_idx = torch.topk(-Btest**2,self.M - self.N, dim = 1)
                        Btest = Btest.scatter(src=torch.zeros((new_dim,self.M-self.N)).to(device),dim=1,index=loss_idx)  
                        Btest = Btest.reshape(num_cout, totp).t()
                
                    Resc = torch.matmul(H.to(device),Btest) - GT.T
                    Resc = torch.diag(torch.matmul((Btest-B_orig.to(device)).t(), Resc))
            
                    errorc = torch.sum(Resc)/Res0
                    errorc = errorc.item()
                    if i_admm >= self.switch_iter and supp_change / k_spar < 0.0003:
                        break

            if self.N == -1:
                B = B.reshape(-1)
                _, loss_idx = torch.topk(-B**2,totp * num_cout - k_spar)
                B[loss_idx] = 0    
                B = B.reshape(totp, num_cout)
            else:
                B = B.t().reshape((new_dim, self.M))
                _, loss_idx = torch.topk(-B**2,self.M - self.N, dim = 1)
                B = B.scatter(src=torch.zeros((new_dim,self.M-self.N)).to(device),dim=1,index=loss_idx)  
                B = B.reshape(num_cout, totp).t()
            
            W_mask_T = (B == 0).to(torch.bool)

            V = None
            D = None

            Res = torch.matmul(H, B) - GT.T
            Res = torch.diag(torch.matmul((B  -B_orig).t(), Res))
            
            error = torch.sum(Res)/Res0
            error = error.item()
    
            B = self.cg_batch(H, GT.T, (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10)
            Res = torch.matmul(H,B) - GT.T
            Res = torch.diag(torch.matmul((B -B_orig).t(), Res))
            
            error = torch.sum(Res)/Res0
            error = error.item()
            
            torch.cuda.synchronize()

            if isinstance(layer, transformers.Conv1D):
                layer.weight.data = (B.t() / X_norm).t().reshape(layer.weight.shape).to(layer.weight.data.dtype)
            else:
                layer.weight.data = (B.t() / X_norm).reshape(layer.weight.shape).to(layer.weight.data.dtype)
            self.W_mask[global_layer_name] = W_mask_T.T.reshape(layer.weight.shape)