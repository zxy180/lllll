import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel, Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import GammaPrior, UniformPrior
from gpytorch.distributions import MultivariateNormal
import numpy as np
from typing import Optional, List
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HammingARDKernel(Kernel):
    """
    with Automatic Relevance Determination (ARD)
    """
    def __init__(self, seq_length: int, ard_num_dims: Optional[int] = None, batch_shape=torch.Size([]), active_dims=None):
        super().__init__(batch_shape=batch_shape, active_dims=active_dims)
        self.seq_length = seq_length
        self.extended_length = seq_length
        
        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1, self.extended_length) * 0.5)
        )
        
        self.register_constraint(
            "raw_lengthscale",
            Interval(1.0, 2.0)
        )
        
        set_seed(42)
        with torch.no_grad():
            raw_vals = self.raw_lengthscale
            init_vals = 0.5 + 1.5 * torch.rand(self.extended_length)
            self.raw_lengthscale[0, :] = torch.log(init_vals)
    
    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)
    
    @lengthscale.setter
    def lengthscale(self, value):
        self._set_lengthscale(value)
    
    def forward(self, x1, x2, diag=False, **params):
        n1 = x1.size(0)
        n2 = x2.size(0)
        lengthscale = self.lengthscale
        block_size = 100
        
        if diag:
            if n1 != n2:
                raise ValueError("diag=True requires n1 == n2")
            result = torch.zeros(n1, device=x1.device)
            for i in range(0, n1, block_size):
                end = min(i + block_size, n1)
                x1_block = x1[i:end]
                x2_block = x2[i:end]
                
                x1_exp = x1_block.unsqueeze(1)
                x2_exp = x2_block.unsqueeze(0)
                diff_mask = (x1_exp != x2_exp).float()
                block_res = torch.exp(-diff_mask / lengthscale).prod(dim=-1)
                result[i:end] = block_res.diag()
                
                del diff_mask, block_res
                torch.cuda.empty_cache()
            return result
        
        rows = []
        for i in range(0, n1, block_size):
            i_end = min(i + block_size, n1)
            x1_block = x1[i:i_end]
            
            cols = []
            for j in range(0, n2, block_size):
                j_end = min(j + block_size, n2)
                x2_block = x2[j:j_end]
                
                x1_exp = x1_block.unsqueeze(1)
                x2_exp = x2_block.unsqueeze(0)
                diff_mask = (x1_exp != x2_exp).float()
                
                block_res = torch.exp(-diff_mask / lengthscale).prod(dim=-1)
                
                cols.append(block_res)
                del diff_mask
            
            rows.append(torch.cat(cols, dim=1))
            torch.cuda.empty_cache()
        
        return torch.cat(rows, dim=0)

class ScaleHammingKernel(ScaleKernel):
    """
    带输出尺度的汉明核
    """
    def __init__(self, seq_length: int, ard_num_dims: Optional[int] = None, batch_shape=torch.Size([])):
        base_kernel = HammingARDKernel(seq_length=seq_length, ard_num_dims=ard_num_dims, batch_shape=batch_shape)
        super().__init__(
            base_kernel,
            outputscale_constraint=Interval(1e-2, 1e2),
            batch_shape=batch_shape
        )

class TextKernel(gpytorch.kernels.Kernel):
    def __init__(self, train_x, custom_device="cuda:0", predict_mode=False):
        super().__init__()
        self._custom_device = custom_device
        self.predict_mode = predict_mode
        
        self.rbf = ScaleKernel(
            RBFKernel(ard_num_dims=1024),
            outputscale_constraint=Interval(0.5, 5.0)
        ).to(self._custom_device)
        
        with torch.no_grad():
            self.rbf.outputscale = torch.tensor(1.0, device=self._custom_device)
            
        self.matern = MaternKernel(nu=1.5, ard_num_dims=1024).to(self._custom_device)
        
        print("需要得到的lengthscale参数！！！！！！！！！！！！！！！", train_x.shape[1]-1024)
        self.hamming = ScaleHammingKernel(seq_length=train_x.shape[1]-1024).to(self._custom_device)
        
        self.rbf_weight = torch.nn.Parameter(torch.tensor(0.99, device=self._custom_device))
        self.matern_weight = torch.nn.Parameter(torch.tensor(0.0, device=self._custom_device))
        self.hamming_weight = torch.nn.Parameter(torch.tensor(0.01, device=self._custom_device))

    def forward(self, x1, x2, diag=False, **params):
        x1 = x1.to(self._custom_device)
        x2 = x2.to(self._custom_device)
        
        x1_c = x1[:, :1024]
        x2_c = x2[:, :1024]
        
        x1_d = x1[:, 1024:]
        x2_d = x2[:, 1024:]
        
        if self.predict_mode:
            combined = self.rbf(x1_c, x2_c, diag=diag, **params)
        else:
            raw_weights = torch.stack([
                self.rbf_weight, 
                self.matern_weight, 
                self.hamming_weight
            ])
            weights = torch.softmax(raw_weights, dim=0)
            
            rbf_out = self.rbf(x1_c, x2_c, diag=diag, **params) * weights[0]
            matern_out = self.matern(x1_c, x2_c, diag=diag, **params) * weights[1]
            hamming_out = self.hamming(x1_d, x2_d, diag=diag, **params) * weights[2]
            
            combined = rbf_out + matern_out + hamming_out
        
        if x1.shape[-2] == x2.shape[-2] and not diag:
            combined = combined.add_jitter(1e-6)
            
        return combined
        
    def get_hamming_lengthscale(self):
        return self.hamming.base_kernel.lengthscale.squeeze()

class SimpleGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, custom_device="cuda:0", predict_mode=False):
        super().__init__(train_x, train_y, likelihood)
        self._custom_device = custom_device
        print("debug!!!!!!!!!!")
        self.mean_module = gpytorch.means.ConstantMean().to(self._custom_device)
        
        self.covar_module = TextKernel(
            train_x=train_x,
            custom_device=self._custom_device,
            predict_mode=predict_mode
        )
    
    def forward(self, x):
        x = x.to(self._custom_device)
        mean_x = self.mean_module(x[:, :1024])
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_vulnerable_positions(self, topK=5, temperature=2.0, noise_level=0.01):
        lengthscale = self.covar_module.get_hamming_lengthscale()
        return get_position_importance_with_all_gaps(
            lengthscale, 
            topK=topK, 
            temperature=temperature, 
            noise_level=noise_level
        )

def get_position_importance_with_all_gaps(
    lengthscale: torch.Tensor,
    topK: int = 5,
    temperature: float = 2.0,
    noise_level: float = 0.01
) -> List[int]:
    
    print(f"\n--- get_position_importance_with_all_gaps (扩展版) ---")
    print(lengthscale)
    
    lengthscale_np = lengthscale.detach().cpu().numpy()
    if len(lengthscale_np.shape) > 1:
        lengthscale_np = lengthscale_np.squeeze()
    
    print(f"lengthscale stats: min={lengthscale_np.min():.4f}, max={lengthscale_np.max():.4f}, "
          f"mean={lengthscale_np.mean():.4f}, std={lengthscale_np.std():.4f}")

    top_values, top_indices = torch.topk(lengthscale, k=topK, dim=0, largest=False, sorted=True)
    top_positions = []
    
    print(f"Top-{topK} 最小的lengthscale值：")
    for i in range(topK):
        idx = top_indices[i].item()
        top_positions.append(idx)
        val = top_values[i].item()
        print(f"排名{i+1} | 索引：{idx} | 值：{val:.4f}")
    
    return top_positions
