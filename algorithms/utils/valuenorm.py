import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """
    Value normalization module.
    """
    def __init__(self, input_shape, device=torch.device("cpu")):
        super(ValueNorm, self).__init__()
        self.device = device
        self.input_shape = input_shape
        
        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(device)
        self.running_var = nn.Parameter(torch.ones(input_shape), requires_grad=False).to(device)
        self.count = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(device)
        
        self.epsilon = 1e-8
        
    def forward(self, x):
        return x
    
    def update(self, x):
        """Update running statistics."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.numel()
        
        # Update running statistics
        delta = batch_mean - self.running_mean
        tot_count = self.count + batch_count
        
        self.running_mean.data += delta * batch_count / tot_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.running_var.data = M2 / tot_count
        self.count.data = tot_count
        
    def normalize(self, x):
        """Normalize input using running statistics."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
    
    def denormalize(self, x):
        """Denormalize input using running statistics."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        
        return x * torch.sqrt(self.running_var + self.epsilon) + self.running_mean

