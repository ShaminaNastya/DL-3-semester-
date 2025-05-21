import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x_norm = x / rms
        return self.scale * x_norm

  from torch.nn import RMSNorm as TorchRMSNorm

x = torch.randn(2, 5, 10)
my_norm = RMSNorm(10)
torch_norm = TorchRMSNorm(10)

# синхрониз парычи для настоящего сравнения
with torch.no_grad():
    torch_norm.weight.copy_(my_norm.scale)

out_my = my_norm(x)
out_torch = torch_norm(x)

print(torch.allclose(out_my, out_torch, atol=1e-6))  # д б True
