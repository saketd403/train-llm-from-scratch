import torch
import torch.nn as nn
import math

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha, dtype=torch.float32):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, dtype
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, dtype=torch.float32):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # similar to standard weight initialization
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim, dtype=dtype))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


def replace_linear_with_lora(model, rank, alpha,dtype=torch.float32):

    for name, module in model.named_children():
        
        if isinstance(module, torch.nn.Linear):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, rank, alpha, dtype))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, rank, alpha, dtype)