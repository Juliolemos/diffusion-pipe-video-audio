import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=1.0):
        super().__init__()
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(rank, base_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(base_layer.out_features, rank) * 0.01)
        self.scaling = alpha / rank
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + (self.lora_B @ self.lora_A @ x.transpose(-1, -2)).transpose(-1, -2) * self.scaling


def inject_lora(model, rank=16, alpha=1.0, target_keywords=("attn", "proj", "mlp", "ffn")):
    for name, module in model.named_modules():
        for attr in dir(module):
            target = getattr(module, attr, None)
            if isinstance(target, nn.Linear):
                if any(k in attr.lower() or k in name.lower() for k in target_keywords):
                    setattr(module, attr, LoRALinear(target, rank=rank, alpha=alpha))
