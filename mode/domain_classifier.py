import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    @staticmethod
    def masked_mean(x, mask, eps=1e-8):
        m = mask.float().unsqueeze(-1)
        s = (x * m).sum(dim=1)
        d = m.sum(dim=1).clamp_min(eps)
        return s / d

    def forward(self, x, alpha=1.0, attention_mask=None):
        # x: [B,D] 或 [B,T,D]
        if x.dim() == 3:
            if self.pooling == "mean":
                if attention_mask is None:
                    x = x.mean(dim=1)
                else:
                    x = self.masked_mean(x, attention_mask)
            elif self.pooling == "cls":
                x = x[:, 0, :]
            else:
                raise ValueError("Unknown pooling")

        x_reversed = GradientReversalLayer.apply(x, alpha)
        h = F.relu(self.fc1(x_reversed))
        logits = self.fc2(h)  # [B,1]
        return logits

    def compute_loss(self, logits, domain_labels):
        return F.binary_cross_entropy_with_logits(
            logits.view(-1),
            domain_labels.float().view(-1)
        )


