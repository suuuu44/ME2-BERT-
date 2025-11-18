import torch
import torch.nn as nn
import torch.nn.functional as F

class MoralClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_labels=5, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    @staticmethod
    def masked_mean(x, mask, eps=1e-8):
        # x: [B,T,D], mask: [B,T] (1=valid,0=pad)
        m = mask.float().unsqueeze(-1)
        s = (x * m).sum(dim=1)
        d = m.sum(dim=1).clamp_min(eps)
        return s / d

    def forward(self, x, attention_mask=None):
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
                raise ValueError(f"Unknown pooling method: {self.pooling}")

        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)
        probs = torch.sigmoid(logits)
        return logits, probs

    def compute_loss(self, logits, labels, pos_weight: torch.Tensor | None = None):
        # labels: [B,num_labels] float/0-1
        if pos_weight is not None:
            return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
        return F.binary_cross_entropy_with_logits(logits, labels.float())
