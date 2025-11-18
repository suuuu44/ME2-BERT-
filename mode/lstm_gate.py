import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMGateFusion(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.W_f = nn.Linear(hidden_dim, hidden_dim)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_bert, x_edae):
        f_i = torch.sigmoid(self.W_f(x_bert))
        q_i = torch.sigmoid(self.W_q(x_edae))
        c_i = f_i * x_bert + q_i * x_edae
        o_i = torch.sigmoid(self.W_o(c_i))
        x_fused = o_i * torch.tanh(c_i)
        return x_fused   # [B,T,H]



