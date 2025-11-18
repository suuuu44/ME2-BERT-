import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionDAE(nn.Module):
    def __init__(self,
                 input_dim=768,
                 bottleneck_dim=512,
                 noise_prob=0.3,
                 pooling="mean",
                 device="cpu"):
        super().__init__()
        self.noise_prob = noise_prob
        self.device = device
        self.pooling = pooling

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim)
        )
        if pooling == "attention":
            self.attention_fc = nn.Linear(bottleneck_dim, 1)

    @staticmethod
    def masked_mean(x, mask, eps=1e-8):
        mask = mask.float().unsqueeze(-1)       # [B,T,1]
        s = (x * mask).sum(dim=1)               # [B,D]
        denom = mask.sum(dim=1).clamp_min(eps)  # [B,1]
        return s / denom

    @staticmethod
    def masked_mse(pred, target, mask, reduction='mean'):
        mask = mask.float().unsqueeze(-1)
        se = (pred - target) ** 2 * mask
        if reduction == 'mean':
            denom = mask.sum().clamp_min(1.0)
            return se.sum() / denom
        elif reduction == 'sum':
            return se.sum()
        return se

    def forward(self, bert_embeds, attention_mask,
                emo_labels=None, domain_labels=None,
                contrast_mask=None, contrast_weight=None,
                margin=0.2):
        """
        bert_embeds: [B,T,H]
        attention_mask: [B,T]
        emo_labels: [B] in {0,1,2,3,4}, 4=no_emotion
        domain_labels: [B] in {0,1}
        contrast_mask: [B] 1=参与对比, 0=跳过（建议: emo!=4）
        contrast_weight: [B] 样本权重（可为None；复现时可不使用）
        """
        # 1) 加噪
        if self.noise_prob > 0:
            keep_p = 1.0 - self.noise_prob
            noise = torch.bernoulli(keep_p * torch.ones_like(bert_embeds)).to(bert_embeds)
            noisy_input = bert_embeds * noise
        else:
            noisy_input = bert_embeds

        # 2) 编码/解码
        z = self.encoder(noisy_input)         # [B,T,D]
        recon = self.decoder(z)               # [B,T,H]

        # 3) 池化
        if self.pooling == "mean":
            z_pool = self.masked_mean(z, attention_mask)     # [B,D]
        elif self.pooling == "cls":
            z_pool = z[:, 0, :]
        elif self.pooling == "attention":
            attn = self.attention_fc(z).squeeze(-1)          # [B,T]
            attn = attn.masked_fill(attention_mask==0, -1e9)
            attn = torch.softmax(attn, dim=1).unsqueeze(-1)  # [B,T,1]
            z_pool = (z * attn).sum(dim=1)                   # [B,D]
        else:
            raise ValueError("Unknown pooling")

        # 4) 重构损失
        mse_loss = self.masked_mse(recon, bert_embeds, attention_mask, reduction='mean')

        # 5) Triplet 对比（域内、向量化、跳过 no_emotion）
        contrast_loss = torch.tensor(0.0, device=bert_embeds.device)
        if emo_labels is not None and domain_labels is not None and contrast_mask is not None:
            e = emo_labels.long()
            d = domain_labels.long()
            m = contrast_mask > 0.5
            z_norm = F.normalize(z_pool, dim=1)

            loss_parts = []
            for dom in [0, 1]:
                sel = (d == dom) & m
                if sel.sum() < 2:
                    continue
                h = z_norm[sel]  # [N,D]
                y = e[sel]  # [N]

                # 余弦相似度矩阵（不再改对角线）
                sim = h @ h.t()  # [N,N]

                # 显式去掉 self
                N = h.size(0)
                eye = torch.eye(N, dtype=torch.bool, device=h.device)

                # 正/负掩码（同簇为正，但排除 self）
                if y.numel() == 0:
                    # 如果当前batch没有有效样本，返回零损
                    return h, h, h.mean(dim=1), {"loss_edae": torch.tensor(0.0, device=h.device)}

                # === 防止空 batch 导致维度错误 ===
                if y is None or y.numel() == 0 or y.size(0) == 0:
                    # 返回空输出，不计算 E-DAE 损失
                    dummy = h.mean(dim=1)
                    return h, h, dummy, {"loss_edae": torch.tensor(0.0, device=h.device)}

                N = y.size(0)
                eye = torch.eye(N, dtype=torch.bool, device=h.device)
                pos_mask = (y.unsqueeze(1) == y.unsqueeze(0)) & (~eye)
                neg_mask = (~pos_mask) & (~eye)

                # 取每个 i 的最大正样本 & 最大负样本（或半难样本可选）
                sim_pos = sim.masked_fill(~pos_mask, -1e9).max(dim=1).values  # [N]
                sim_neg = sim.masked_fill(~neg_mask, -1e9).max(dim=1).values  # [N]

                # 某些样本可能没有正样本（单独簇），跳过
                valid = sim_pos > -1e8
                if valid.sum() == 0:
                    continue

                loss_vec = F.relu(sim_pos[valid] - sim_neg[valid] + margin)

                if contrast_weight is not None:
                    w = contrast_weight[sel][valid]
                    if w.sum() > 0:
                        loss_parts.append((loss_vec * w).sum() / w.sum())
                    else:
                        loss_parts.append(loss_vec.mean())
                else:
                    loss_parts.append(loss_vec.mean())

            if len(loss_parts) > 0:
                contrast_loss = torch.stack(loss_parts).mean()

        return recon, z, z_pool, {"mse": mse_loss, "contrast": contrast_loss}
