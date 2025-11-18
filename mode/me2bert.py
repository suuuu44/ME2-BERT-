# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bert_backbone import BertBackbone
from models.e_dae import EmotionDAE
from models.lstm_gate import LSTMGateFusion
from models.domain_classifier import DomainClassifier
from models.moral_classifier import MoralClassifier


class ME2BERT(nn.Module):
    """
    ME2-BERT 整体模型
    ------------------------------------------------------------
    - 模块结构:
        1) BERT 编码器
        2) Emotion-aware Denoising Autoencoder (E-DAE)
        3) LSTM 门控融合模块
        4) Moral Classifier（多标签）
        5) Domain Classifier（带 GRL 的域对抗）
    ------------------------------------------------------------
    目标:
        L_total = L_MSE + L_contrast + L_MF + λ * L_domain
    """

    def __init__(self,
                 num_labels=5,
                 hidden_dim=768,
                 dae_bottleneck_dim=512,
                 lambda_dom=0.1,
                 bert_model_name="bert-base-uncased",
                 device="cpu",
                 use_edae=True):
        super().__init__()
        self.device = device
        self.use_edae = use_edae
        self.lambda_dom = lambda_dom

        # === 模块初始化 ===
        self.bert_backbone = BertBackbone(model_name=bert_model_name, device=device)

        # E-DAE 模块
        self.e_dae = EmotionDAE(
            input_dim=hidden_dim,
            bottleneck_dim=dae_bottleneck_dim,
            device=device
        )

        # 门控融合层
        self.lstm_gate = LSTMGateFusion(hidden_dim=hidden_dim)

        # 域分类器（根据 use_edae 决定输入维度）
        self.domain_classifier = DomainClassifier(
            input_dim=dae_bottleneck_dim if self.use_edae else hidden_dim
        )

        # 道德分类器
        self.moral_classifier = MoralClassifier(
            input_dim=hidden_dim,
            hidden_dim=256,
            num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask,
                labels_moral=None, labels_domain=None, labels_emotion=None,
                contrast_mask=None, contrast_weight=None,
                alpha=1.0, margin=0.2):
        """
        Args:
            input_ids, attention_mask: BERT tokenizer 输出
            labels_moral: [B, num_labels]
            labels_domain: [B] 或 [B, 1]
            labels_emotion: [B] 0..4
            contrast_mask: [B] 1=参与对比, 0=跳过 (如 no_emotion)
            contrast_weight: [B] 对比样本权重
            alpha: GRL 系数 (动态调整)
            margin: Triplet 对比损失 margin
        Returns:
            total_loss: 标量
            outputs: dict
        """
        use_edae = self.use_edae  # ✅ 统一来源

        # === 1️⃣ BERT 表示 ===
        x_bert_seq, _ = self.bert_backbone(input_ids, attention_mask)  # [B, T, H]

        # === 2️⃣ E-DAE 重构 & 对比学习 ===
        if labels_domain is not None and labels_domain.dim() == 2:
            labels_domain = labels_domain.squeeze(1)

        if use_edae:
            x_recon, bottleneck_seq, bottleneck_pooled, loss_edae_dict = self.e_dae(
                bert_embeds=x_bert_seq,
                attention_mask=attention_mask,
                emo_labels=labels_emotion,
                domain_labels=labels_domain.long() if labels_domain is not None else None,
                contrast_mask=contrast_mask,
                contrast_weight=contrast_weight,
                margin=margin
            )
            loss_edae = sum(loss_edae_dict.values())  # L_MSE + L_contrast
        else:
            # 🔹 如果不使用 E-DAE，则不进行重构，loss_edae 设为 0
            x_recon = None
            bottleneck_pooled = x_bert_seq.mean(dim=1)
            loss_edae = torch.tensor(0.0, device=x_bert_seq.device)

        # === 3️⃣ 门控融合（根据 use_edae 控制）
        if use_edae:
            # 🔹 使用 E-DAE 输出进行门控融合
            x_fused = self.lstm_gate(x_bert_seq, x_recon)  # [B, T, H]
        else:
            # 🔹 Baseline：直接使用 BERT 表示（不融合）
            x_fused = x_bert_seq

        # === 4️⃣ 道德分类 (L_MF) ===
        logits_moral, probs_moral = self.moral_classifier(x_fused, attention_mask=attention_mask)
        loss_moral = None
        if labels_moral is not None:
            loss_moral = F.binary_cross_entropy_with_logits(logits_moral, labels_moral.float())

        # === 5️⃣ 域分类 (GRL + BCE) ===
        if use_edae:
            dom_input = bottleneck_pooled  # [B,512]
        else:
            dom_input = x_bert_seq.mean(dim=1)  # [B,768]

        logits_domain = self.domain_classifier(dom_input, alpha=alpha)
        loss_adv = None
        if labels_domain is not None:
            loss_adv = F.binary_cross_entropy_with_logits(
                logits_domain.view(-1), labels_domain.float().view(-1)
            )

        # === 6️⃣ 总损失 ===
        total_loss = (
                (loss_moral or 0)
                + self.lambda_dom * (loss_adv or 0)
                + (loss_edae or 0)
        )

        return total_loss, {
            "moral_probs": torch.sigmoid(logits_moral).detach(),
            "domain_probs": torch.sigmoid(logits_domain).detach(),
            "loss_edae": float(loss_edae.item()) if torch.is_tensor(loss_edae) else 0.0,
            "loss_moral": float(loss_moral.item()) if loss_moral is not None else 0.0,
            "loss_adv": float(loss_adv.item()) if loss_adv is not None else 0.0,
        }


# === 测试 ===
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    texts = ["This is a source domain example.", "This is a target domain example."]
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=12, return_tensors="pt")

    labels_moral = torch.randint(0, 2, (2, 5)).float()
    labels_domain = torch.randint(0, 2, (2,))
    labels_emotion = torch.randint(0, 5, (2,))
    contrast_mask = (labels_emotion != 4).float()

    model = ME2BERT(num_labels=5, device="cpu")
    total_loss, outputs = model(
        encodings["input_ids"], encodings["attention_mask"],
        labels_moral=labels_moral,
        labels_domain=labels_domain,
        labels_emotion=labels_emotion,
        contrast_mask=contrast_mask,
        alpha=1.0
    )

    print(f"总损失: {total_loss.item() if total_loss is not None else None:.4f}")
    print("L_MSE+Contrast:", outputs["loss_edae"])
    print("L_MF:", outputs["loss_moral"])
    print("L_Domain:", outputs["loss_adv"])
