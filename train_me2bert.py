# -*- coding: utf-8 -*-
"""
ME2-BERT 复现训练脚本（支持 sample_train/sample_test 子集采样 + F1 评估）
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from models.me2bert import ME2BERT
from input_data.load_dataset import load_dataset

# ================================
# 设备检测
# ================================
try:
    import torch_directml
    device = torch_directml.device()
    print("✅ Using DirectML GPU for acceleration!")
except ImportError:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ Using CUDA GPU!")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no GPU backend detected)")

# ================================
# 动态调度函数
# ================================
def schedule_margin(init_margin, min_margin, epoch, max_epoch):
    decay_rate = 0.95
    return max(min_margin, init_margin * (decay_rate ** epoch))


def schedule_alpha(epoch, max_epoch):
    p = epoch / max(1, max_epoch)
    return 2. / (1. + np.exp(-10 * p)) - 1.0  # 从 -10 改成 -5



# ================================
# F1 评估函数
# ================================
@torch.no_grad()
def evaluate_f1_per_class(model, dataloader, threshold=0.3, device="cpu"):
    model.eval()
    all_preds, all_true = [], []
    for batch in dataloader:
        _, outputs = model(batch["input_ids"].to(device),
                           batch["attention_mask"].to(device),
                           alpha=1.0)
        probs = outputs["moral_probs"].cpu().numpy()
        preds = (probs > threshold).astype(int)
        labels = batch["mf_labels"].cpu().numpy().astype(int)
        all_preds.extend(preds)
        all_true.extend(labels)
    all_preds, all_true = np.array(all_preds), np.array(all_true)
    names = ["Care", "Fairness", "Loyalty", "Authority", "Purity"]
    f1s = [f1_score(all_true[:, i], all_preds[:, i], zero_division=0)
           for i in range(all_true.shape[1])]
    return dict(zip(names + ["AVG"], f1s + [np.mean(f1s)]))


# ================================
# 案例推理函数
# ================================
@torch.no_grad()
def predict_case(model, tokenizer, texts, device="cpu"):
    model.eval()
    results = []
    for txt in texts:
        enc = tokenizer(txt, truncation=True, padding="max_length",
                        max_length=128, return_tensors="pt").to(device)
        _, outputs = model(enc["input_ids"], enc["attention_mask"], alpha=1.0)
        results.append({
            "text": txt,
            "moral_probs": outputs["moral_probs"].cpu().numpy().tolist()[0],
            "domain_probs": outputs["domain_probs"].cpu().numpy().tolist()[0]
        })
    return results


# ================================
# 主训练函数
# ================================
def train(csv_path="data/moral_events_lex.csv",
          epochs=3,
          batch_size=8,
          lr=3e-5,
          init_margin=0.3,
          min_margin=0.05,
          f1_threshold=0.3,
          sample_train=None,
          sample_val=None,
          sample_test=None,
          out_dir="runs_me2bert_final",
          use_edae=True):  # 👈 新增参数

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # === 数据加载 ===
    datasets = load_dataset(csv_path)
    train_ds = datasets["train"]
    val_ds = datasets["val"]
    test_ds = datasets["test"]

    # === 采样逻辑（原始稳定版） ===
    if sample_train is not None and sample_train < len(train_ds):
        train_ds = Subset(train_ds, range(sample_train))
    if sample_val is not None and sample_val < len(val_ds):
        val_ds = Subset(val_ds, range(sample_val))
    if sample_test is not None and sample_test < len(test_ds):
        test_ds = Subset(test_ds, range(sample_test))

    print(f"📊 训练集: {len(train_ds)} | 验证集: {len(val_ds)} | 测试集: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # === 模型初始化 ===
    #model = ME2BERT(num_labels=5, device=device).to(device)
    # === 模型初始化 ===
    model = ME2BERT(num_labels=5, device=device, use_edae=use_edae).to(device)

    weight_decay = 0.01
    warmup_ratio = 0.1

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # === 示例句子 ===
    demo_texts = ["The government should help the poor.", "People must obey the law."]
    before_cases = predict_case(model, tokenizer, demo_texts, device=device)

    logs = []
    best_val_f1 = 0
    best_path = os.path.join(out_dir, "best.pt")

    # === 训练循环 ===
    for epoch in range(epochs):
        model.train()
        total_loss, total_edae, total_mf, total_adv = [], [], [], []

        margin = schedule_margin(init_margin, min_margin, epoch, epochs)
        alpha = schedule_alpha(epoch, epochs)

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_moral = batch["mf_labels"].to(device)
            labels_domain = batch["domain_labels"].to(device).float()
            labels_emotion = batch["emotion_labels"].to(device).squeeze(-1).long()

            contrast_mask = (labels_emotion != 4).float()

            loss, outputs = model(
                input_ids, attention_mask,
                labels_moral=labels_moral,
                labels_domain=labels_domain,
                labels_emotion=labels_emotion,
                contrast_mask=contrast_mask,
                alpha=alpha, margin=margin
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss.append(loss.item())
            total_edae.append(outputs["loss_edae"])
            total_mf.append(outputs["loss_moral"])
            total_adv.append(outputs["loss_adv"])

        # ✅ Epoch 结束后计算验证
        avg_loss = np.mean(total_loss)
        print(f"\nEpoch {epoch + 1}/{epochs} | Loss {avg_loss:.4f} | Margin {margin:.3f} | Alpha {alpha:.3f}")

        val_f1 = evaluate_f1_per_class(model, val_loader, threshold=f1_threshold, device=device)
        print(f"Val F1: {val_f1}")

        logs.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "margin": margin,
            "alpha": alpha,
            **{f"val_{k}": v for k, v in val_f1.items()}
        })

        if val_f1["AVG"] > best_val_f1:
            best_val_f1 = val_f1["AVG"]
            torch.save(model.state_dict(), best_path)
            print(f"✅ Saved best (Val AVG F1={best_val_f1:.4f}) → {best_path}")

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # === 测试评估 ===
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_f1 = evaluate_f1_per_class(model, test_loader, threshold=f1_threshold, device=device)
    after_cases = predict_case(model, tokenizer, demo_texts, device=device)

    print("\n📊 Test F1:")
    for k, v in test_f1.items():
        print(f"  {k}: {v:.4f}")

    # === 保存结果 ===
    pd.DataFrame(logs).to_csv(os.path.join(out_dir, "train_log.csv"), index=False)
    pd.DataFrame([test_f1]).to_csv(os.path.join(out_dir, "test_f1.csv"), index=False)
    with open(os.path.join(out_dir, "cases_before.json"), "w", encoding="utf-8") as f:
        json.dump(before_cases, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "cases_after.json"), "w", encoding="utf-8") as f:
        json.dump(after_cases, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已保存结果到 {out_dir}")


# ================================
# 程序入口
# ================================
if __name__ == "__main__":
    # 🔘 开关：是否使用情绪增强模块
    USE_EDAE = False  # 改为 False 即 baseline

    train(
        csv_path="data/moral_events_lex.csv",
        epochs=5,
        batch_size=8,
        lr=8e-6,
        init_margin=0.3,
        min_margin=0.05,
        f1_threshold=0.1,
        sample_train=3000,
        sample_val=1000,
        sample_test=1000,
        out_dir="runs_me2bert_final_5e" if USE_EDAE else "runs_me2bert_base_5e",
        use_edae=USE_EDAE
    )
