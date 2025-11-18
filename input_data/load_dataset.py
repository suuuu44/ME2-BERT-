# -*- coding: utf-8 -*-
"""
读取 preprocess_moral_events_lex.py 产出的 CSV：
- 安全解析 event，并按“有无事件”设 domain（冗余校验）
- 富文本：原文 + 事件描述（不丢原文）
- 按 article_id 分组切分（防泄漏）
- 8情绪 -> 4对 -> 5簇 emotion_labels（对比学习簇）
- 生成 contrast_mask (no_emotion 跳过) 与 contrast_weight（簇强度）
- 构建 Dataset、BatchSampler（仅采样 0..3 簇给对比学习）、collate_fn
"""
import ast
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer
from sklearn.model_selection import GroupShuffleSplit

# --- 配置 ---
MORAL5 = ['care','fairness','loyalty','authority','purity']
EMO8 = ['anger','fear','joy','sadness','surprise','disgust','anticipation','trust']


# --- 安全解析 ---
def safe_parse_event(x):
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "nan":
            return []
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    return x if isinstance(x, list) else []


# --- 富文本（原文 + 事件描述追加） ---
def enrich_text(row):
    base = row['text'] if isinstance(row['text'], str) else ""
    evs = row['event'] or []
    if len(evs) == 0:
        return base
    desc = []
    for i, ev in enumerate(evs):
        mention = ev.get('mention') or ev.get('trigger','')
        ents = ev.get('entities', {})
        if ents:
            ent_str = ", ".join([f"{e} ({r})" for e, r in ents.items()])
            desc.append(f"[Event {i}] '{mention}' involving {ent_str}.")
        else:
            desc.append(f"[Event {i}] '{mention}'.")
    return (base + " " + " ".join(desc)).strip()


# --- 8情绪 -> 4对 -> (cluster, strength, is_noemo) ---
def emotions_to_cluster_and_strength(row, thresh_noemo=0.2):
    """
    返回 (cluster_id, strength, is_noemo)
      cluster: 0..3 四对；4 表示 no_emotion
      strength: 顶层对的强度(用于对比权重)
    """
    pairs = {
        0: max(float(row.get('anger', 0.0)), float(row.get('fear', 0.0))),
        1: max(float(row.get('trust', 0.0)), float(row.get('disgust', 0.0))),
        2: max(float(row.get('joy', 0.0)), float(row.get('sadness', 0.0))),
        3: max(float(row.get('surprise', 0.0)), float(row.get('anticipation', 0.0))),
    }
    top_pair = max(pairs, key=pairs.get)
    top_val = float(pairs[top_pair])
    is_noemo = bool(row.get('no_emotion', 0.0) == 1.0) or (top_val < thresh_noemo)
    return (4 if is_noemo else top_pair), (0.0 if is_noemo else top_val), is_noemo


# --- Dataset ---
class ME2Dataset(Dataset):
    """
    enc_list: 由 tokenizer(..., padding='max_length') 得到的编码（每条为 dict）
    y_moral:  (N, 5)
    y_domain: (N,) 0/1
    clusters: (N,) in {0..4}
    c_strength: (N,) float
    is_noemo: (N,) 0/1
    """
    def __init__(self, enc_list, y_moral, y_domain, clusters, c_strength, is_noemo):
        self.enc_list = enc_list
        self.y_moral  = y_moral
        self.y_domain = y_domain.astype(np.float32)
        self.clusters = clusters.astype(np.int64)
        self.c_strength = c_strength.astype(np.float32)
        self.is_noemo = is_noemo.astype(np.int64)

    def __len__(self):
        return len(self.enc_list["input_ids"])

    def __getitem__(self, idx):
        # HuggingFace 批量编码返回 dict(list/np.ndarray)
        x = {
            "input_ids": torch.tensor(self.enc_list["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.enc_list["attention_mask"][idx], dtype=torch.long),
        }
        if "token_type_ids" in self.enc_list:
            x["token_type_ids"] = torch.tensor(self.enc_list["token_type_ids"][idx], dtype=torch.long)

        item = {
            "input_ids": x["input_ids"],
            "attention_mask": x["attention_mask"],
            "mf_labels": torch.tensor(self.y_moral[idx], dtype=torch.float),
            "domain_labels": torch.tensor(self.y_domain[idx], dtype=torch.float),    # [ ]
            "emotion_labels": torch.tensor(self.clusters[idx], dtype=torch.long),    # 0..4
            "contrast_mask": torch.tensor(0 if self.is_noemo[idx] else 1, dtype=torch.float),
            "contrast_weight": torch.tensor(self.c_strength[idx], dtype=torch.float),
        }
        if "token_type_ids" in x:
            item["token_type_ids"] = x["token_type_ids"]
        return item


# --- BatchSampler：仅采样 0..3 簇给对比学习 ---
class EmotionClusterBatchSampler(Sampler):
    def __init__(self, cluster_labels, batch_size):
        self.cluster_to_indices = defaultdict(list)
        for i, c in enumerate(cluster_labels):
            c = int(c)
            if c == 4:  # no_emotion 不进对比采样器
                continue
            self.cluster_to_indices[c].append(i)
        self.clusters = list(self.cluster_to_indices.keys())
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = []
            random.shuffle(self.clusters)
            for c in self.clusters:
                if self.cluster_to_indices[c]:
                    batch.append(random.choice(self.cluster_to_indices[c]))
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
            if len(batch) == 0:
                break

    def __len__(self):
        total = sum(len(v) for v in self.cluster_to_indices.values())
        return max(1, total // self.batch_size)


# --- collate_fn（已统一长度 → 直接 stack；同时兼容存在/不存在的字段） ---
def collate_fn(batch):
    out = {}

    # 这些键在 encode 阶段已统一到相同长度（padding='max_length'）
    for k in ("input_ids", "attention_mask", "token_type_ids"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)

    # 标签
    out["mf_labels"]      = torch.stack([b["mf_labels"] for b in batch]).float()        # [B,5]
    out["domain_labels"]  = torch.stack([b["domain_labels"] for b in batch]).float()    # [B]
    out["emotion_labels"] = torch.stack([b["emotion_labels"] for b in batch]).long()    # [B]

    # 对比掩码/权重（若不存在则给默认值）
    if "contrast_mask" in batch[0]:
        out["contrast_mask"] = torch.stack([b["contrast_mask"] for b in batch]).float()
    else:
        out["contrast_mask"] = torch.ones(len(batch), dtype=torch.float)

    if "contrast_weight" in batch[0]:
        out["contrast_weight"] = torch.stack([b["contrast_weight"] for b in batch]).float()
    else:
        out["contrast_weight"] = torch.ones(len(batch), dtype=torch.float)

    return out


# --- 主入口：读取CSV并构建数据集 ---
def load_dataset(csv_path,
                 pretrained_model="bert-base-uncased",
                 max_len=128,
                 seed=42,
                 train_size=0.8,
                 val_size=0.1,
                 thresh_noemo=0.2):
    """
    返回:
      {
        "train": Dataset,
        "val": Dataset,
        "test": Dataset,
        "train_clusters": np.ndarray,
        "val_clusters": np.ndarray,
        "test_clusters": np.ndarray,
      }
    """
    df = pd.read_csv(csv_path)

    # 解析 event，冗余确认 domain
    df["event"] = df["event"].apply(safe_parse_event)
    df["domain"] = df["event"].apply(lambda ev: 1 if len(ev) > 0 else 0)

    # 富文本
    df["e_text"] = df.apply(enrich_text, axis=1)

    # 8情绪 -> (cluster, strength, is_noemo)
    df["cluster"], df["cluster_strength"], df["is_noemo"] = zip(
        *df.apply(lambda r: emotions_to_cluster_and_strength(r, thresh_noemo=thresh_noemo), axis=1)
    )

    # 按文章分组切分（防泄漏）
    if "article_id" in df.columns:
        groups = df["article_id"].astype(str).values
    else:
        # 若没有 article_id 列，退化为按索引分组
        groups = df.index.astype(str).values

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, hold_idx = next(gss1.split(df, groups=groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_hold  = df.iloc[hold_idx].reset_index(drop=True)

    # 再把 hold 切 val/test
    hold_groups = df_hold["article_id"].astype(str).values if "article_id" in df_hold.columns else df_hold.index.astype(str).values
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size/(1-train_size), random_state=seed)
    val_idx, test_idx = next(gss2.split(df_hold, groups=hold_groups))
    df_val  = df_hold.iloc[val_idx].reset_index(drop=True)
    df_test = df_hold.iloc[test_idx].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, use_fast=True)

    # === 批量编码（统一长度）===
    def encode_texts(text_list):
        enc = tokenizer(
            [t if isinstance(t, str) else "" for t in text_list],
            truncation=True,
            padding="max_length",   # ✅ 关键：统一长度，彻底避免 DataLoader 拼接报错
            max_length=max_len,
            return_attention_mask=True,
        )
        # enc: dict('input_ids': list[list[int]], 'attention_mask': list[list[int]], [maybe 'token_type_ids'])
        return enc

    def pack(df_part):
        enc = encode_texts(df_part["e_text"].tolist())
        y_m = df_part[MORAL5].values.astype(np.float32)
        y_d = df_part["domain"].values.astype(np.float32)
        clusters   = df_part["cluster"].values
        c_strength = df_part["cluster_strength"].values
        is_noemo   = df_part["is_noemo"].astype(int).values
        return enc, y_m, y_d, clusters, c_strength, is_noemo

    tr = pack(df_train)
    va = pack(df_val)
    te = pack(df_test)

    ds_train = ME2Dataset(*tr)
    ds_val   = ME2Dataset(*va)
    ds_test  = ME2Dataset(*te)

    return {
        "train": ds_train,
        "val":   ds_val,
        "test":  ds_test,
        "train_clusters": tr[3],  # for sampler
        "val_clusters":   va[3],
        "test_clusters":  te[3],
    }
