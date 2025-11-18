# -*- coding: utf-8 -*-
"""
MoralEvents JSON -> CSV（词典法情绪）
- 若未提供 --nrc_lex_path，将自动在常见目录查找；若无则尝试下载到 ./data/lexicons/。
输出列：
  article_id, para_id, text, event(json), domain(0/1),
  care, fairness, loyalty, authority, purity,
  anger, fear, joy, sadness, surprise, disgust, anticipation, trust,
  no_emotion
"""
import os
import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
from tqdm import tqdm

# -----------------------------
# 自动定位 / 下载 NRC 词典
# -----------------------------
CANDIDATE_URLS = [
    # 这两个镜像可能失效；脚本会提示手动放置
    "https://raw.githubusercontent.com/ishikota/NRC-Emotion-Lexicon-Wordlevel/master/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
    "https://raw.githubusercontent.com/wwbp/lexica/master/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
]

DEFAULT_LEX_DIR = Path("./data/lexicons")
DEFAULT_LEX_NAME = "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

def ensure_lexicon(lex_path_arg: str | None) -> Path:
    """优先使用传入路径；否则在常见目录寻找；再不行尝试下载。"""
    if lex_path_arg:
        p = Path(lex_path_arg)
        if p.is_file():
            return p
        raise FileNotFoundError(f"指定的 lexicon 不存在: {p}")

    candidates = [
        DEFAULT_LEX_DIR / DEFAULT_LEX_NAME,
        Path("./") / DEFAULT_LEX_NAME,
    ]
    for c in candidates:
        if c.is_file():
            return c

    DEFAULT_LEX_DIR.mkdir(parents=True, exist_ok=True)
    dst = DEFAULT_LEX_DIR / DEFAULT_LEX_NAME
    try:
        import requests
        for url in CANDIDATE_URLS:
            try:
                print(f"⬇️  尝试下载 NRC 词典: {url}")
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.text) > 1024:
                    with open(dst, "w", encoding="utf-8") as f:
                        f.write(r.text)
                    print(f"✅ 已下载: {dst}")
                    return dst
                else:
                    print(f"⚠️ 下载失败/内容异常（status={r.status_code}），尝试下一个镜像...")
            except Exception as e:
                print(f"⚠️ 下载出错: {e}，尝试下一个镜像...")
        raise RuntimeError("所有镜像均未成功。请手动放置词典文件。")
    except ImportError:
        raise RuntimeError(
            "未找到 NRC 词典，且未安装 requests 无法自动下载。\n"
            f"请手动下载 {DEFAULT_LEX_NAME} 到 {DEFAULT_LEX_DIR.resolve()}，或通过 --nrc_lex_path 指定路径。"
        )

# -----------------------------
# 词典读取 & 打分
# -----------------------------
EMO8 = ['anger','fear','joy','sadness','surprise','disgust','anticipation','trust']

def load_nrc_lexicon(lex_path, lowercase=True):
    """支持官方 .txt（三列：word emotion assoc）。若是 .csv，需含 word + 8情绪列。"""
    lex = defaultdict(lambda: {e: 0.0 for e in EMO8})
    lex_path = Path(lex_path)
    if lex_path.suffix.lower() == ".csv":
        df = pd.read_csv(lex_path)
        cols = {c.lower().strip(): c for c in df.columns}
        word_col = cols.get('word', None)
        if word_col is None:
            raise ValueError("CSV 需包含 'word' 列")
        for e in EMO8:
            if e not in cols:
                raise ValueError(f"CSV 缺少情绪列: {e}")
        for _, row in df.iterrows():
            w = str(row[word_col])
            if lowercase: w = w.lower()
            for e in EMO8:
                val = float(row[cols[e]])
                if val > 0:
                    lex[w][e] = val
    else:
        with open(lex_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                w, emo, assoc = parts[0], parts[1].lower(), parts[2]
                if emo not in EMO8:
                    continue
                if lowercase: w = w.lower()
                try:
                    a = float(assoc)
                except:
                    a = 0.0
                if a > 0:
                    lex[w][emo] = a
    return dict(lex)

def score_emotions_by_lex(text, lex, lowercase=True):
    """最简词频加总→总和归一化；可替换为更复杂的清洗/词形还原。"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return {e: 0.0 for e in EMO8}
    t = text.lower() if lowercase else text
    tokens = []
    for w in t.split():
        w = ''.join([ch for ch in w if ch.isalpha()])
        if w:
            tokens.append(w)
    if not tokens:
        return {e: 0.0 for e in EMO8}
    counts = Counter()
    for w in tokens:
        if w in lex:
            for e, sc in lex[w].items():
                counts[e] += sc
    total = sum(counts.values())
    if total <= 0:
        return {e: 0.0 for e in EMO8}
    return {e: float(counts[e]) / total for e in EMO8}

# -----------------------------
# 道德标签 10->5 合并
# -----------------------------
MORAL_POS = ['care','fairness','loyalty','authority','purity']
MORAL_VICE = {
    'care': 'harm',
    'fairness': 'cheating',
    'loyalty': 'betrayal',
    'authority': 'subversion',
    'purity': 'degradation'
}
def morals_to_5pos(moral_labels):
    y = {m: 0.0 for m in MORAL_POS}
    for lab in moral_labels:
        if not lab:
            continue
        s = str(lab).strip().lower()
        if s in MORAL_POS:
            y[s] = 1.0
        else:
            for pos, neg in MORAL_VICE.items():
                if s == neg:
                    y[pos] = 1.0
    return y

# -----------------------------
# 主流程
# -----------------------------
def main(input_dir, output_csv, nrc_lex_path=None):
    lex_file = ensure_lexicon(nrc_lex_path)
    print(f"🧩 使用 NRC 词典: {lex_file}")
    lex = load_nrc_lexicon(lex_file, lowercase=True)
    print(f"✅ Loaded lexicon: {len(lex)} words")

    rows = []
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"📂 Found {len(files)} JSON documents")

    for file in tqdm(files, desc="Parsing JSON"):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        article_id = Path(file).stem
        news_paragraphs = {n['id']: (n.get('text','') or '').strip() for n in data.get("news", [])}

        para_events = defaultdict(list)
        para_morals = defaultdict(list)

        for ann in data.get("annotations", []):
            para_id = ann.get("para")
            if para_id is None:
                continue
            event_info = {
                "mention": ann.get("event", ""),
                "trigger": ann.get("event_trigger", ""),
                "entities": {}
            }
            for k in ["agent","patient","agent2","patient2","agent3","patient3"]:
                if ann.get(k):
                    role = "agent" if "agent" in k else "patient"
                    event_info["entities"][ann[k]] = role
            para_events[para_id].append(event_info)
            morals = [ann.get("morality"), ann.get("morality2"), ann.get("morality3")]
            para_morals[para_id].extend([m for m in morals if m])

        for pid, text in news_paragraphs.items():
            ev_list = para_events.get(pid, [])
            y5 = morals_to_5pos(para_morals.get(pid, []))
            domain_label = 1 if len(ev_list) > 0 else 0
            emo_scores = score_emotions_by_lex(text, lex, lowercase=True)

            # 标记 no_emotion（供对比学习掩码使用）
            noemo = 1.0 if sum(emo_scores.values()) == 0 else 0.0
            emo_scores["no_emotion"] = noemo

            rows.append({
                "article_id": article_id,
                "para_id": pid,
                "text": text,
                "event": json.dumps(ev_list, ensure_ascii=False),
                "domain": domain_label,
                **y5,
                **emo_scores
            })

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ Done. Saved {len(df)} rows -> {output_csv}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/MoralEvents/articles",
                        help="Path to MoralEvents articles/*.json")
    parser.add_argument("--output_csv", type=str, default="./data/moral_events_lex.csv",
                        help="Output CSV path")
    parser.add_argument("--nrc_lex_path", type=str, default=None,
                        help="Path to NRC/EmoLex (txt/csv). 若不提供，将自动查找/下载。")
    args = parser.parse_args()
    main(args.input_dir, args.output_csv, args.nrc_lex_path)
