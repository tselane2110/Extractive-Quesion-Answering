#!/usr/bin/env python
"""
SQuAD 1.1 extractive QA: three transformer + start/end-head approaches in one pipeline.

  token_cls : BERT encoder + two linear heads (start / end), cross-entropy.
  rl        : same architecture, warm-started from a supervised checkpoint and fine-tuned with
              policy gradient (REINFORCE, leave-one-out baseline) using a span-level F1 reward,
              mixed with cross-entropy for stability.
  roberta   : RoBERTa encoder + two MLP span heads, cross-entropy.

Every model reads "[CLS] question [SEP] context [SEP]" jointly through the encoder and predicts
a start distribution and an end distribution over tokens.

Install:  pip install torch transformers numpy tqdm
Usage:
  python squad_span_qa.py --approach token_cls
  python squad_span_qa.py --approach roberta
  python squad_span_qa.py --approach rl --init_from runs/token_cls/best
  python squad_span_qa.py --compare runs
Data (train-v1.1.json / dev-v1.1.json) is downloaded automatically into --data_dir.
"""
import argparse
import collections
import contextlib
import json
import math
import os
import random
import re
import string
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}
NEG = -1e9  # mask value for logits (applied after casting logits to float32)

PRESETS = {
    "token_cls": dict(encoder="bert-base-uncased", head="linear", lr=3e-5, epochs=3, batch_size=32,
                      ce_warmup_epochs=0),
    "rl": dict(encoder="bert-base-uncased", head="linear", lr=1e-5, epochs=3, batch_size=32,
               ce_warmup_epochs=0),
    "roberta": dict(encoder="roberta-base", head="mlp", lr=3e-5, epochs=3, batch_size=32,
                    ce_warmup_epochs=0),
}


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_amp(device, precision):
    """Returns (autocast_factory, grad_scaler_or_None)."""
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext, None
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = None
    if precision == "fp16":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    return (lambda: torch.autocast("cuda", dtype=dtype)), scaler


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------
def load_squad(split, data_dir):
    path = Path(data_dir) / f"{split}-v1.1.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {SQUAD_URLS[split]} -> {path}")
        urllib.request.urlretrieve(SQUAD_URLS[split], path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)["data"]
    examples = []
    for article in raw:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                examples.append({
                    "id": qa["id"],
                    "title": article["title"],
                    "question": qa["question"],
                    "context": para["context"],
                    "answers": [a["text"] for a in qa["answers"]],
                    "answer_starts": [a["answer_start"] for a in qa["answers"]],
                })
    return examples


def split_by_article(examples, n_val_articles, seed):
    """Hold out whole articles so validation questions never share a context with training."""
    if n_val_articles <= 0:
        return examples, []
    titles = sorted({e["title"] for e in examples})
    random.Random(seed).shuffle(titles)
    val_titles = set(titles[:n_val_articles])
    train = [e for e in examples if e["title"] not in val_titles]
    val = [e for e in examples if e["title"] in val_titles]
    return train, val


def build_features(tok, examples, max_len, stride, with_labels, desc, chunk=2000):
    """
    Tokenizes (question, context) jointly. Long contexts become several overlapping windows.
    Per window we keep:
      input_ids, token_type_ids (if the model uses them), ctx_mask (context tokens + CLS),
      offsets (char span per token, -1 for non-context)  -- used at eval time
      start / end token labels (train only; CLS index 0 if the answer is not inside the window)
    """
    use_tt = "token_type_ids" in tok.model_input_names
    feats = []
    for c0 in tqdm(range(0, len(examples), chunk), desc=desc):
        batch_ex = examples[c0:c0 + chunk]
        enc = tok(
            [e["question"].lstrip() for e in batch_ex],
            [e["context"] for e in batch_ex],
            truncation="only_second", max_length=max_len, stride=stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
        )
        for i in range(len(enc["input_ids"])):
            ex_idx = c0 + enc["overflow_to_sample_mapping"][i]
            seq_ids = enc.sequence_ids(i)
            offs = enc["offset_mapping"][i]
            ctx_idx = [k for k, s in enumerate(seq_ids) if s == 1]
            ctx_start, ctx_end = ctx_idx[0], ctx_idx[-1]
            ctx_mask = np.zeros(len(seq_ids), dtype=np.bool_)
            ctx_mask[ctx_start:ctx_end + 1] = True
            ctx_mask[0] = True  # CLS is an allowed "answer not in this window" target
            f = {
                "ex_idx": ex_idx,
                "input_ids": np.asarray(enc["input_ids"][i], dtype=np.int64),
                "ctx_mask": ctx_mask,
            }
            if use_tt:
                f["token_type_ids"] = np.asarray(enc["token_type_ids"][i], dtype=np.int64)
            if with_labels:
                ex = examples[ex_idx]
                s_char = ex["answer_starts"][0]
                e_char = s_char + len(ex["answers"][0])
                if offs[ctx_start][0] > s_char or offs[ctx_end][1] < e_char:
                    f["start"] = f["end"] = 0
                else:
                    ts = ctx_start
                    while ts <= ctx_end and offs[ts][0] <= s_char:
                        ts += 1
                    te = ctx_end
                    while te >= ctx_start and offs[te][1] >= e_char:
                        te -= 1
                    f["start"], f["end"] = ts - 1, te + 1
            else:
                o = np.full((len(seq_ids), 2), -1, dtype=np.int64)
                for k in ctx_idx:
                    o[k] = offs[k]
                f["offsets"] = o
            feats.append(f)
    return feats


def make_collate(pad_id):
    def collate(batch):
        B, L = len(batch), max(len(b["input_ids"]) for b in batch)
        out = {
            "input_ids": torch.full((B, L), pad_id, dtype=torch.long),
            "attention_mask": torch.zeros((B, L), dtype=torch.long),
            "ctx_mask": torch.zeros((B, L), dtype=torch.bool),
        }
        has_tt = "token_type_ids" in batch[0]
        if has_tt:
            out["token_type_ids"] = torch.zeros((B, L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            out["input_ids"][i, :n] = torch.from_numpy(b["input_ids"])
            out["attention_mask"][i, :n] = 1
            out["ctx_mask"][i, :n] = torch.from_numpy(b["ctx_mask"])
            if has_tt:
                out["token_type_ids"][i, :n] = torch.from_numpy(b["token_type_ids"])
        if "start" in batch[0]:
            out["start"] = torch.tensor([b["start"] for b in batch], dtype=torch.long)
            out["end"] = torch.tensor([b["end"] for b in batch], dtype=torch.long)
        return out
    return collate


# --------------------------------------------------------------------------------------
# Model: encoder + two prediction heads (start, end)
# --------------------------------------------------------------------------------------
class SpanHead(nn.Module):
    def __init__(self, hidden, kind, dropout):
        super().__init__()
        if kind == "linear":
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        elif kind == "mlp":
            self.net = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
                nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1))
        else:
            raise ValueError(f"unknown head type {kind}")

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SpanQAModel(nn.Module):
    def __init__(self, encoder_src, head="linear", dropout=0.1, pretrained=True):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(encoder_src)
        else:  # architecture only; weights come from our own checkpoint
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder_src))
        hidden = self.encoder.config.hidden_size
        self.head_kind, self.dropout_p = head, dropout
        self.start_head = SpanHead(hidden, head, dropout)
        self.end_head = SpanHead(hidden, head, dropout)
        std = getattr(self.encoder.config, "initializer_range", 0.02)
        for m in list(self.start_head.modules()) + list(self.end_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, ctx_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        hs = self.encoder(**kw).last_hidden_state
        # Only context tokens (and CLS = "not in this window") can be start/end.
        start = self.start_head(hs).float().masked_fill(~ctx_mask, NEG)
        end = self.end_head(hs).float().masked_fill(~ctx_mask, NEG)
        return start, end


def save_ckpt(model, tok, path, args):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.encoder.config.save_pretrained(path)
    torch.save(model.state_dict(), path / "model.pt")
    tok.save_pretrained(path)
    with open(path / "span_config.json", "w") as f:
        json.dump({"head": model.head_kind, "dropout": model.dropout_p,
                   "max_len": args.max_len, "stride": args.stride}, f)


def load_ckpt(path, device):
    path = Path(path)
    with open(path / "span_config.json") as f:
        meta = json.load(f)
    model = SpanQAModel(str(path), meta["head"], meta["dropout"], pretrained=False)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    return model.to(device), meta


# --------------------------------------------------------------------------------------
# Official SQuAD v1.1 metrics
# --------------------------------------------------------------------------------------
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def squad_metrics(examples, preds):
    em = f1 = 0.0
    for ex in examples:
        p = preds.get(ex["id"], "")
        em += max(float(normalize_answer(p) == normalize_answer(g)) for g in ex["answers"])
        f1 += max(f1_score(p, g) for g in ex["answers"])
    n = max(len(examples), 1)
    return {"exact_match": 100 * em / n, "f1": 100 * f1 / n}


# --------------------------------------------------------------------------------------
# Inference + postprocessing
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, feats, collate, batch_size, device, amp, workers):
    model.eval()
    dl = DataLoader(feats, batch_size=batch_size, shuffle=False, collate_fn=collate,
                    num_workers=workers)
    S, E = [], []
    for b in tqdm(dl, desc="predict", leave=False):
        inputs = {k: v.to(device) for k, v in b.items() if k not in ("start", "end")}
        with amp():
            s, e = model(**inputs)
        lens = b["attention_mask"].sum(1).tolist()
        s, e = s.cpu().numpy(), e.cpu().numpy()
        for i, n in enumerate(lens):
            S.append(s[i, :n])
            E.append(e[i, :n])
    return S, E


def postprocess(examples, feats, S, E, n_best, max_answer_len):
    """Best (start, end) over all windows of an example by start_logit + end_logit."""
    by_ex = collections.defaultdict(list)
    for fi, f in enumerate(feats):
        by_ex[f["ex_idx"]].append(fi)
    preds = {}
    for ei, ex in enumerate(examples):
        best_score, best_text = -float("inf"), ""
        for fi in by_ex.get(ei, []):
            s_log, e_log, offs = S[fi], E[fi], feats[fi]["offsets"]
            valid = offs[:, 0] >= 0  # context tokens only (excludes CLS and question)
            for a in np.argsort(s_log)[::-1][:n_best]:
                if not valid[a]:
                    continue
                for b in np.argsort(e_log)[::-1][:n_best]:
                    if not valid[b] or b < a or b - a + 1 > max_answer_len:
                        continue
                    score = s_log[a] + e_log[b]
                    if score > best_score:
                        best_score = score
                        best_text = ex["context"][offs[a, 0]:offs[b, 1]]
        preds[ex["id"]] = best_text
    return preds


def evaluate(model, examples, feats, collate, args, device, amp):
    S, E = predict_logits(model, feats, collate, args.eval_batch_size, device, amp, args.workers)
    preds = postprocess(examples, feats, S, E, args.n_best, args.max_answer_len)
    return squad_metrics(examples, preds), preds


# --------------------------------------------------------------------------------------
# Losses: supervised CE and policy-gradient with span-level reward
# --------------------------------------------------------------------------------------
def ce_loss(start_logits, end_logits, start_gold, end_gold):
    return 0.5 * (F.cross_entropy(start_logits, start_gold) + F.cross_entropy(end_logits, end_gold))


def span_f1_reward(s, e, gs, ge, max_answer_len):
    """
    Token-level F1 between predicted span [s, e] and gold span [gs, ge] (all shape (K, B) / (B,)).
    F1 of two intervals = 2 * overlap / (len_pred + len_gold).
    Invalid spans (end < start, too long, or half-CLS) get reward 0.
    """
    overlap = (torch.minimum(e, ge) - torch.maximum(s, gs) + 1).clamp(min=0).float()
    len_p = (e - s + 1).clamp(min=1).float()
    len_g = (ge - gs + 1).clamp(min=1).float()
    reward = 2 * overlap / (len_p + len_g)
    valid = (e >= s) & ((e - s + 1) <= max_answer_len) & ((s == 0) == (e == 0))
    return reward * valid.float()


def rl_loss(start_logits, end_logits, gs, ge, k, max_answer_len, entropy_coef):
    """
    REINFORCE with leave-one-out baseline.
    Policy: pi(s, e | x) = pi_start(s | x) * pi_end(e | x). Draw K spans per example, reward each
    with span-level F1, baseline for sample i = mean reward of the other K-1 samples.
    """
    ds, de = Categorical(logits=start_logits), Categorical(logits=end_logits)
    s, e = ds.sample((k,)), de.sample((k,))          # (K, B)
    logp = ds.log_prob(s) + de.log_prob(e)             # (K, B)
    with torch.no_grad():
        r = span_f1_reward(s, e, gs.unsqueeze(0), ge.unsqueeze(0), max_answer_len)
        baseline = (r.sum(0, keepdim=True) - r) / (k - 1)
        adv = r - baseline
    pg = -(adv * logp).mean()
    entropy = (ds.entropy() + de.entropy()).mean()
    return pg - entropy_coef * entropy, r.mean().item()


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train(args, model, tok, train_feats, val_ex, val_feats, collate, device, amp, scaler, run_dir):
    dl = DataLoader(train_feats, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=args.workers, pin_memory=device.type == "cuda")
    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    decay = [p for p in model.parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)

    best_f1, history = -1.0, []
    for epoch in range(args.epochs):
        use_rl = args.approach == "rl" and epoch >= args.ce_warmup_epochs
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss = run_reward = 0.0
        pbar = tqdm(dl, desc=f"epoch {epoch + 1}/{args.epochs} [{'RL+CE' if use_rl else 'CE'}]")
        for step, b in enumerate(pbar):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            inputs = {k: v for k, v in b.items() if k not in ("start", "end")}
            with amp():
                s_log, e_log = model(**inputs)
            ce = ce_loss(s_log, e_log, b["start"], b["end"])
            if use_rl:
                rl, reward = rl_loss(s_log, e_log, b["start"], b["end"], args.rl_samples,
                                     args.max_answer_len, args.entropy_coef)
                loss = (1 - args.rl_weight) * ce + args.rl_weight * rl
                run_reward = 0.98 * run_reward + 0.02 * reward if step else reward
            else:
                loss = ce
            run_loss = 0.98 * run_loss + 0.02 * ce.item() if step else ce.item()

            loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(dl):
                if scaler is not None:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler is not None:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            pbar.set_postfix(ce=f"{run_loss:.3f}", **({"reward": f"{run_reward:.3f}"} if use_rl else {}))

        rec = {"epoch": epoch + 1, "train_ce": run_loss, "mode": "rl" if use_rl else "ce"}
        if val_feats:
            m, _ = evaluate(model, val_ex, val_feats, collate, args, device, amp)
            rec.update({f"val_{k}": v for k, v in m.items()})
            print(f"epoch {epoch + 1}: val EM {m['exact_match']:.2f}  F1 {m['f1']:.2f}")
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                save_ckpt(model, tok, run_dir / "best", args)
        else:
            save_ckpt(model, tok, run_dir / "best", args)
        history.append(rec)
    return history


# --------------------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------------------
def compare(runs_dir):
    rows = []
    for p in sorted(Path(runs_dir).glob("*/metrics.json")):
        with open(p) as f:
            m = json.load(f)
        rows.append((m["approach"], m["encoder"], m["dev"]["exact_match"], m["dev"]["f1"]))
    print(f"{'approach':<12}{'encoder':<24}{'dev EM':>8}{'dev F1':>8}")
    for a, enc, em, f1 in rows:
        print(f"{a:<12}{enc:<24}{em:>8.2f}{f1:>8.2f}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--approach", choices=list(PRESETS))
    p.add_argument("--compare", metavar="RUNS_DIR", help="print dev metrics of finished runs and exit")
    p.add_argument("--encoder", default=None, help="HF model name or local path")
    p.add_argument("--head", choices=["linear", "mlp"], default=None)
    p.add_argument("--init_from", default=None, help="checkpoint dir to start from (RL / eval_only)")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--max_len", type=int, default=384)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_len", type=int, default=30)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_articles", type=int, default=20,
                   help="train articles held out for checkpoint selection (0 = keep last epoch)")
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--max_train_samples", type=int, default=None, help="debug: subsample")
    p.add_argument("--max_eval_samples", type=int, default=None, help="debug: subsample")
    # RL
    p.add_argument("--rl_weight", type=float, default=0.5, help="loss = (1-w)*CE + w*policy-gradient")
    p.add_argument("--rl_samples", type=int, default=4, help="spans sampled per example (>=2)")
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--ce_warmup_epochs", type=int, default=None,
                   help="RL approach: epochs of pure CE before switching to RL")
    args = p.parse_args()
    if args.compare:
        return args
    if not args.approach:
        p.error("--approach is required")
    preset = dict(PRESETS[args.approach])
    if args.approach == "rl" and not args.init_from:  # no supervised start -> CE warm-up first
        preset.update(lr=3e-5, epochs=3, ce_warmup_epochs=1)
    for k, v in preset.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.rl_samples < 2:
        p.error("--rl_samples must be >= 2 (leave-one-out baseline)")
    if args.eval_only and not args.init_from:
        p.error("--eval_only needs --init_from")
    return args


def main():
    args = parse_args()
    if args.compare:
        compare(args.compare)
        return
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp, scaler = setup_amp(device, args.precision)
    run_dir = Path(args.out_dir) / args.approach
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- model + tokenizer
    if args.init_from:
        model, meta = load_ckpt(args.init_from, device)
        tok = AutoTokenizer.from_pretrained(args.init_from, use_fast=True)
        args.encoder, args.head = str(args.init_from), meta["head"]
        args.max_len, args.stride = meta["max_len"], meta["stride"]
    else:
        tok = AutoTokenizer.from_pretrained(args.encoder, use_fast=True)
        model = SpanQAModel(args.encoder, args.head, args.dropout).to(device)
    assert tok.is_fast, "A fast tokenizer is required for offset mappings."
    collate = make_collate(tok.pad_token_id)

    # ---- data
    t0 = time.time()
    dev_ex = load_squad("dev", args.data_dir)
    if args.max_eval_samples:
        dev_ex = dev_ex[:args.max_eval_samples]
    dev_feats = build_features(tok, dev_ex, args.max_len, args.stride, False, "dev features")
    history = []
    if not args.eval_only:
        train_ex = load_squad("train", args.data_dir)
        train_ex, val_ex = split_by_article(train_ex, args.val_articles, args.seed)
        if args.max_train_samples:
            train_ex = train_ex[:args.max_train_samples]
            val_ex = val_ex[:max(args.max_train_samples // 10, 1)]
        print(f"train questions: {len(train_ex)}  val questions: {len(val_ex)}  dev questions: {len(dev_ex)}")
        train_feats = build_features(tok, train_ex, args.max_len, args.stride, True, "train features")
        val_feats = build_features(tok, val_ex, args.max_len, args.stride, False, "val features") if val_ex else []
        print(f"train windows: {len(train_feats)}")
        history = train(args, model, tok, train_feats, val_ex, val_feats, collate, device, amp,
                        scaler, run_dir)
        model, _ = load_ckpt(run_dir / "best", device)  # evaluate the selected checkpoint

    # ---- final evaluation on the official dev set
    dev_metrics, preds = evaluate(model, dev_ex, dev_feats, collate, args, device, amp)
    print(f"[{args.approach}] dev EM {dev_metrics['exact_match']:.2f}  F1 {dev_metrics['f1']:.2f}")
    with open(run_dir / "predictions.json", "w") as f:
        json.dump(preds, f)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"approach": args.approach, "encoder": args.encoder, "head": args.head,
                   "dev": dev_metrics, "history": history,
                   "seconds": time.time() - t0, "args": vars(args)}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
