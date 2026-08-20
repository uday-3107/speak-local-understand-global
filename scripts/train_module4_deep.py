"""Module 4: Deep Learning language classification (RNN / LSTM / Transformer).

Classifies speech language (en/hi/te) from sequential MFCC frames using three
neural architectures, reports accuracy/precision/recall/F1 on the held-out
test set, and persists each model's best weights.

Built on the Module 2 clean dataset (data/processed/clean/) — one WAV +
feature row per segment. Run from repo root:

    /opt/anaconda3/bin/python -m scripts.train_module4_deep --epochs 15
    /opt/anaconda3/bin/python -m scripts.train_module4_deep --limit 600 --epochs 3   # smoke

Writes: data/processed/dl/results.json and data/processed/dl/<arch>_best.pt
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

SR = 16000
N_MFCC = 13
N_CLASSES = 3
LANG_IDX = {"en": 0, "hi": 1, "te": 2}
CLASSES = ["en", "hi", "te"]
CLEAN_DIR = "data/processed/clean"
OUT_DIR = "data/processed/dl"
MAX_FRAMES = 160
SEED = 42


def log(msg: str) -> None:
    print(f"[module4] {msg}")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    log(f"device: {dev}")
    return dev


def seq_path(language: str, segment_id: int) -> str:
    return os.path.join(CLEAN_DIR, f"{language}_{segment_id:06d}.wav")


def load_mfcc_sequence(path: str) -> np.ndarray | None:
    """Return (T, N_MFCC) float32 padded to MAX_FRAMES, or None if missing."""
    if not os.path.exists(path):
        return None
    import soundfile as sf
    import librosa
    x, _ = sf.read(path, dtype="float32", always_2d=False)
    if x.size == 0:
        return None
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=N_MFCC).T  # (T, 13)
    if len(mfcc) > MAX_FRAMES:
        mfcc = mfcc[:MAX_FRAMES]
    elif len(mfcc) < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, MAX_FRAMES - len(mfcc)), (0, 0)), mode="constant")
    return mfcc.astype(np.float32)


# ----------------------------- PyTorch models -----------------------------

class RNNClassifier(nn.Module):
    def __init__(self, input_size: int = N_MFCC, hidden: int = 64):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)                    # (B, T, hidden)
        return self.head(out[:, -1, :])


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int = N_MFCC, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                   # (B, T, hidden)
        return self.head(out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_FRAMES):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int = N_MFCC, d_model: int = 64, nhead: int = 4,
                 n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               batch_first=True, dim_feedforward=128)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.proj(x))             # (B, T, d_model)
        x = self.encoder(x)                    # (B, T, d_model)
        return self.head(x.mean(dim=1))        # global mean pooling


def make_model(mode: str, device: torch.device) -> nn.Module:
    models = {
        "rnn": RNNClassifier,
        "lstm": LSTMClassifier,
        "transformer": TransformerClassifier,
    }
    if mode not in models:
        raise SystemExit(f"unknown mode {mode!r}; pick from {sorted(models)}")
    return models[mode]().to(device)


# ----------------------------- Dataset / loop -----------------------------

class SpeechDataset(Dataset):
    def __init__(self, x: list[np.ndarray], y: list[int]):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.x[i]), torch.tensor(self.y[i], dtype=torch.long)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(yb.tolist())
    return (accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro"),
            recall_score(y_true, y_pred, average="macro"),
            f1_score(y_true, y_pred, average="macro"),
            y_true, y_pred)


def train(model: nn.Module, train_dl: DataLoader, test_dl: DataLoader,
          device: torch.device, epochs: int, mode: str) -> tuple[nn.Module, float]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            total += loss.item()
        acc, _, _, val_f1, _, _ = evaluate(model, test_dl, device)
        log(f"{mode} epoch {epoch}/{epochs} loss={total/len(train_dl):.4f} "
            f"val_acc={acc:.4f} val_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap segments per language (smoke test)")
    parser.add_argument("--mode", default="all", choices=["all", "rnn", "lstm", "transformer"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_parquet(os.path.join(CLEAN_DIR, "clean.parquet"))
    if args.limit:
        df = df.groupby("language").head(args.limit)
    log(f"segments selected: {len(df)} ({df['language'].value_counts().to_dict()})")

    x, y, missing = [], [], 0
    for row in df.itertuples():
        frames = load_mfcc_sequence(seq_path(row.language, row.segment_id))
        if frames is None:
            missing += 1
            continue
        x.append(frames)
        y.append(LANG_IDX[row.language])
    if missing:
        log(f"missing wavs: {missing}")
    if not x:
        raise SystemExit("no audio loaded — check data/processed/clean exists")

    idx = np.arange(len(x))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=y,
                                           random_state=args.seed)
    x_tr = [x[i] for i in train_idx]; y_tr = [y[i] for i in train_idx]
    x_te = [x[i] for i in test_idx];  y_te = [y[i] for i in test_idx]
    train_dl = DataLoader(SpeechDataset(x_tr, y_tr), batch_size=args.batch, shuffle=True)
    test_dl = DataLoader(SpeechDataset(x_te, y_te), batch_size=args.batch, shuffle=False)
    log(f"train {len(x_tr)} / test {len(x_te)}")

    modes = ["rnn", "lstm", "transformer"] if args.mode == "all" else [args.mode]
    results, best_meta = [], None
    for mode in modes:
        model = make_model(mode, device)
        n_params = sum(p.numel() for p in model.parameters())
        log(f"training {mode} ({n_params} params)")
        model, _ = train(model, train_dl, test_dl, device, args.epochs, mode)
        acc, prec, rec, f1, y_true, y_pred = evaluate(model, test_dl, device)
        results.append({
            "model": mode,
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec), 4),
            "recall_macro": round(float(rec), 4),
            "f1_macro": round(float(f1), 4),
            "classification_report": classification_report(
                y_true, y_pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True),
        })
        log(f"{mode}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{mode}_best.pt"))
        if best_meta is None or f1 > best_meta[1]:
            best_meta = (mode, f1)

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"results": results, "best": best_meta[0], "classes": CLASSES}, f, indent=2)

    print("\n=== Module 4 results (test set, n=%d) ===" % len(x_te))
    print("{:<13} {:>8} {:>9} {:>9} {:>8}".format("model", "acc", "prec", "rec", "f1"))
    for r in results:
        print("{:<13} {:>8} {:>9} {:>9} {:>8}".format(
            r["model"], r["accuracy"], r["precision_macro"],
            r["recall_macro"], r["f1_macro"]))
    print("\nBest:", best_meta[0])
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()