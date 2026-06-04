"""V7 Full per-sequence analysis"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, os, time
from pathlib import Path
from sklearn.metrics import roc_auc_score
from unicad_torch import Galaxy, GalaxyConfig

import sys; sys.path.insert(0,"scripts")
from alfa_detect_v7 import *

torch.manual_seed(42); np.random.seed(42)
print("Loading...")
seqs = load_alfa_data()
print("Training ConditionalGRU (hist=75)...")
model = train_conditional_predictor(seqs, hist_len=75, epochs=100, device="cuda")
fseqs = compute_features(seqs, model, hist_len=75, device="cuda")
X_n = np.vstack([s["features"][(s["labels"]==0)&s["valid"]] for s in fseqs]).astype(np.float32)
X_a = np.vstack([s["features"] for s in fseqs]).astype(np.float32)
y_a = np.concatenate([s["labels"] for s in fseqs])
va = np.concatenate([s["valid"] for s in fseqs])
zm, zs = X_n.mean(0,keepdims=True), X_n.std(0,keepdims=True)+1e-8
X_nz, X_az = (X_n-zm)/zs, (X_a-zm)/zs

cfg = GalaxyConfig(k=5, hidden_dim=128, latent_dim=64, seed=42,
    pretrain_epochs=100, em_iters=3, em_finetune_steps=100,
    gravity_version="vector", score_type="vector", device="cuda", verbose=False)
g = Galaxy(cfg); g.fit(X_nz)
scores = g.predict_score(X_az)

# Per-sequence details
fault_types = {"engine": [], "elevator": [], "aileron": [], "rudder": []}
normal_aucs = []

print(f"\n{'#':<3} {'Sequence':<55} {'AUC':>6} {'Fault':>6} {'Total':>6}  Category")
print("-" * 90)
off = 0
for i, s in enumerate(fseqs):
    n = s["n_frames"]
    sv = scores[off:off+n][s["valid"]]
    yv = s["labels"][s["valid"]]
    off += n
    ft = "normal"
    for t in ["engine", "elevator", "aileron", "rudder"]:
        if t in s["name"]: ft = t; break
    if yv.sum() > 0 and yv.sum() < len(yv):
        a = roc_auc_score(yv, sv)
        fault_types[ft].append(a)
    else:
        a = float("nan")
        # Normal seq: compute false positive tendency
        if len(sv) > 0:
            normal_aucs.append(sv.mean())  # lower = better (fewer false positives)
    tag = "NORMAL" if not s["has_fault"] else ""
    print(f"{i+1:<3} {s['name']:<55} {a:>6.3f} {int(yv.sum()):>6} {len(yv):>6}  {ft if ft!='normal' else tag}")

# Summary
print(f"\n{'='*60}")
print("Category Summary (Galaxy AE 128->64, k=5, hist=75)")
print(f"{'='*60}")
for ft in ["engine", "elevator", "aileron", "rudder"]:
    aucs = fault_types[ft]
    if aucs:
        print(f"  {ft:<10}: {len(aucs):>2} seq  [{min(aucs):.3f} ... {max(aucs):.3f}]  mean={np.mean(aucs):.3f}")

all_fault = [a for aucs in fault_types.values() for a in aucs]
print(f"\n  Total fault seq: {len(all_fault)}  Mean AUC: {np.mean(all_fault):.3f}")
if normal_aucs:
    print(f"  Normal seq mean score: {np.mean(normal_aucs):.4f} (lower=cleaner separation)")

# Weakest sequences
print(f"\nWeakest sequences (AUC < 0.8):")
off = 0
print(f"  {'#':>3}  {'Name':<55} {'AUC':>6}")
for i, s in enumerate(fseqs):
    n = s["n_frames"]
    sv = scores[off:off+n][s["valid"]]
    yv = s["labels"][s["valid"]]
    off += n
    if yv.sum() <= 0 or yv.sum() >= len(yv): continue
    a = roc_auc_score(yv, sv)
    if a < 0.8:
        print(f"  {i+1:>3}  {s['name']:<55} {a:.3f}")
