"""
ARIEL Exoplanet Atmosphere Hackathon — Competitive Solution
============================================================
Strategy:
  1. Feature engineering: spectrum + noise + SNR + log-noise + supplementary
  2. Two-stage training:
       Stage 1 — pre-train ONE shared model on ALL 91k planets (sim targets)
       Stage 2 — clone + fine-tune per fold on 22k labeled retrieval targets
  3. Neural network with Gaussian NLL loss (directly optimises CRPS)
  4. K-fold ensemble × MC-Dropout for aleatoric + epistemic uncertainty
  5. Post-hoc temperature scaling for uncertainty calibration
"""

import copy
import io
import os
import sys
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_spectral_data, compute_participant_score,
    array_to_submission, TARGET_COLS, TRAINING_MEAN, TRAINING_STD,
)

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────────────────────────
DATA_DIR        = "Hackathon_training"
N_SPLITS        = 5
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 200
BATCH_SIZE      = 512
HIDDEN_DIMS     = [512, 512, 256, 128]
DROPOUT_RATE    = 0.15
LR_PRETRAIN     = 3e-3
LR_FINETUNE     = 2e-4
LR_HEAD_WARMUP  = 8e-4
N_MC_SAMPLES    = 50
DEVICE          = torch.device("cpu")   # change to "cuda" if GPU available
AUTO_SUBMIT     = False
SUBMISSION_URL  = "https://www.ariel-datachallenge.space/api/score/exohack4/calculate/"
SUBMISSION_DATA = {
    "secret_key": "TESS-700",
    "team_name": "TOI-700-d",
    "team_no": "8",
}

# ─────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────
print("Loading data …")
spectrum_stack, noise_stack, wl_grid, width = load_spectral_data(
    os.path.join(DATA_DIR, "Training_SpectralData.hdf5")
)
y_true   = pd.read_csv(os.path.join(DATA_DIR, "Training_targets.csv"))
sim_data = pd.read_csv(os.path.join(DATA_DIR, "Training_supp_simulation_data.csv"))
x_supp   = pd.read_csv(os.path.join(DATA_DIR, "Training_supplementary_data.csv"), index_col=0)

print(f"  Spectra:       {spectrum_stack.shape}")
print(f"  Targets:       {y_true.shape}")
print(f"  Sim targets:   {sim_data.shape}")
print(f"  Supplementary: {x_supp.shape}")


# ─────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────
def build_features(spec, noise, supp_df):
    eps       = 1e-10
    snr       = spec / (noise + eps)
    log_noise = np.log(noise + eps)
    supp      = supp_df.values.astype(np.float64)
    # Log-transform columns that span more than 2 orders of magnitude
    for j in range(supp.shape[1]):
        col = supp[:, j]
        if col.min() > 0 and col.max() / col.min() > 100:
            supp[:, j] = np.log(col)
    return np.concatenate([spec, noise, snr, log_noise, supp], axis=1).astype(np.float32)


def submit_predictions(mu_df, std_df):
    mu_buffer = io.StringIO()
    std_buffer = io.StringIO()
    mu_df.to_csv(mu_buffer, index=False)
    std_df.to_csv(std_buffer, index=False)

    submission_files = {
        "mu_file": ("mu_predictions.csv", mu_buffer.getvalue(), "text/csv"),
        "std_file": ("std_predictions.csv", std_buffer.getvalue(), "text/csv"),
    }

    response = requests.post(
        SUBMISSION_URL,
        data=SUBMISSION_DATA,
        files=submission_files,
        timeout=60,
    )

    if response.status_code == 200:
        print(f"  Submission successful. Score: {response.json()['score']}")
    else:
        try:
            error_msg = response.json().get("error", response.text)
        except ValueError:
            error_msg = response.text
        print(f"  Submission failed with status code: {response.status_code}")
        print(f"  Response: {error_msg}")


print("\nBuilding features …")
X_all = build_features(spectrum_stack, noise_stack, x_supp)
print(f"  Feature matrix: {X_all.shape}")


# ─────────────────────────────────────────────────────────────────
# 3.  SPLITS
# ─────────────────────────────────────────────────────────────────
labeled_mask = ~np.isnan(y_true[TARGET_COLS].values.sum(axis=1))
X_labeled    = X_all[labeled_mask]
y_labeled    = y_true[TARGET_COLS].values[labeled_mask].astype(np.float32)
y_sim_all    = sim_data[TARGET_COLS].values.astype(np.float32)

print(f"\n  Labeled planets : {labeled_mask.sum()} / {len(y_true)}")

X_dev, X_test, y_dev, y_test = train_test_split(
    X_labeled, y_labeled, test_size=0.1, random_state=42
)
print(f"  Dev: {X_dev.shape[0]}  |  Test: {X_test.shape[0]}")


# ─────────────────────────────────────────────────────────────────
# 4.  SCALERS
# ─────────────────────────────────────────────────────────────────
X_scaler = StandardScaler().fit(X_all)   # fit on all 91k (same distribution as leaderboard)
y_scaler = StandardScaler().fit(y_dev)   # fit on labeled dev only (no test leakage)

# Transform once; split indices let us avoid re-calling transform on derived subsets
X_all_s     = X_scaler.transform(X_all).astype(np.float32)
X_labeled_s = X_all_s[labeled_mask]

# Recover the shuffled train/test indices produced by the earlier train_test_split
_idx = np.arange(len(X_labeled))
idx_dev, idx_test = train_test_split(_idx, test_size=0.1, random_state=42)
X_dev_s  = X_labeled_s[idx_dev]
X_test_s = X_labeled_s[idx_test]

y_dev_s  = y_scaler.transform(y_dev).astype(np.float32)
y_sim_s  = y_scaler.transform(y_sim_all).astype(np.float32)
y_sim_labeled_s = y_sim_s[labeled_mask]
y_sim_dev_s = y_sim_labeled_s[idx_dev]


# ─────────────────────────────────────────────────────────────────
# 5.  MODEL
# ─────────────────────────────────────────────────────────────────
BASE_EPOCHS      = 100
RESIDUAL_EPOCHS  = 200
LR_BASE          = 2e-3
LR_RESIDUAL      = 4e-4


def _dense_block(in_dim, out_dim, dropout):
    return [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)]


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(x + self.net(x)))


class DeterministicNet(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim=6, dropout=0.15):
        super().__init__()
        layers = _dense_block(in_dim, hidden_dims[0], dropout)
        for i in range(1, len(hidden_dims)):
            prev, cur = hidden_dims[i - 1], hidden_dims[i]
            if prev == cur:
                layers.append(ResBlock(cur, dropout))
            else:
                layers.extend(_dense_block(prev, cur, dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class ProbabilisticResidualNet(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim=1, dropout=0.15):
        super().__init__()
        layers = _dense_block(in_dim, hidden_dims[0], dropout)
        for i in range(1, len(hidden_dims)):
            prev, cur = hidden_dims[i - 1], hidden_dims[i]
            if prev == cur:
                layers.append(ResBlock(cur, dropout))
            else:
                layers.extend(_dense_block(prev, cur, dropout))
        self.backbone = nn.Sequential(*layers)
        self.target_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[-1], 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(out_dim)
        ])
        self.mean_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(out_dim)])
        self.logstd_heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(out_dim)])

        for head in self.logstd_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x):
        h = self.backbone(x)
        mean_parts, std_parts = [], []
        for tower, mean_head, logstd_head in zip(self.target_towers, self.mean_heads, self.logstd_heads):
            target_h = tower(h)
            mean_parts.append(mean_head(target_h))
            std_parts.append(torch.exp(logstd_head(target_h)).clamp(1e-4, 10.0))
        mu = torch.cat(mean_parts, dim=1)
        std = torch.cat(std_parts, dim=1)
        return mu, std


def gaussian_nll(y, mu, std):
    return (torch.log(std) + 0.5 * ((y - mu) / std) ** 2).mean()


def make_base_model(in_dim):
    return DeterministicNet(in_dim, HIDDEN_DIMS, dropout=DROPOUT_RATE).to(DEVICE)


def make_residual_model(in_dim, out_dim=1):
    return ProbabilisticResidualNet(in_dim, HIDDEN_DIMS, out_dim=out_dim, dropout=DROPOUT_RATE).to(DEVICE)


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def predict_single_model_scaled(model, X_scaled, n_mc=N_MC_SAMPLES):
    n_rows = len(X_scaled)
    X_t = torch.tensor(X_scaled).to(DEVICE)

    enable_mc_dropout(model)
    X_rep = X_t.repeat(n_mc, 1)
    with torch.no_grad():
        mu_rep, std_rep = model(X_rep)

    mc_mu = mu_rep.cpu().numpy().reshape(n_mc, n_rows, -1)
    mc_var = (std_rep ** 2).cpu().numpy().reshape(n_mc, n_rows, -1)
    mean_s = mc_mu.mean(axis=0)
    var_s = mc_var.mean(axis=0) + mc_mu.var(axis=0)
    return mean_s, var_s


def predict_mean_model(model, X_scaled):
    model.eval()
    X_t = torch.tensor(X_scaled).to(DEVICE)
    with torch.no_grad():
        return model(X_t).cpu().numpy()


# ─────────────────────────────────────────────────────────────────
# 6.  TRAIN HELPER
# ─────────────────────────────────────────────────────────────────
def train_mean_model(model, X_tr, y_tr, X_val, y_val,
                     epochs, lr, batch_size, patience=20, tag=""):
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)
    ldr   = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    Xv = torch.tensor(X_val).to(DEVICE)
    yv = torch.tensor(y_val).to(DEVICE)

    best_val, best_state, no_imp = 1e9, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in ldr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            ((pred - yb) ** 2).mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_nll = ((model(Xv) - yv) ** 2).mean().item()

        if val_nll < best_val - 1e-5:
            best_val   = val_nll
            best_state = copy.deepcopy(model.state_dict())
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
        if epoch % max(1, epochs // 4) == 0 or epoch == epochs:
            print(f"    [{tag}] ep {epoch:4d}/{epochs}  val_mse={val_nll:.4f}")

    model.load_state_dict(best_state)
    return model


def train_prob_model(model, X_tr, y_tr, X_val, y_val,
                     epochs, lr, batch_size, patience=20, tag=""):
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)
    ldr   = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    Xv = torch.tensor(X_val).to(DEVICE)
    yv = torch.tensor(y_val).to(DEVICE)

    best_val, best_state, no_imp = 1e9, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in ldr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            gaussian_nll(yb, *model(Xb)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_nll = gaussian_nll(yv, *model(Xv)).item()

        if val_nll < best_val - 1e-4:
            best_val   = val_nll
            best_state = copy.deepcopy(model.state_dict())
            no_imp     = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
        if epoch % max(1, epochs // 4) == 0 or epoch == epochs:
            print(f"    [{tag}] ep {epoch:4d}/{epochs}  val_nll={val_nll:.4f}")

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────
# 7.  TRAINING  (pre-train once → fine-tune per fold)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Stage 1: Train explicit simulation base model")
print("=" * 60)

base_model = make_base_model(X_all_s.shape[1])
base_model = train_mean_model(
    base_model,
    X_all_s, y_sim_s,
    X_dev_s, y_sim_dev_s,
    epochs=BASE_EPOCHS, lr=LR_BASE, batch_size=1024,
    patience=15, tag="pretrain",
)

base_dev_s = predict_mean_model(base_model, X_dev_s).astype(np.float32)
base_test_s = predict_mean_model(base_model, X_test_s).astype(np.float32)

print("\n" + "=" * 60)
print("Stage 2: Train per-target residual ensemble on labeled targets")
print("=" * 60)

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
models = [[] for _ in TARGET_COLS]
oof_mean_s = np.zeros_like(y_dev_s)
oof_var_s  = np.zeros_like(y_dev_s)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_dev_s)):
    print(f"\n── Fold {fold + 1}/{N_SPLITS} ──")
    base_tr = base_dev_s[tr_idx]
    base_val = base_dev_s[val_idx]
    Xtr = np.concatenate([X_dev_s[tr_idx], base_tr], axis=1).astype(np.float32)
    Xval = np.concatenate([X_dev_s[val_idx], base_val], axis=1).astype(np.float32)
    for target_idx, target_name in enumerate(TARGET_COLS):
        ytr = (y_dev_s[tr_idx, [target_idx]] - base_tr[:, [target_idx]]).astype(np.float32)
        yval = (y_dev_s[val_idx, [target_idx]] - base_val[:, [target_idx]]).astype(np.float32)

        model = make_residual_model(Xtr.shape[1], out_dim=1)
        model = train_prob_model(
            model, Xtr, ytr, Xval, yval,
            epochs=RESIDUAL_EPOCHS, lr=LR_RESIDUAL, batch_size=BATCH_SIZE,
            patience=30, tag=f"residual-f{fold+1}-{target_name}",
        )
        fold_resid_mean_s, fold_resid_var_s = predict_single_model_scaled(model, Xval)
        oof_mean_s[val_idx, target_idx] = base_val[:, target_idx] + fold_resid_mean_s[:, 0]
        oof_var_s[val_idx, target_idx] = fold_resid_var_s[:, 0]
        models[target_idx].append(model)
    print(f"  Fold {fold + 1} complete.")


# ─────────────────────────────────────────────────────────────────
# 8.  ENSEMBLE + MC DROPOUT INFERENCE
# ─────────────────────────────────────────────────────────────────
def predict(models, X_scaled, y_scaler, n_mc=N_MC_SAMPLES):
    """
    Returns (mean, std) in original target space.
    Uses a single batched forward pass per model instead of n_mc loops.
    Variance: total = E[sigma²] (aleatoric) + Var[mu] (epistemic)
    """
    base_mean_s = predict_mean_model(base_model, X_scaled).astype(np.float32)
    residual_input = np.concatenate([X_scaled, base_mean_s], axis=1).astype(np.float32)

    fold_means_s, fold_vars_s = [], []
    for fold_idx in range(N_SPLITS):
        fold_mean_s = base_mean_s.copy()
        fold_var_s = np.zeros_like(base_mean_s)
        for target_idx in range(len(TARGET_COLS)):
            fold_resid_mean_s, target_var_s = predict_single_model_scaled(
                models[target_idx][fold_idx], residual_input, n_mc=n_mc
            )
            fold_mean_s[:, target_idx] += fold_resid_mean_s[:, 0]
            fold_var_s[:, target_idx] = target_var_s[:, 0]
        fold_means_s.append(fold_mean_s)
        fold_vars_s.append(fold_var_s)

    all_means_s = np.stack(fold_means_s)
    all_vars_s = np.stack(fold_vars_s)

    all_means = np.stack([y_scaler.inverse_transform(fold_mean_s) for fold_mean_s in all_means_s])
    all_vars = np.stack([
        (np.sqrt(fold_var_s) * y_scaler.scale_) ** 2
        for fold_var_s in all_vars_s
    ])

    final_mean = all_means.mean(axis=0)
    final_std  = np.sqrt(all_vars.mean(axis=0) + all_means.var(axis=0)).clip(min=1e-6)
    return final_mean, final_std


print("\n" + "=" * 60)
print("INFERENCE ON HELD-OUT TEST SET")
print("=" * 60)
y_pred_mean, y_pred_std = predict(models, X_test_s, y_scaler)


# ─────────────────────────────────────────────────────────────────
# 9.  TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────
print("\nCalibrating uncertainty from out-of-fold dev predictions …")

oof_mean = y_scaler.inverse_transform(oof_mean_s)
oof_std  = np.sqrt(oof_var_s) * y_scaler.scale_

y_dev_n    = (y_dev - TRAINING_MEAN) / TRAINING_STD
oof_mu_n   = (oof_mean - TRAINING_MEAN) / TRAINING_STD
oof_sigma_n = oof_std / np.array(TRAINING_STD)

T_opt = np.ones(6)
for p in range(6):
    # default-arg trick captures p by value, avoiding closure-over-loop-variable pitfall
    def neg_nll(log_T, p=p):
        sig = np.clip(oof_sigma_n[:, p] * np.exp(log_T), 1e-6, None)
        return (np.log(sig) + 0.5 * ((y_dev_n[:, p] - oof_mu_n[:, p]) / sig) ** 2).mean()
    T_opt[p] = np.exp(minimize_scalar(neg_nll, bounds=(-2.0, 2.0), method="bounded").x)

print(f"  Temperature factors: {T_opt.round(3)}")
y_pred_std_cal = y_pred_std * T_opt


# ─────────────────────────────────────────────────────────────────
# 10.  SCORE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SCORES")
print("=" * 60)
pids = np.arange(len(y_test))
score_raw = compute_participant_score(
    array_to_submission(y_test,         pids),
    array_to_submission(y_pred_mean,    pids),
    array_to_submission(y_pred_std,     pids),
)
score_cal = compute_participant_score(
    array_to_submission(y_test,         pids),
    array_to_submission(y_pred_mean,    pids),
    array_to_submission(y_pred_std_cal, pids),
)
print(f"\n  Uncalibrated score : {score_raw['score']:.4f}")
print(f"  Calibrated   score : {score_cal['score']:.4f}")
print(f"\n  Per-parameter skill (calibrated):")
for i, col in enumerate(TARGET_COLS):
    print(f"    {col:<20s}: {score_cal['score_per_param'][i]:+.4f}")
print("\n  (Baseline ExtraTrees was ~0.066)")


# ─────────────────────────────────────────────────────────────────
# 11.  LEADERBOARD PREDICTIONS
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LEADERBOARD PREDICTIONS")
print("=" * 60)

lb_spec, lb_noise, _, _ = load_spectral_data(
    os.path.join(DATA_DIR, "Test_SpectralData.hdf5")
)
lb_supp = pd.read_csv(os.path.join(DATA_DIR, "Test_supplementary_data.csv"), index_col=0)
X_lb_s  = X_scaler.transform(build_features(lb_spec, lb_noise, lb_supp)).astype(np.float32)

lb_mean, lb_std = predict(models, X_lb_s, y_scaler)

pids_lb = np.arange(len(lb_mean))
lb_mu_submission = array_to_submission(lb_mean, pids_lb)
lb_std_submission = array_to_submission(lb_std * T_opt, pids_lb)

lb_mu_submission.to_csv("lb_mu_predictions.csv", index=False)
lb_std_submission.to_csv("lb_std_predictions.csv", index=False)
print(f"  Saved {len(lb_mean)} predictions → lb_mu_predictions.csv / lb_std_predictions.csv")

if AUTO_SUBMIT:
    print("  Submitting predictions to challenge server …")
    submit_predictions(lb_mu_submission, lb_std_submission)

# ─────────────────────────────────────────────────────────────────
# 12.  SAVE MODELS
# ─────────────────────────────────────────────────────────────────
os.makedirs("saved_models", exist_ok=True)
torch.save(base_model.state_dict(), "saved_models/base_model.pt")
for target_idx, target_name in enumerate(TARGET_COLS):
    for fold_idx, model in enumerate(models[target_idx], start=1):
        torch.save(model.state_dict(), f"saved_models/{target_name}_fold_{fold_idx}.pt")

np.savez("saved_models/scalers.npz",
    T_opt          = T_opt,
    X_scaler_mean  = X_scaler.mean_,
    X_scaler_scale = X_scaler.scale_,
    y_scaler_mean  = y_scaler.mean_,
    y_scaler_scale = y_scaler.scale_,
)
print("\nModels saved to saved_models/")
print("Done!")
