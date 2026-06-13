"""Tier1.1: 完整无监督 domain adaptation cross-dataset。
关掉"只测了 naive mixing"这扇门:per-domain z-norm / CORAL / Subspace Alignment / DANN,
看特征对齐式 DA 能否修复 cross-dataset collapse。对照 baseline(no-adapt)与 target-only 上界(参考)。
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch, torch.nn as nn

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rf(): return RandomForestClassifier(n_estimators=300, random_state=0, class_weight='balanced', n_jobs=-1)


def coral(Xs, Xt, eps=1e-3):
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    def msqrt(C, inv=False):
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
        w = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
        return V @ np.diag(w) @ V.T
    return Xs @ msqrt(Cs, inv=True) @ msqrt(Ct)


def subspace_align(Xs, Xt, d=20):
    d = min(d, Xs.shape[1] - 1)
    Ps = PCA(d).fit(Xs).components_.T; Pt = PCA(d).fit(Xt).components_.T
    return Xs @ Ps @ (Ps.T @ Pt), Xt @ Pt


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l): ctx.l = l; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.l * g, None


class DANN(nn.Module):
    def __init__(self, d, hid=32, rep=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, rep), nn.ReLU())
        self.lab = nn.Linear(rep, 2); self.dom = nn.Sequential(nn.Linear(rep, hid), nn.ReLU(), nn.Linear(hid, 2))
    def forward(self, x, l=0.0):
        r = self.enc(x); return self.lab(r), self.dom(GRL.apply(r, l))


def dann_da(Xs, ys, Xt, seed=0, epochs=250, lam=0.5):
    torch.manual_seed(seed); np.random.seed(seed)
    m = DANN(Xs.shape[1]).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    xs = torch.tensor(Xs, dtype=torch.float32, device=dev); yl = torch.tensor(ys, device=dev)
    xt = torch.tensor(Xt, dtype=torch.float32, device=dev)
    ds = torch.zeros(len(xs), dtype=torch.long, device=dev); dt = torch.ones(len(xt), dtype=torch.long, device=dev)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        lp, dp = m(xs, lam); _, dpt = m(xt, lam)
        loss = ce(lp, yl) + ce(dp, ds) + ce(dpt, dt)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.softmax(m(xt)[0], 1)[:, 1].cpu().numpy()
    return p


def load(p, lab='label'):
    d = pd.read_csv(p); cols = [c for c in FULL if c in d.columns]
    X = d[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = (d[lab] == 'fake').astype(int).values
    return X, y


def target_only_cv(Xt, yt):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); s = np.zeros(len(yt))
    for tr, te in skf.split(Xt, yt):
        s[te] = rf().fit(Xt[tr], yt[tr]).predict_proba(Xt[te])[:, 1]
    return roc_auc_score(yt, s)


def main():
    ctr = load('outputs/ctrsvdd_features_e1.csv')
    sl = load('outputs/features_fixed.csv')
    wild = pd.read_csv('outputs/wildsvdd_features.csv')
    t02 = set(pd.read_csv('/home/admin2/xf/wildsvdd/wildsvdd_bili_t02.csv')['idx'])
    wt = wild[wild['idx'].isin(t02)]
    cols = [c for c in FULL if c in wt.columns]
    Xwt = wt[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    ywt = (wt['label'] == 'fake').astype(int).values

    pairs = [('CtrSVDD->WildT02', ctr[0], ctr[1], Xwt, ywt),
             ('CtrSVDD->SingerLens', ctr[0], ctr[1], sl[0], sl[1]),
             ('SingerLens->CtrSVDD', sl[0], sl[1], ctr[0], ctr[1])]
    rows = []
    for name, Xs, ys, Xt, yt in pairs:
        # 标准化(源 fit)
        sc = StandardScaler().fit(Xs); Xss = sc.transform(Xs); Xts = sc.transform(Xt)
        res = {}
        # baseline no-adapt
        res['baseline'] = roc_auc_score(yt, rf().fit(Xss, ys).predict_proba(Xts)[:, 1])
        # per-domain z-norm (各自标准化)
        Xs_z = StandardScaler().fit_transform(Xs); Xt_z = StandardScaler().fit_transform(Xt)
        res['perdomain_znorm'] = roc_auc_score(yt, rf().fit(Xs_z, ys).predict_proba(Xt_z)[:, 1])
        # CORAL
        Xs_c = coral(Xss, Xts)
        res['CORAL'] = roc_auc_score(yt, rf().fit(Xs_c, ys).predict_proba(Xts)[:, 1])
        # Subspace Alignment
        Xs_a, Xt_a = subspace_align(Xss, Xts)
        res['subspace_align'] = roc_auc_score(yt, rf().fit(Xs_a, ys).predict_proba(Xt_a)[:, 1])
        # DANN (3 seeds)
        res['DANN'] = float(np.mean([roc_auc_score(yt, dann_da(Xs_z, ys, Xt_z, seed=s)) for s in range(3)]))
        # 参考: target-only CV 上界
        res['target_only_ref'] = target_only_cv(Xt, yt)
        for m, a in res.items():
            rows.append(dict(pair=name, method=m, AUC=round(a, 3)))
        print(name, {k: round(v, 3) for k, v in res.items()}, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv('outputs/domain_adaptation_cross.csv', index=False)
    print('\n=== DA cross-dataset AUC ===')
    print(out.pivot(index='pair', columns='method', values='AUC').to_string())
    print('\nSaved -> outputs/domain_adaptation_cross.csv', flush=True)


if __name__ == '__main__':
    main()
