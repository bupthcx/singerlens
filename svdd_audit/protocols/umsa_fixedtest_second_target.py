"""Fixed-test U-MSA replication on a second target domain.

This is a generic version of the WildSVDD-T02 active-learning MSA script.
For every trial, it reserves a fixed stratified test split and lets each
selection strategy query labels only from the remaining pool. This avoids the
artifact where active learning removes hard examples from the evaluation set.

Example:
    python scripts/umsa_fixedtest_second_target.py --target ctrsvdd
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MFCC = [f"mfcc_{i}_{s}" for i in range(1, 14) for s in ("mean", "std")]
VRI = [
    "vibrato_rate_mean",
    "vibrato_rate_std",
    "vibrato_depth_mean",
    "vibrato_depth_std",
    "periodicity_score",
    "micro_variation",
    "vri_score",
]
VQ = ["hnr_mean", "hnr_std", "hnr_low_ratio"]
FULL = (
    ["rms_mean", "rms_std", "energy_dynamic", "spectral_flatness_mean", "spectral_flatness_std"]
    + MFCC
    + ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range_semitones", "f0_jitter"]
    + VRI
    + VQ
    + ["long_note_stability"]
)

DEFAULT_KS = [6, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]


def rf(n_estimators: int, seed: int, n_jobs: int) -> Pipeline:
    return Pipeline(
        [
            ("sc", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=seed,
                    class_weight="balanced",
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def load_target(target: str, path: str | None) -> tuple[pd.DataFrame, str]:
    if path:
        return pd.read_csv(path), Path(path).stem
    if target == "ctrsvdd":
        return pd.read_csv("outputs/ctrsvdd_features_e1.csv"), "ctrsvdd_e1"
    if target == "singerlens":
        return pd.read_csv("outputs/features_fixed.csv"), "singerlens_fixed"
    if target == "wild":
        wild = pd.read_csv("outputs/wildsvdd_features.csv")
        t02 = set(pd.read_csv("/home/admin2/xf/svdd_workspace/wildsvdd/wildsvdd_bili_t02.csv")["idx"])
        return wild[wild["idx"].isin(t02)].reset_index(drop=True), "wildsvdd_t02"
    raise ValueError(f"unknown target: {target}")


def balanced_draw(rng: np.random.RandomState, pool: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    ir = pool[y[pool] == 0]
    iff = pool[y[pool] == 1]
    kr = k // 2
    kf = k - kr
    return np.concatenate(
        [
            rng.choice(ir, min(kr, len(ir)), replace=False),
            rng.choice(iff, min(kf, len(iff)), replace=False),
        ]
    )


def auc_fixed(
    X: np.ndarray,
    y: np.ndarray,
    labeled: list[int] | np.ndarray,
    test: np.ndarray,
    n_estimators: int,
    seed: int,
) -> float:
    labeled = np.array(sorted(set(int(i) for i in labeled)))
    if len(np.unique(y[labeled])) < 2:
        return np.nan
    model = rf(n_estimators=n_estimators, seed=seed, n_jobs=-1).fit(X[labeled], y[labeled])
    return float(roc_auc_score(y[test], model.predict_proba(X[test])[:, 1]))


def grow_uncertainty(
    X: np.ndarray,
    y: np.ndarray,
    labeled: list[int],
    pool: np.ndarray,
    target_k: int,
    rng: np.random.RandomState,
    batch: int,
    proxy_trees: int,
) -> list[int]:
    labeled = list(labeled)
    while len(labeled) < target_k:
        rem = np.setdiff1d(pool, labeled)
        if len(rem) == 0:
            break
        b = min(batch, target_k - len(labeled), len(rem))
        if len(np.unique(y[labeled])) < 2:
            picks = rng.choice(rem, b, replace=False)
        else:
            model = rf(n_estimators=proxy_trees, seed=0, n_jobs=1).fit(X[labeled], y[labeled])
            p = model.predict_proba(X[rem])[:, 1]
            picks = rem[np.argsort(np.abs(p - 0.5))[:b]]
        labeled.extend(int(x) for x in picks)
    return labeled


def grow_kcenter(Xs: np.ndarray, labeled: list[int], pool: np.ndarray, target_k: int) -> list[int]:
    labeled = list(labeled)
    poolset = set(int(i) for i in pool.tolist())
    dmin = np.full(Xs.shape[0], np.inf)
    for j in labeled:
        dmin = np.minimum(dmin, np.linalg.norm(Xs - Xs[j], axis=1))
    mask = np.ones(Xs.shape[0], dtype=bool)
    mask[[i for i in range(Xs.shape[0]) if i not in poolset]] = False
    while len(labeled) < target_k:
        cand = np.where(mask)[0]
        if len(cand) == 0:
            break
        pick = int(cand[np.argmax(dmin[cand])])
        labeled.append(pick)
        mask[pick] = False
        dmin = np.minimum(dmin, np.linalg.norm(Xs - Xs[pick], axis=1))
    return labeled


def milestone_line(res: pd.DataFrame, strategy: str) -> str:
    d = res[res.strategy == strategy].sort_values("k")
    parts = []
    for mil in (0.75, 0.80, 0.85, 0.90):
        hit = d[d.AUC >= mil]
        parts.append(f"{mil:.2f}->k={int(hit.iloc[0].k) if len(hit) else '>max'}")
    return "  ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["ctrsvdd", "singerlens", "wild"], default="ctrsvdd")
    ap.add_argument("--input-csv", default=None)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--ntrial", type=int, default=20)
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--seed-k", type=int, default=6)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--trees", type=int, default=220)
    ap.add_argument("--proxy-trees", type=int, default=80)
    ap.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    args = ap.parse_args()

    df, default_name = load_target(args.target, args.input_csv)
    cols = [c for c in FULL if c in df.columns]
    if not cols:
        raise ValueError("no FULL feature columns found")
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = (df["label"] == "fake").astype(int).values
    Xs = StandardScaler().fit_transform(X)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    out_prefix = args.out_prefix or f"outputs/d3_secondtarget_{default_name}_fixedtest"

    print(
        f"Target={args.target} rows={len(y)} real={int((y == 0).sum())} fake={int((y == 1).sum())} "
        f"features={len(cols)} fixed_test={args.test_frac:.0%} trials={args.ntrial}",
        flush=True,
    )

    strategies = ["random_balanced", "uncertainty", "kcenter"]
    acc = {s: {k: [] for k in ks} for s in strategies}
    sss = StratifiedShuffleSplit(n_splits=args.ntrial, test_size=args.test_frac, random_state=17)

    for t, (pool_idx, test_idx) in enumerate(sss.split(X, y)):
        pool = np.array(pool_idx)
        test = np.array(test_idx)
        rng = np.random.RandomState(1000 + t)

        for k in ks:
            kk = min(k, len(pool))
            labels = balanced_draw(rng, pool, y, kk)
            acc["random_balanced"][k].append(auc_fixed(X, y, labels, test, args.trees, seed=t))

        seed = list(balanced_draw(rng, pool, y, args.seed_k))
        for strat in ["uncertainty", "kcenter"]:
            labeled = list(seed)
            for k in ks:
                kk = min(k, len(pool))
                if kk > len(labeled):
                    if strat == "uncertainty":
                        labeled = grow_uncertainty(
                            X, y, labeled, pool, kk, rng, args.batch, args.proxy_trees
                        )
                    else:
                        labeled = grow_kcenter(Xs, labeled, pool, kk)
                acc[strat][k].append(auc_fixed(X, y, labeled, test, args.trees, seed=t))

        if (t + 1) % 5 == 0:
            print(f"  trial {t + 1}/{args.ntrial}", flush=True)

    rows = []
    for strategy in strategies:
        for k in ks:
            v = np.array(acc[strategy][k], dtype=float)
            v = v[~np.isnan(v)]
            rows.append(
                {
                    "target": args.target,
                    "strategy": strategy,
                    "k": k,
                    "AUC": round(float(np.mean(v)), 3),
                    "std": round(float(np.std(v)), 3),
                    "ntrial": int(len(v)),
                }
            )
    res = pd.DataFrame(rows)
    csv_path = f"{out_prefix}.csv"
    png_path = f"{out_prefix}.png"
    res.to_csv(csv_path, index=False)

    print("\n=== Fixed-test U-MSA replication ===")
    print(res.pivot(index="k", columns="strategy", values="AUC").to_string())
    print("\n=== Milestones ===")
    for strategy in strategies:
        print(f"{strategy:16s} {milestone_line(res, strategy)}")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    palette = {"random_balanced": "#888888", "uncertainty": "#e5484d", "kcenter": "#2563eb"}
    for strategy in strategies:
        d = res[res.strategy == strategy].sort_values("k")
        ax.errorbar(
            d.k,
            d.AUC,
            yerr=d["std"],
            marker="o",
            capsize=3,
            color=palette[strategy],
            label=strategy,
        )
    ax.axhline(0.5, color="gray", linewidth=0.7)
    ax.set_xlabel("# labeled target clips (k)")
    ax.set_ylabel(f"fixed held-out {args.target} AUC")
    ax.set_title(f"U-MSA replication on {args.target} (fixed test set)")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)

    marker = f"{out_prefix}.DONE"
    Path(marker).write_text("done\n", encoding="utf-8")
    print(f"\nSaved -> {csv_path}, {png_path}", flush=True)


if __name__ == "__main__":
    main()
