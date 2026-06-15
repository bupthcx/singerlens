from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "svdd_audit" / "results"
OUT = ROOT / "presentation_figures"

COLORS = {
    "real": "#4C78A8",
    "risk": "#D08A00",
    "ok": "#2A9D55",
    "gray": "#8A8F98",
    "dark": "#1F2937",
    "light": "#E8EDF3",
    "danger_fill": "#F8D7DA",
}


def setup() -> None:
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
        }
    )


def finish(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cross_dataset_collapse() -> None:
    df = pd.read_csv(RESULTS / "cross_dataset_aasist_vs_rf.csv")
    within = df.iloc[:2]
    cross = df.iloc[2:]
    labels = ["AASIST", "手工特征 RF"]
    within_vals = [within["AASIST_AUC"].mean(), within["RF_FULL_AUC"].mean()]
    cross_vals = [cross["AASIST_AUC"].mean(), cross["RF_FULL_AUC"].mean()]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - width / 2, within_vals, width, color=COLORS["real"], label="域内评测")
    ax.bar(x + width / 2, cross_vals, width, color=COLORS["risk"], label="跨数据集")
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.4)
    ax.text(1.45, 0.515, "随机基线 0.5", color=COLORS["gray"], fontsize=10)

    for i, (w, c) in enumerate(zip(within_vals, cross_vals)):
        ax.annotate(
            f"-{w-c:.2f}",
            xy=(i, c + 0.02),
            xytext=(i, w - 0.10),
            ha="center",
            color=COLORS["dark"],
            arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.4),
            fontsize=11,
            fontweight="bold",
        )
        ax.text(i - width / 2, w + 0.025, f"{w:.2f}", ha="center", fontsize=10)
        ax.text(i + width / 2, c + 0.025, f"{c:.2f}", ha="center", fontsize=10)

    ax.set_ylim(0, 1.08)
    ax.set_xticks(x, labels)
    ax.set_ylabel("AUC")
    ax.set_title("基准高分不等于真实泛化：换数据集后接近随机")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", color=COLORS["light"], linewidth=1)
    finish(fig, OUT / "01_cross_dataset_collapse_ppt.png")


def factorial_axis() -> None:
    df = pd.read_csv(RESULTS / "ctrsvdd_factorial_axis.csv")
    full = df[df["feature_set"] == "FULL"].copy()
    full["kind"] = np.where(full["contrast"].str.startswith("para-shift"), "换生成范式\n声码器固定", "换声码器\n范式固定")
    agg = full.groupby("kind", sort=False)["ratio"].agg(["mean", "max"]).reset_index()
    order = ["换声码器\n范式固定", "换生成范式\n声码器固定"]
    vals = [float(full[full["kind"] == k]["ratio"].max()) for k in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(order, vals, color=[COLORS["gray"], COLORS["risk"]], width=0.52)
    ax.set_ylabel("EER 退化倍数")
    ax.set_title("关键转折：声码器相同，只换生成范式也会崩")
    ax.grid(axis="y", color=COLORS["light"])
    ax.set_ylim(0, max(vals) * 1.22)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.5, f"{v:.0f}x", ha="center", fontsize=16, fontweight="bold", color=COLORS["dark"])
    ax.text(1, vals[1] * 0.55, "不是只认声码器\n而是生成流程签名", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    finish(fig, OUT / "02_factorial_89x_ppt.png")


def bandwidth_sweep() -> None:
    df = pd.read_csv(RESULTS / "bandwidth_sweep.csv")
    x = df["cutoff_hz"] / 1000
    fig, ax1 = plt.subplots(figsize=(8.2, 4.7))
    ax1.axvspan(3, 6, color=COLORS["danger_fill"], alpha=0.65, label="危险带 3-6kHz")
    ax1.plot(x, df["mean_fake_score"], marker="o", linewidth=2.4, color=COLORS["risk"], label="真唱平均 AI 分数")
    ax1.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    ax1.set_xlabel("低通截止频率 (kHz，越往右越窄带)")
    ax1.set_ylabel("平均 AI 分数")
    ax1.set_ylim(0, 0.62)
    ax1.invert_xaxis()
    ax1.grid(axis="y", color=COLORS["light"])

    ax2 = ax1.twinx()
    ax2.plot(x, df["flip_rate"], marker="s", linewidth=2.0, color=COLORS["real"], label="真唱误判率")
    ax2.set_ylabel("真唱误判为 AI 的比例")
    ax2.set_ylim(0, 0.55)
    ax1.annotate(
        "6-7kHz 附近开始翻转",
        xy=(6, float(df.loc[df["cutoff_hz"] == 6000, "mean_fake_score"].iloc[0])),
        xytext=(8.4, 0.52),
        arrowprops=dict(arrowstyle="->", color=COLORS["dark"]),
        fontsize=11,
        fontweight="bold",
    )
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    ax1.set_title("只压窄带宽，就能把真唱推向“AI”")
    finish(fig, OUT / "03_bandwidth_threshold_ppt.png")


def msa_curve() -> None:
    df = pd.read_csv(RESULTS / "msa_learning_curve.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.plot(df["k"], df["AUC"], color=COLORS["ok"], marker="o", linewidth=2.6)
    ax.fill_between(df["k"], df["AUC"] - df["std"], df["AUC"] + df["std"], color=COLORS["ok"], alpha=0.15, linewidth=0)
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    for k in [10, 50, 100, 200]:
        row = df[df["k"] == k].iloc[0]
        ax.scatter([k], [row["AUC"]], s=80, color=COLORS["risk"], zorder=3)
        ax.text(k, row["AUC"] + 0.035, f"{k}个\n{row['AUC']:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("目标域标注数")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.48, 0.98)
    ax.set_title("真正有效的出口：少量目标域标签先把模型拉出随机带")
    ax.grid(axis="y", color=COLORS["light"])
    finish(fig, OUT / "04_msa_learning_curve_ppt.png")


def umsa_second_target() -> None:
    df = pd.read_csv(RESULTS / "d3_secondtarget_ctrsvdd_fixedtest.csv")
    keep = df[df["strategy"].isin(["random_balanced", "uncertainty"])].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    style = {
        "random_balanced": ("随机选样", COLORS["gray"], "o"),
        "uncertainty": ("U-MSA 不确定性选样", COLORS["risk"], "o"),
    }
    for strategy, (label, color, marker) in style.items():
        sub = keep[keep["strategy"] == strategy].sort_values("k")
        ax.plot(sub["k"], sub["AUC"], marker=marker, linewidth=2.4, color=color, label=label)
    for k in [50, 100, 200]:
        r = float(keep[(keep["strategy"] == "random_balanced") & (keep["k"] == k)]["AUC"].iloc[0])
        u = float(keep[(keep["strategy"] == "uncertainty") & (keep["k"] == k)]["AUC"].iloc[0])
        ax.text(k, u + 0.018, f"+{u-r:.3f}", ha="center", fontsize=9, color=COLORS["dark"], fontweight="bold")
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("目标域标注预算 k")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.52, 0.84)
    ax.set_title("第二目标域复验：U-MSA 方向成立，但收益更克制")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=COLORS["light"])
    finish(fig, OUT / "05_umsa_second_target_ppt.png")


def repair_attempts() -> None:
    da = pd.read_csv(RESULTS / "domain_adaptation_cross.csv")
    methods = ["baseline", "perdomain_znorm", "CORAL", "subspace_align", "DANN", "target_only_ref"]
    labels = ["原始跨域", "域归一化", "CORAL", "子空间对齐", "DANN", "目标域监督"]
    vals = [float(da[da["method"] == m]["AUC"].mean()) for m in methods]
    colors = [COLORS["gray"]] * 5 + [COLORS["ok"]]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.axhspan(0.45, 0.55, color=COLORS["light"], alpha=0.75, label="随机带")
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.35, 1.04)
    ax.set_ylabel("跨域 AUC")
    ax.set_title("免标注修复基本贴近随机，真正拉开差距的是目标域监督")
    ax.grid(axis="y", color=COLORS["light"])
    ax.tick_params(axis="x", rotation=18)
    finish(fig, OUT / "06_repair_attempts_ppt.png")


def emotion_flip() -> None:
    df = pd.read_csv(RESULTS / "wild_emotion_auc.csv")
    order = ["SingerLens(base)", "WildSVDD"]
    labels = ["域内案例\nSingerLens", "跨域测试\nWildSVDD"]
    vals = [float(df[df["dataset"] == d]["AUC"].iloc[0]) for d in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, vals, color=[COLORS["real"], COLORS["risk"]], width=0.5)
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    ax.text(1.1, 0.515, "随机基线", color=COLORS["gray"], fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("语义线索也会跨域反转", xy=(1, vals[1]), xytext=(0.35, 0.82), arrowprops=dict(arrowstyle="->", color=COLORS["dark"]), fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("AUC")
    ax.set_title("换成情感一致性，也不能免疫跨域问题")
    ax.grid(axis="y", color=COLORS["light"])
    finish(fig, OUT / "07_emotion_consistency_ppt.png")


def simple_vs_complex() -> None:
    ps = pd.read_csv(RESULTS / "per_singer_norm.csv")
    clean = ps[ps["feature_set"] == "CLEAN"].iloc[0]
    dann = pd.read_csv(RESULTS / "dann_singer_invariant.csv")
    vals = {
        "RF 原始特征": float(clean["raw_auc"]),
        "普通 MLP": 0.716,
        "对抗 DANN": float(dann["dann_auc"].mean()),
        "per-singer 归一化": float(clean["z_auc"]),
    }
    labels = list(vals.keys())
    y = list(vals.values())
    colors = [COLORS["gray"], COLORS["risk"], "#E84A5F", COLORS["ok"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    bars = ax.bar(labels, y, color=colors, width=0.58)
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.45, 0.82)
    ax.set_ylabel("跨歌手 LOSO 平均 AUC")
    ax.set_title("复杂模型不一定赢：简单身份归一化反而更稳")
    ax.grid(axis="y", color=COLORS["light"])
    ax.tick_params(axis="x", rotation=12)
    finish(fig, OUT / "08_simple_vs_complex_ppt.png")


def window_mil() -> None:
    df = pd.read_csv(RESULTS / "window_cross_dataset_summary.csv")
    sub = df[(df["protocol"] == "CtrSVDD->Wild_T02") & (df["feature_set"] == "FULL")]
    order = ["clip_mean", "MIL_mean", "MIL_max", "POOL_RICH"]
    label_map = {"clip_mean": "片段均值", "MIL_mean": "窗口均值", "MIL_max": "窗口最大", "POOL_RICH": "丰富池化"}
    vals = [float(sub[sub["method"] == m]["AUC"].iloc[0]) for m in order]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar([label_map[m] for m in order], vals, color=[COLORS["gray"], COLORS["gray"], COLORS["risk"], COLORS["gray"]], width=0.58)
    ax.axhspan(0.45, 0.55, color=COLORS["light"], alpha=0.75)
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.annotate("局部最大化反而更差", xy=(2, vals[2]), xytext=(1.45, 0.64), arrowprops=dict(arrowstyle="->", color=COLORS["dark"]), fontsize=12, fontweight="bold")
    ax.set_ylim(0.35, 0.7)
    ax.set_ylabel("跨域 AUC")
    ax.set_title("窗级 MIL 没救回来：局部“像 AI”会同时抬高真唱")
    ax.grid(axis="y", color=COLORS["light"])
    finish(fig, OUT / "09_window_mil_ppt.png")


def external_case() -> None:
    path = RESULTS / "external_bilibili_BV1pQ4y1G7gV_singerlens_scores.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(df["start_sec"], df["fake_probability"], marker="o", linewidth=2.2, color=COLORS["risk"], label="逐段 AI 概率")
    ax.axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1.2, label="阈值 0.5")
    above = (df["fake_probability"] >= 0.5).mean()
    ax.text(0.02, 0.92, f"已知 AI，但仅 {above:.1%} 片段超过阈值", transform=ax.transAxes, fontsize=12, fontweight="bold", color=COLORS["dark"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("人声时间轴 (秒)")
    ax.set_ylabel("AI 概率")
    ax.set_title("真实平台挑战案例：已知 AI，检测分数仍明显波动")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", color=COLORS["light"])
    finish(fig, OUT / "10_external_bilibili_case_ppt.png")


def main() -> None:
    setup()
    cross_dataset_collapse()
    factorial_axis()
    bandwidth_sweep()
    msa_curve()
    umsa_second_target()
    repair_attempts()
    emotion_flip()
    simple_vs_complex()
    window_mil()
    external_case()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
