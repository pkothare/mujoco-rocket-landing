"""Plotting utilities for PPO ablation experiments.

Usage:
    # Compare learning curves across experiments
    uv run python plot.py results/ppo_baseline \
        results/reward_sparse results/obs_no_angvel

    # Specific plot types
    uv run python plot.py results/*_s0 results/*_s1 results/*_s2 \
          --group_by_ablation
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(experiment_dir: str) -> list[dict]:
    """Load metrics.jsonl from an experiment directory."""
    path = Path(experiment_dir) / "metrics.jsonl"
    if not path.exists():
        print(f"Warning: {path} not found")
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curves(
    experiment_dirs: list[str],
    metric: str = "ep_reward_mean",
    ylabel: str = "Episode Reward",
    title: str = "Learning Curves",
    smooth_window: int = 10,
    output_path: str | None = None,
):
    """Plot a metric over training steps for multiple experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_dir in experiment_dirs:
        records = load_metrics(exp_dir)
        if not records:
            continue
        name = Path(exp_dir).name
        steps = [r["global_step"] for r in records if metric in r]
        values = [r[metric] for r in records if metric in r]
        if not values:
            continue
        smoothed = smooth(values, smooth_window)
        offset = len(values) - len(smoothed)
        ax.plot(steps[offset:], smoothed, label=name, alpha=0.9)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_success_rate(
    experiment_dirs: list[str],
    smooth_window: int = 10,
    output_path: str | None = None,
):
    """Plot success rate over training steps."""
    plot_learning_curves(
        experiment_dirs,
        metric="ep_success_rate",
        ylabel="Landing Success Rate",
        title="Landing Success Rate During Training",
        smooth_window=smooth_window,
        output_path=output_path,
    )


def plot_eval_metrics(
    experiment_dirs: list[str],
    output_path: str | None = None,
):
    """Plot evaluation metrics over training."""
    metrics = [
        ("eval_success_rate", "Success Rate"),
        ("eval_reward_mean", "Mean Reward"),
        ("eval_td_speed_mean", "Touchdown Speed (m/s)"),
        ("eval_td_tilt_mean", "Touchdown Tilt ( deg)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for exp_dir in experiment_dirs:
            records = load_metrics(exp_dir)
            if not records:
                continue
            name = Path(exp_dir).name
            steps = [r["global_step"] for r in records if metric in r]
            values = [r[metric] for r in records if metric in r]
            if values:
                ax.plot(
                    steps,
                    values,
                    label=name,
                    marker=".",
                    markersize=3,
                    alpha=0.8,
                )
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def _group_experiments(experiment_dirs: list[str]) -> dict[str, list[str]]:
    """Group experiment directories by ablation name (strip seed suffix)."""
    groups: dict[str, list[str]] = {}
    for d in experiment_dirs:
        name = Path(d).name
        # Strip _s0, _s1, _s2 suffix to group seeds
        base = name
        for suffix in ["_s0", "_s1", "_s2", "_s3", "_s4"]:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                break
        groups.setdefault(base, []).append(d)
    return groups


def plot_ablation_comparison(
    experiment_dirs: list[str],
    metric: str = "ep_reward_mean",
    ylabel: str = "Episode Reward",
    title: str = "Ablation Comparison",
    smooth_window: int = 20,
    output_path: str | None = None,
):
    """Plot mean +/- std across seeds for grouped ablations."""
    groups = _group_experiments(experiment_dirs)
    fig, ax = plt.subplots(figsize=(12, 7))

    for group_name, dirs in sorted(groups.items()):
        all_steps = []
        all_values = []
        for d in dirs:
            records = load_metrics(d)
            steps = [r["global_step"] for r in records if metric in r]
            values = [r[metric] for r in records if metric in r]
            if values:
                s = smooth(values, smooth_window)
                offset = len(values) - len(s)
                all_steps.append(np.array(steps[offset:]))
                all_values.append(s)

        if not all_values:
            continue

        # Align to shortest run
        min_len = min(len(v) for v in all_values)
        aligned = np.array([v[:min_len] for v in all_values])
        steps_aligned = all_steps[0][:min_len]
        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)

        line = ax.plot(steps_aligned, mean, label=group_name, alpha=0.9)[0]
        ax.fill_between(
            steps_aligned,
            mean - std,
            mean + std,
            alpha=0.15,
            color=line.get_color(),
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_final_bar_chart(
    experiment_dirs: list[str],
    output_path: str | None = None,
):
    """Bar chart of final evaluation metrics across experiments."""
    groups = _group_experiments(experiment_dirs)
    names = []
    success_means, success_stds = [], []
    speed_means, tilt_means, fuel_means = [], [], []

    for group_name, dirs in sorted(groups.items()):
        sr_vals, sp_vals, ti_vals, fu_vals = [], [], [], []
        for d in dirs:
            records = load_metrics(d)
            eval_records = [r for r in records if "eval_success_rate" in r]
            if eval_records:
                last = eval_records[-1]
                sr_vals.append(last.get("eval_success_rate", 0))
                sp_vals.append(last.get("eval_td_speed_mean", 0))
                ti_vals.append(last.get("eval_td_tilt_mean", 0))
                fu_vals.append(last.get("eval_fuel_mean", 0))
        if sr_vals:
            names.append(group_name)
            success_means.append(np.mean(sr_vals))
            success_stds.append(np.std(sr_vals))
            speed_means.append(np.mean(sp_vals) if sp_vals else 0)
            tilt_means.append(np.mean(ti_vals) if ti_vals else 0)
            fuel_means.append(np.mean(fu_vals) if fu_vals else 0)

    if not names:
        print("No evaluation data found.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(names))

    axes[0].bar(x, success_means, yerr=success_stds, capsize=3, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Landing Success Rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, speed_means, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Touchdown Speed (m/s)")
    axes[1].set_title("Touchdown Speed")

    axes[2].bar(x, fuel_means, alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    axes[2].set_ylabel("Fuel Usage (sum throttle)")
    axes[2].set_title("Fuel Consumption")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def _get_peak_and_final_metrics(
    experiment_dirs: list[str],
) -> dict[str, dict]:
    """Extract peak and final eval metrics per group (mean +/- std across seeds)."""
    groups = _group_experiments(experiment_dirs)
    results = {}
    for group_name, dirs in sorted(groups.items()):
        seed_data = []
        for d in dirs:
            records = load_metrics(d)
            if not records:
                continue
            eval_recs = [r for r in records if "eval_success_rate" in r]
            max_stage = max(
                (r.get("curriculum_stage", 0) for r in records), default=0
            )
            peak_sr = (
                max(r["eval_success_rate"] for r in eval_recs)
                if eval_recs
                else 0.0
            )
            final_sr = eval_recs[-1]["eval_success_rate"] if eval_recs else 0.0
            final_stage = records[-1].get("curriculum_stage", 0)
            seed_data.append(
                {
                    "max_stage": max_stage,
                    "peak_sr": peak_sr,
                    "final_sr": final_sr,
                    "final_stage": final_stage,
                }
            )
        if seed_data:
            results[group_name] = {
                "max_stage_mean": np.mean([s["max_stage"] for s in seed_data]),
                "max_stage_max": max(s["max_stage"] for s in seed_data),
                "peak_sr_mean": np.mean([s["peak_sr"] for s in seed_data]),
                "peak_sr_std": np.std([s["peak_sr"] for s in seed_data]),
                "final_sr_mean": np.mean([s["final_sr"] for s in seed_data]),
                "final_sr_std": np.std([s["final_sr"] for s in seed_data]),
                "n_seeds": len(seed_data),
            }
    return results


_CONDITION_LIST = [
    ("Reward", "reward_sparse"),
    ("Reward", "reward_no_altitude"),
    ("Reward", "reward_no_tilt"),
    ("Reward", "reward_no_angvel"),
    ("Reward", "reward_no_horiz"),
    ("Reward", "reward_no_slowdesc"),
    ("Reward", "reward_no_legs"),
    ("Obs", "obs_no_angvel"),
    ("Obs", "obs_noisy"),
    ("Obs", "obs_delayed"),
    ("Action", "act_no_gridfins"),
    ("Action", "act_no_rcs"),
    ("Action", "act_no_gridfins_rcs"),
    ("Stab", "stab_no_obs_norm"),
    ("Stab", "stab_no_adv_norm"),
    ("Stab", "stab_no_entropy"),
    ("Stab", "stab_const_lr"),
    ("Stab", "stab_curriculum"),
]


def print_ablation_table(results_dir: str) -> None:
    """Print a summary table of all ablation results."""
    results_path = Path(results_dir)
    all_dirs = []
    for _, cond in _CONDITION_LIST:
        for s in range(3):
            d = results_path / f"{cond}_s{s}"
            if d.exists():
                all_dirs.append(str(d))

    metrics = _get_peak_and_final_metrics(all_dirs)

    print(f"{'Axis':<8} {'Condition':<25} {'Seeds':>5} "
          f"{'Max Stage':>10} {'Final SR':>12}")
    print("-" * 65)
    for axis, cond in _CONDITION_LIST:
        m = metrics.get(cond)
        if m is None:
            print(f"{axis:<8} {cond:<25} {'N/A':>5}")
            continue
        print(
            f"{axis:<8} {cond:<25} {m['n_seeds']:>5} "
            f"{m['max_stage_max']:>10} "
            f"{m['final_sr_mean']:>5.0%}+/-{m['final_sr_std']:>4.0%}"
        )


# Condition definitions for paper plots
REWARD_CONDITIONS = [
    "reward_sparse", "reward_no_altitude", "reward_no_tilt",
    "reward_no_angvel", "reward_no_horiz", "reward_no_slowdesc",
    "reward_no_legs",
]
OBS_ACT_CONDITIONS = [
    "obs_no_angvel", "obs_noisy", "obs_delayed",
    "act_no_gridfins", "act_no_rcs", "act_no_gridfins_rcs",
]
STAB_CONDITIONS = [
    "stab_no_obs_norm", "stab_no_adv_norm", "stab_no_entropy",
    "stab_const_lr", "stab_curriculum",
]
ALL_CONDITIONS = REWARD_CONDITIONS + OBS_ACT_CONDITIONS + STAB_CONDITIONS

# Human-readable labels for paper figures
DISPLAY_NAMES = {
    "reward_sparse": "Sparse only",
    "reward_no_altitude": "No altitude",
    "reward_no_tilt": "No tilt",
    "reward_no_angvel": "No ang. vel.",
    "reward_no_horiz": "No horiz. speed",
    "reward_no_slowdesc": "No slow-descent",
    "reward_no_legs": "No legs",
    "obs_no_angvel": "No ang. vel. obs",
    "obs_noisy": "Noisy (sigma=0.1)",
    "obs_delayed": "1-step delay",
    "act_no_gridfins": "No grid fins",
    "act_no_rcs": "No RCS",
    "act_no_gridfins_rcs": "Gimbal only",
    "stab_no_obs_norm": "No obs. norm",
    "stab_no_adv_norm": "No adv. norm",
    "stab_no_entropy": "No entropy",
    "stab_const_lr": "Constant LR",
    "stab_curriculum": "No curriculum",
}


def _collect_dirs(results_dir: str, conditions: list[str]) -> list[str]:
    """Collect all seed directories for a list of condition names."""
    rp = Path(results_dir)
    dirs = []
    for cond in conditions:
        for s in range(3):
            d = rp / f"{cond}_s{s}"
            if d.exists():
                dirs.append(str(d))
    return dirs


def _load_smoothed_seeds(
    results_dir: str,
    condition: str,
    metric: str,
    smooth_window: int,
    max_step: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load and align metric across seeds, return (steps, mean, std) or Nones."""
    rp = Path(results_dir)
    all_steps, all_values = [], []
    for s in range(3):
        d = rp / f"{condition}_s{s}"
        if not d.exists():
            continue
        records = load_metrics(str(d))
        if max_step is not None:
            records = [r for r in records if r["global_step"] <= max_step]
        steps = [r["global_step"] for r in records if metric in r]
        values = [r[metric] for r in records if metric in r]
        if values:
            sv = smooth(values, smooth_window)
            offset = len(values) - len(sv)
            all_steps.append(np.array(steps[offset:]))
            all_values.append(sv)
    if not all_values:
        return None, None, None
    min_len = min(len(v) for v in all_values)
    aligned = np.array([v[:min_len] for v in all_values])
    return all_steps[0][:min_len], aligned.mean(axis=0), aligned.std(axis=0)


def _load_baseline(
    results_dir: str,
    metric: str,
    smooth_window: int,
    max_step: int = 5_000_000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load baseline (ppo_v40) up to max_step, return (steps, smoothed)."""
    rp = Path(results_dir) / "ppo_v40"
    if not rp.exists():
        return None, None
    records = load_metrics(str(rp))
    records = [r for r in records if r["global_step"] <= max_step]
    steps = [r["global_step"] for r in records if metric in r]
    values = [r[metric] for r in records if metric in r]
    if not values:
        return None, None
    sv = smooth(values, smooth_window)
    offset = len(values) - len(sv)
    return np.array(steps[offset:]), sv


def _plot_individual_ablation(
    results_dir: str,
    condition: str,
    output_path: str,
    metric: str = "ep_success_rate",
    ylabel: str = "Success Rate",
    smooth_window: int = 20,
):
    """Plot one ablation condition (mean +/- std) vs baseline (first 5M steps)."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    fig, ax = plt.subplots(figsize=(5, 3.2))

    # Baseline
    bl_steps, bl_vals = _load_baseline(results_dir, metric, smooth_window)
    if bl_steps is not None:
        ax.plot(bl_steps, bl_vals, label="Baseline", color="#2c3e50",
                linewidth=1.5, alpha=0.85)

    # Ablation (mean +/- std across seeds)
    ab_steps, ab_mean, ab_std = _load_smoothed_seeds(
        results_dir, condition, metric, smooth_window,
    )
    if ab_steps is not None:
        label = DISPLAY_NAMES.get(condition, condition)
        line = ax.plot(ab_steps, ab_mean, label=label,
                       linewidth=1.5, alpha=0.9, color="#e74c3c")[0]
        ax.fill_between(
            ab_steps, ab_mean - ab_std, ab_mean + ab_std,
            alpha=0.15, color=line.get_color(),
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(DISPLAY_NAMES.get(condition, condition))
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(0.15, ax.get_ylim()[1] * 1.1))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def _plot_ablation_panel(
    results_dir: str,
    conditions: list[str],
    title: str,
    output_path: str,
    metric: str = "ep_success_rate",
    ylabel: str = "Success Rate",
    smooth_window: int = 20,
):
    """Plot mean +/- std success curves with paper-quality formatting."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    rp = Path(results_dir)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for cond in conditions:
        dirs = [
            str(rp / f"{cond}_s{s}")
            for s in range(3)
            if (rp / f"{cond}_s{s}").exists()
        ]
        if not dirs:
            continue
        all_steps, all_values = [], []
        for d in dirs:
            records = load_metrics(d)
            steps = [r["global_step"] for r in records if metric in r]
            values = [r[metric] for r in records if metric in r]
            if values:
                s = smooth(values, smooth_window)
                offset = len(values) - len(s)
                all_steps.append(np.array(steps[offset:]))
                all_values.append(s)
        if not all_values:
            continue
        min_len = min(len(v) for v in all_values)
        aligned = np.array([v[:min_len] for v in all_values])
        steps_aligned = all_steps[0][:min_len]
        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)

        label = DISPLAY_NAMES.get(cond, cond)
        line = ax.plot(steps_aligned, mean, label=label, alpha=0.9, linewidth=1.5)[0]
        ax.fill_between(
            steps_aligned, mean - std, mean + std,
            alpha=0.12, color=line.get_color(),
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(0.15, ax.get_ylim()[1] * 1.1))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def _plot_bar_chart_paper(
    results_dir: str,
    conditions: list[str],
    output_path: str,
):
    """Two-panel bar chart: max curriculum stage + final eval SR."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.dpi": 150,
    })
    rp = Path(results_dir)
    names = []
    stage_means, stage_stds = [], []
    sr_means, sr_stds = [], []
    axis_labels = []

    for cond in conditions:
        dirs = [
            str(rp / f"{cond}_s{s}")
            for s in range(3)
            if (rp / f"{cond}_s{s}").exists()
        ]
        seed_stages, seed_srs = [], []
        for d in dirs:
            records = load_metrics(d)
            if not records:
                continue
            max_stage = max(
                (r.get("curriculum_stage", 0) for r in records), default=0
            )
            eval_recs = [r for r in records if "eval_success_rate" in r]
            final_sr = eval_recs[-1]["eval_success_rate"] if eval_recs else 0.0
            seed_stages.append(max_stage)
            seed_srs.append(final_sr)
        if seed_stages:
            names.append(DISPLAY_NAMES.get(cond, cond))
            stage_means.append(np.mean(seed_stages))
            stage_stds.append(np.std(seed_stages))
            sr_means.append(np.mean(seed_srs))
            sr_stds.append(np.std(seed_srs))
            if cond.startswith("reward"):
                axis_labels.append("Reward")
            elif cond.startswith("obs"):
                axis_labels.append("Obs")
            elif cond.startswith("act"):
                axis_labels.append("Action")
            else:
                axis_labels.append("Stabilizer")

    if not names:
        print("No data found.")
        return

    colors = {"Reward": "#e74c3c", "Obs": "#3498db",
              "Action": "#2ecc71", "Stabilizer": "#9b59b6"}
    bar_colors = [colors[a] for a in axis_labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(names))

    ax1.bar(x, stage_means, yerr=stage_stds, capsize=3, alpha=0.85,
            color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Max Curriculum Stage")
    ax1.set_title("Ablation Results: Max Curriculum Stage Reached (mean +/- std, 3 seeds)")
    ax1.axhline(y=8, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.text(len(names) - 0.5, 8.2, "baseline peak", ha="right",
             fontsize=8, alpha=0.6)
    ax1.grid(True, alpha=0.2, axis="y")

    ax2.bar(x, sr_means, yerr=sr_stds, capsize=3, alpha=0.85,
            color=bar_colors, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Final Eval Success Rate")
    ax2.set_title("Ablation Results: Final Evaluation Success Rate (mean +/- std, 3 seeds)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim(0, max(0.6, max(sr_means) * 1.3) if sr_means else 1.0)
    ax2.grid(True, alpha=0.2, axis="y")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    ax1.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# Baseline training-progress figure

RESUME_STEP = 4_300_800  # v35 checkpoint where v40 resumed


def _plot_training_progress(
    results_dir: str,
    output_path: str,
    smooth_window: int = 20,
):
    """Three-panel figure: curriculum stage, success rate, mean reward."""
    rp = Path(results_dir) / "ppo_v40"
    records = load_metrics(str(rp))
    if not records:
        print("Warning: ppo_v40 metrics not found, skipping training_progress")
        return

    steps = np.array([r["global_step"] for r in records])
    stages = np.array([r["curriculum_stage"] for r in records])
    sr = np.array([r["ep_success_rate"] for r in records])
    reward = np.array([r["ep_reward_mean"] for r in records])

    sr_smooth = smooth(sr.tolist(), smooth_window)
    rew_smooth = smooth(reward.tolist(), smooth_window)
    offset = len(sr) - len(sr_smooth)
    steps_smooth = steps[offset:]

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "figure.dpi": 150,
    })
    fig, axes = plt.subplots(3, 1, figsize=(5, 5.5), sharex=True)

    color = "#2c3e50"

    # Top: curriculum stage
    axes[0].plot(steps, stages, color=color, linewidth=0.8, alpha=0.85)
    axes[0].set_ylabel("Curriculum Stage")
    axes[0].grid(True, alpha=0.3)

    # Middle: success rate
    axes[1].plot(steps_smooth, sr_smooth, color=color, linewidth=1.0,
                 alpha=0.85)
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].grid(True, alpha=0.3)

    # Bottom: mean reward
    axes[2].plot(steps_smooth, rew_smooth, color=color, linewidth=1.0,
                 alpha=0.85)
    axes[2].set_ylabel("Mean Reward")
    axes[2].set_xlabel("Environment Steps")
    axes[2].grid(True, alpha=0.3)

    # Dashed vertical line at resume point
    for ax in axes:
        ax.axvline(RESUME_STEP, color="gray", linestyle="--", linewidth=0.8,
                    alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def generate_paper_plots(results_dir: str, output_dir: str) -> None:
    """Generate all ablation figures for the final report.

    Produces 18 individual plots (one per ablation condition vs. baseline),
    the summary bar chart, and the training progress figure.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating paper plots...")
    print("=" * 60)

    # Training progress (Figure 1)
    _plot_training_progress(
        results_dir,
        output_path=str(out / "training_progress.pdf"),
    )

    # Individual ablation-vs-baseline plots (18 total)
    for cond in ALL_CONDITIONS:
        _plot_individual_ablation(
            results_dir, cond,
            output_path=str(out / f"ablation_{cond}.pdf"),
        )

    # Summary bar chart
    _plot_bar_chart_paper(
        results_dir, ALL_CONDITIONS,
        output_path=str(out / "ablation_bar.pdf"),
    )

    print("=" * 60)
    print("All paper plots generated.")
    print("=" * 60)

    # Also print the summary table
    print()
    print_ablation_table(results_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot PPO ablation results")
    parser.add_argument(
        "experiments", nargs="*", help="Experiment result directories"
    )
    parser.add_argument(
        "--plot_type",
        choices=[
            "reward",
            "success",
            "eval",
            "ablation_reward",
            "ablation_success",
            "bar",
            "paper",
            "table",
        ],
        default="reward",
    )
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("-o", "--output", type=str, default=None)
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Results directory (for 'paper' and 'table' plot types)",
    )
    args = parser.parse_args()

    if args.plot_type == "paper":
        rd = args.results_dir or "results"
        od = args.output or "final_report/figs"
        generate_paper_plots(rd, od)
        return

    if args.plot_type == "table":
        rd = args.results_dir or "results"
        print_ablation_table(rd)
        return

    if not args.experiments:
        parser.error("experiments dirs required for this plot type")

    if args.plot_type == "reward":
        plot_learning_curves(
            args.experiments,
            smooth_window=args.smooth,
            output_path=args.output,
        )
    elif args.plot_type == "success":
        plot_success_rate(
            args.experiments,
            smooth_window=args.smooth,
            output_path=args.output,
        )
    elif args.plot_type == "eval":
        plot_eval_metrics(args.experiments, output_path=args.output)
    elif args.plot_type == "ablation_reward":
        plot_ablation_comparison(
            args.experiments,
            metric="ep_reward_mean",
            ylabel="Episode Reward",
            title="Reward Ablation Comparison",
            smooth_window=args.smooth,
            output_path=args.output,
        )
    elif args.plot_type == "ablation_success":
        plot_ablation_comparison(
            args.experiments,
            metric="ep_success_rate",
            ylabel="Success Rate",
            title="Success Rate Ablation Comparison",
            smooth_window=args.smooth,
            output_path=args.output,
        )
    elif args.plot_type == "bar":
        plot_final_bar_chart(args.experiments, output_path=args.output)


if __name__ == "__main__":
    main()
