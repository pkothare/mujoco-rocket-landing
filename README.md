# Ablation Studies for 3D Rocket Landing with PPO

**Deterministic Powered Descent via Deep Reinforcement Learning**

> Systematic ablation studies of PPO pipeline components for 3D rocket first-stage propulsive landing in MuJoCo

---

## Video: Final Trained Policy in Action

<video width="640" height="480" controls>
  <source src="final_model.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Best model achieving 790m autonomous landings with 3.9 m/s touchdown speed and 0.6° tilt*

---

## Executive Summary

This project systematically investigates which components of a modern PPO pipeline are **essential** for learning 3D rocket landings in a custom MuJoCo environment with thrust-vector control, grid fins, and reaction-control (RCS) jets.

**Key Finding:** Naive PPO fails catastrophically on this task—policy entropy diverges and no landings occur. A set of five stabilizers (curriculum learning, reward normalization, observation normalization, reduced entropy coefficient, and early KL stopping) yields a functioning baseline. Systematic ablations across four axes (18 conditions, 3 seeds each) reveal:

- **14 of 18 ablation conditions achieve 0% final success**
- **Curriculum learning, dense reward shaping, and observation normalization are inseparable**
- Removing any one is as catastrophic as removing the reward signal entirely
- Horizontal drift dominates failure modes (accounting for 80–95% of impact speed in failed landings)

---

## Task: 3D Rocket First-Stage Propulsive Landing

**Environment:** Custom MuJoCo simulator with high-fidelity actuation:
- Single gimbaled main engine (thrust-vector control: TVC pitch/yaw)
- Four grid fins for aerodynamic torque
- Four RCS jets for attitude control
- Four deployable landing legs
- Realistic ground-contact physics

**Challenge:** 6-DOF underactuated system with highly nonlinear dynamics and discontinuous terminal events (touchdown). Episodes span from 25m "easy" curriculum stage (Stage 0) to 5km free-fall at 180° tilt (Stage 39).

---

## Ablation Results Summary

| **Axis** | **Condition** | **Max Curriculum Stage** | **Final Success Rate** |
|----------|---------------|--------------------------|----------------------|
| **Reward** | Sparse only | 0 | 0% ± 0% |
| | No altitude reward | 0 | 0% ± 0% |
| | No tilt penalty | 1 | 3% ± 7% |
| **Observation** | No angular velocity | 2 | 0% ± 0% |
| | Noisy obs (σ=0.1) | 3 | 7% ± 5% |
| | 1-step delay | 4 | 0% ± 0% |
| **Action** | No grid fins | 5 | 15% ± 12% |
| | No RCS jets | 4 | 0% ± 0% |
| | No fins + RCS | 3 | 8% ± 12% |
| **Stabilizer** | No observation norm | 1 | 0% ± 0% |
| | No curriculum | 0 | 0% ± 0% |
| | No advantage norm | 2 | 0% ± 0% |
| **Baseline** | Full pipeline | **8** | **40–50%** |

---

## Critical Failures and Diagnosis

### Naive PPO Collapse
Without curriculum learning and reward normalization:
- Policy entropy **doubled** from 21.3 to 44.8 over 960 iterations
- Agent achieved **zero landings** despite 2M environment steps
- Policy converged toward **maximum entropy** (pure randomness), not learning
- **Root cause:** Cumulative episode returns of -30k to -70k swamp the policy gradient; the entropy bonus dominates the objective

### Curriculum Floor Sensitivity
With stabilizers but 200m altitude floor:
- Agent achieved **zero landings** after 1.4M steps
- Learned to minimize per-step penalties via hovering/drifting, but never discovered the throttle–orient–decelerate landing sequence
- KL divergence spiked to 1.0–2.5, causing catastrophic policy updates
- **Solution:** Lowered curriculum floor to 25–790m gradient with 40 stages; increased rollout buffer; added early KL stopping

---

MuJoCo-based 3D rocket first-stage propulsive landing environment with a PPO
training pipeline and systematic ablation experiments for CS 234.

## Setup

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cd rocket_sim
uv sync --project rocket_sim
```

## Project structure

```
rocket_sim/
├── __init__.py
├── rocket.xml          # MJCF rocket model
├── scene.xml           # MJCF scene (ground, skybox, lighting)
├── run_ablations.sh    # Run all ablation experiments
├── core/
│   ├── config.py       # Config dataclass (hyperparams + ablation flags)
│   ├── env.py          # MuJoCo Gymnasium-style environment
│   ├── state.py        # RocketState / RocketAction dataclasses
│   └── utils.py        # Quaternion helpers, autopilot vz target
├── rl/
│   ├── curriculum.py   # CurriculumSchedule + initial-state sampling
│   ├── networks.py     # SquashedActorCritic (tanh-squashed Gaussian)
│   ├── normalization.py# RunningMeanStd + RolloutBuffer
│   ├── obs_act.py      # Observation / action helpers for ablations
│   ├── policies.py     # Hand-designed autopilot & random baseline
│   ├── ppo.py          # PPO trainer (rollout, update, eval, checkpointing)
│   └── reward.py       # Dense shaped reward function (v35)
└── tools/
    ├── train.py        # CLI training entry point
    ├── evaluate.py     # Checkpoint evaluation script
    ├── render.py       # Episode → MP4 video renderer
    ├── dump_episode.py # Per-step diagnostic dumper
    └── plot.py         # Plotting utilities (learning curves, bar charts)
```

## Training

All commands are run from the `project/` directory.

### Single baseline run

```bash
uv run --project rocket_sim python -m rocket_sim.tools.train
```

### Resume from checkpoint

```bash
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --resume results/ppo_baseline/checkpoints/model_2048000.pt \
    --experiment_name ppo_baseline
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--total_timesteps` | 5 000 000 | Total environment steps |
| `--seed` | 0 | Random seed |
| `--experiment_name` | `ppo_baseline` | Output subdirectory name |
| `--output_dir` | `results` | Root results directory |
| `--resume` | — | Checkpoint path to resume from |

### Reward ablations

```bash
# Sparse terminal-only reward (no shaping)
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --reward_mode sparse --experiment_name reward_sparse

# Remove individual reward terms
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_reward_altitude --experiment_name reward_no_altitude

uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_reward_tilt --experiment_name reward_no_tilt

uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_reward_angvel --experiment_name reward_no_angvel

uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_reward_horizontal --experiment_name reward_no_horiz

uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_reward_slow_descent --experiment_name reward_no_slowdesc

uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_reward_legs --experiment_name reward_no_legs
```

### Observation ablations

```bash
# Drop angular velocity from observations
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --obs_mode no_angvel --experiment_name obs_no_angvel

# Add Gaussian sensor noise
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --obs_mode noisy --obs_noise_std 0.1 --experiment_name obs_noisy

# One-step observation delay
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --obs_mode delayed --experiment_name obs_delayed
```

### Action-space ablations

```bash
# Remove grid fins
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --action_mode no_gridfins --experiment_name act_no_gridfins

# Remove RCS thrusters
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --action_mode no_rcs --experiment_name act_no_rcs

# Remove both grid fins and RCS
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --action_mode no_gridfins_rcs --experiment_name act_no_gridfins_rcs
```

### Training-stabilizer ablations

```bash
# Disable observation normalization
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_normalize_obs --experiment_name stab_no_obs_norm

# Disable advantage normalization
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_normalize_advantages --experiment_name stab_no_adv_norm

# Disable entropy regularization
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --entropy_coeff 0.0 --experiment_name stab_no_entropy

# Constant learning rate (no linear decay)
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --lr_schedule constant --experiment_name stab_const_lr

# Disable curriculum (full-difficulty from start)
uv run --project rocket_sim python -m rocket_sim.tools.train \
    --no_curriculum --experiment_name stab_no_curriculum
```

## Run all ablations (3 seeds each)

```bash
# Everything (baseline + all ablations)
./rocket_sim/run_ablations.sh

# Just the ablations (skip baseline)
./rocket_sim/run_ablations.sh ablations

# Or a specific group
./rocket_sim/run_ablations.sh baseline
./rocket_sim/run_ablations.sh reward
./rocket_sim/run_ablations.sh obs
./rocket_sim/run_ablations.sh action
./rocket_sim/run_ablations.sh stabilizer
```

## Evaluation

```bash
uv run --project rocket_sim python -m rocket_sim.tools.evaluate \
    results/ppo_baseline/final_model.pt --episodes 50
```

Metrics reported: success rate, touchdown speed, tilt, horizontal offset, fuel
usage.

## Plotting

```bash
# Learning curves (episode reward)
uv run --project rocket_sim python -m rocket_sim.tools.plot \
    results/baseline_s0 results/reward_sparse_s0 \
    --plot_type reward -o reward_curves.png

# Success rate curves
uv run --project rocket_sim python -m rocket_sim.tools.plot \
    results/* --plot_type success -o success_curves.png

# Multi-seed ablation comparison (mean ± std)
uv run --project rocket_sim python -m rocket_sim.tools.plot \
    results/* --plot_type ablation_success -o ablation_success.png

# Evaluation subplots (success, reward, TD speed, TD tilt)
uv run --project rocket_sim python -m rocket_sim.tools.plot \
    results/* --plot_type eval -o eval_metrics.png

# Final bar chart across conditions
uv run --project rocket_sim python -m rocket_sim.tools.plot \
    results/* --plot_type bar -o bar_chart.png
```

## Rendering episodes

```bash
# Render autopilot baseline
uv run --project rocket_sim python -m rocket_sim.tools.render

# Custom settings
uv run --project rocket_sim python -m rocket_sim.tools.render \
    -o landing.mp4 --seed 42 --num_episodes 5 --fps 30
```
