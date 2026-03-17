#!/usr/bin/env bash
# Run all PPO ablation experiments for the 3D rocket landing project.

set -euo pipefail
cd "$(dirname "$0")/.."

# Activate the project venv so 'python' resolves to the right interpreter
source rocket_sim/.venv/bin/activate

SEEDS="0 1 2"
STEPS=5000000       # 5M timesteps per run, adjust as needed
COMMON="--total_timesteps $STEPS"

run() {
    local name="$1"; shift
    for seed in $SEEDS; do
        echo "=== $name  seed=$seed ==="
        python -m rocket_sim.tools.train $COMMON --seed "$seed" \
            --experiment_name "${name}_s${seed}" "$@"
    done
}

# Baseline
run_baseline() {
    run "baseline" # all defaults
}

# Reward ablations
run_reward() {
    run "reward_sparse"         --reward_mode sparse
    run "reward_no_altitude"    --no_reward_altitude
    run "reward_no_tilt"        --no_reward_tilt
    run "reward_no_angvel"      --no_reward_angvel
    run "reward_no_horiz"       --no_reward_horizontal
    run "reward_no_slowdesc"    --no_reward_slow_descent
    run "reward_no_legs"        --no_reward_legs
}

# Observation ablations
run_obs() {
    run "obs_no_angvel"  --obs_mode no_angvel
    run "obs_noisy"      --obs_mode noisy --obs_noise_std 0.1
    run "obs_delayed"    --obs_mode delayed
}

# Action-space ablations
run_action() {
    run "act_no_gridfins"     --action_mode no_gridfins
    run "act_no_rcs"          --action_mode no_rcs
    run "act_no_gridfins_rcs" --action_mode no_gridfins_rcs
}

# Training-stabilizer ablations
run_stabilizer() {
    run "stab_no_obs_norm"  --no_normalize_obs
    run "stab_no_adv_norm"  --no_normalize_advantages
    run "stab_no_entropy"   --entropy_coeff 0.0
    run "stab_const_lr"     --lr_schedule constant
    run "stab_curriculum"   --no_curriculum
}

# Dispatch
GROUP="${1:-all}"

case "$GROUP" in
    baseline)    run_baseline ;;
    reward)      run_reward ;;
    obs)         run_obs ;;
    action)      run_action ;;
    stabilizer)  run_stabilizer ;;
    ablations)
        run_reward
        run_obs
        run_action
        run_stabilizer
        ;;
    all)
        run_baseline
        run_reward
        run_obs
        run_action
        run_stabilizer
        ;;
    *)
        echo "Unknown group: $GROUP"
        echo "Usage: $0 {all|baseline|ablations|reward|obs|action|stabilizer}"
        exit 1
        ;;
esac

echo ""
echo "=== All requested experiments complete ==="
echo ""
echo "Generate plots with:"
echo "  uv run --project rocket_sim python -m rocket_sim.tools.plot results/*  --plot_type ablation_reward -o ablation_reward.png"
echo "  uv run --project rocket_sim python -m rocket_sim.tools.plot results/*  --plot_type ablation_success -o ablation_success.png"
echo "  uv run --project rocket_sim python -m rocket_sim.tools.plot results/*  --plot_type bar -o ablation_bar.png"
