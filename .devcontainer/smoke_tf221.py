"""End-to-end smoke test for the TF 2.21 / Keras 3 upgrade.

Builds a HighwayDQNAgent against a real highway-env instance, runs a short
3-episode training loop, and reports action / reward / loss stats. Meant to be
executed inside the Carla_RL devcontainer image:

    docker run --rm --platform linux/arm64 \
        -v "$(pwd):/workspace" \
        -w /workspace/model-sim \
        -e PYTHONPATH=/workspace/model-sim/src \
        -e TF_CPP_MIN_LOG_LEVEL=2 \
        -e CUDA_VISIBLE_DEVICES=-1 \
        carla-rl-dev:merged \
        python /workspace/.devcontainer/smoke_tf221.py
"""
from __future__ import annotations

import sys

import tensorflow as tf
import keras

print(f"Python          : {sys.version.split()[0]}")
print(f"TensorFlow      : {tf.__version__}")
print(f"Keras           : {keras.__version__}")

from highway_rl.agent import HighwayDQNAgent
from highway_rl.environment import HighwayEnvironment

env = HighwayEnvironment(scenario="highway", render_mode=None)
obs, _ = env.reset()
print(f"obs shape       : {obs.shape}")
print(
    f"action_space.n  : {env.action_space.n}  "
    f"(type={type(env.action_space.n).__name__})"
)

agent = HighwayDQNAgent(
    state_size=tuple(obs.shape),
    action_size=env.action_space.n,
    use_mixed_precision=False,
    batch_size=8,
    memory_size=64,
)
print(f"agent build     : OK; q_network params={agent.q_network.count_params()}")

total_r = 0.0
for ep in range(3):
    obs, _ = env.reset()
    ep_r = 0.0
    for _ in range(20):
        a = agent.act(obs, training=True)
        obs2, r, done, trunc, _ = env.step(int(a))
        agent.remember(obs, int(a), float(r), obs2, bool(done))
        ep_r += float(r)
        obs = obs2
        if done or trunc:
            break
    if len(agent.memory) >= agent.batch_size:
        metrics = agent.replay()
        loss = metrics.get("loss", float("nan"))
    else:
        loss = float("nan")
    total_r += ep_r
    print(f"  ep {ep + 1}: reward={ep_r:.2f} loss={loss!r} eps={agent.epsilon:.4f}")

env.close()
print(f"smoke OK; total reward over 3 eps = {total_r:.2f}")
