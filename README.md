# DroneVLA

A small, CPU-only drone vision-language-action project: from the same start, the same
camera image and the same state, the drone is told *"Go to the red box and stop."* or
*"Go to the blue cylinder and stop."* -- and the question is whether it actually flies to a
different target depending on the words. Built end to end: simulator setup, data, a tiny
policy, closed-loop evaluation, and later ONNX serving and a C++ runtime core.

Everything runs on a laptop CPU (no GPU) in PyBullet via
[gym-pybullet-drones](https://github.com/learnsyslab/gym-pybullet-drones). The plan lives in
[`robotics_project/`](robotics_project/) and every measurement is written up under
[`docs/`](docs/).

## Status

| Phase | State |
|---|---|
| 0 — environment and measurement path | done: [`docs/phase0_env.md`](docs/phase0_env.md) |
| realism pass against a real drone spec | done: [`docs/realism_mapping.md`](docs/realism_mapping.md) |
| 1 — env, oracle expert, evaluator | done: [`docs/phase1_env.md`](docs/phase1_env.md) |
| 1 — dataset v0.1, tiny policy, closed loop | next |
| 1 — C++ core (C1) replacing `dronevla/action_adapter.py` | after that |

Measured so far (details and caveats in the docs):

- 34-35 recorded 128x96 frames per wall-second on the CPU path, ~7x real time
- the scripted oracle expert succeeds on 40/40 held-out episodes (20/20 counterfactual pairs)
- a gimbal camera's image is 25-31x steadier than the two ungimballed mounts tried
  (standard deviation of the sky's share of the frame)

## Layout

```
dronevla/        the package: env, task, expert, camera, action contract, realism layers
tests/           pytest: action contract, env API, one fixture per evaluation outcome
scripts/         knob scripts to play with (playground.py, try_env.py) and measurement tools
docs/            write-ups of every phase, with the numbers behind each claim
reports/         raw measurement JSON referenced by the docs
robotics_project/  roadmap and literature survey
```

## Setup

Ubuntu 24.04 (native or WSL2), Python 3.12, `build-essential` and `python3.12-dev`. The
simulator is pinned by commit in `env-lock-candidate.txt` rather than vendored. These are
the same steps the CI workflow runs:

```bash
git clone <this repository> dronevla && cd dronevla
SHA=$(sed -n 's/.*gym-pybullet-drones\.git@\([0-9a-f]\{40\}\).*/\1/p' env-lock-candidate.txt)
git clone https://github.com/learnsyslab/gym-pybullet-drones.git third_party/gym-pybullet-drones
git -C third_party/gym-pybullet-drones checkout --detach "$SHA"

python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install "$(grep '^numpy==' env-lock-candidate.txt)"            # before pybullet, see below
pip install --no-build-isolation "$(grep '^pybullet==' env-lock-candidate.txt)"
grep -v -e '^-e ' -e '^torch==' -e '^stable_baselines3==' env-lock-candidate.txt > /tmp/req.txt
pip install -r /tmp/req.txt
pip install --no-deps -e third_party/gym-pybullet-drones
```

pybullet has no Python 3.12 wheel, so it compiles; numpy must be importable during that
build or `getCameraImage` silently returns Python lists. torch is not needed for anything
above (install the CPU build from `https://download.pytorch.org/whl/cpu` when training
arrives).

## Try it

```bash
python -m pytest tests -q                 # 51 tests
python scripts/try_env.py --headless      # same start, two instructions, two outcomes
python scripts/playground.py --headless   # fly a route; change speed, wind, camera, noise
python -m dronevla.profile_env --frames 1000 --renderer tiny --out reports/env.json
```

Both knob scripts open a PyBullet window without `--headless`. Edit the block at the top of
each file and re-run.

## License

MIT, see [`LICENSE`](LICENSE). Depends on gym-pybullet-drones (MIT) and PyBullet (zlib),
which are installed separately and not redistributed here.
