# IONav

**Observation-Grounded Indoor Object Navigation**

IONav combines a learned goal-reaching policy with observation-grounded control for indoor 3D object navigation. Frozen DINOv2 and CLIP encoders, together with an LSTM actor-critic, provide the nominal policy. Three control modules use current observations and recent history to improve approach, recovery, and stopping:

- **Temporal Approach Correction (TAC)** uses temporal target observations to correct premature stopping and improve the final approach.
- **Observation-Gated Recovery (OGR)** applies bounded recovery actions when observation-based checks indicate repeated obstruction.
- **Range-Gated Stopping (RGS)** combines a learned stopping head with a calibrated range check before allowing an additional early `Done` action.

RGS checks the additional learned stopping proposals; it does not override every `Done` action produced by the nominal policy. The iTHOR experiments use simulator-provided target visibility and 2D bounding boxes. This repository does not supply a learned object detector for deployment.

![IONav overview](docs/figures/ionav_framework.png)

The 3D scene in the overview is illustrative. Online inputs follow the iTHOR RGB-D and target-annotation protocol described below.

The implementation builds on [AION](https://github.com/Zichen-Yan/AION). The `AION/` source directory is retained for compatibility with its imports and data layout.

## Contents

- [Results](#results)
- [Installation](#installation)
- [Data and checkpoints](#data-and-checkpoints)
- [Evaluation](#evaluation)
- [Efficiency benchmarks](#efficiency-benchmarks)
- [Verify archived results](#verify-archived-results)
- [Training and calibration](#training-and-calibration)
- [Evaluation protocol and metrics](#evaluation-protocol-and-metrics)
- [Acknowledgments and licensing](#acknowledgments-and-licensing)

## Results

The reference comparison uses the released DINOv2-based IONav configuration and a corresponding AION baseline. Each row evaluates 600 Unseen episodes with four workers, an action budget of 100, and evaluation seed 1. The two methods share episode identities, simulator-accepted start states, and episode-specific action seeds.

| Class split | Method | Successes | SR (%) ↑ | SPL (%) ↑ | CR (%) ↓ | Actions ↓ | Active time (s) ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| 18/4 | AION baseline | 534/600 | 89.00 | 54.58 | 8.92 | 46.94 | 3.505 |
| 18/4 | **IONav** | **572/600** | **95.33** | **82.76** | **5.77** | **29.22** | **2.409** |
| 14/8 | AION baseline | 474/600 | 79.00 | 56.14 | 7.32 | 48.45 | 3.993 |
| 14/8 | **IONav** | **533/600** | **88.83** | **76.89** | **6.51** | **38.41** | **3.271** |

Actions and active time are episode averages. Active time measures synchronized, instrumented navigation with four workers; it excludes scene reset, post-hoc SPL computation, and log serialization. It is separate from single-model inference latency. See [Evaluation protocol and metrics](#evaluation-protocol-and-metrics) for the exact metric definitions.

Full-precision results and raw records are included in the complete artifact archive under `reference/efficiency/`. Historical repeated evaluations and ablations are preserved separately under `reference/ablation/`, `reference/multisplit/`, and `reference/stop_range/`. The table above reports one consistent evaluation campaign rather than combining metrics from different runs.

## Installation

### Requirements

| Component | Reference environment |
|---|---|
| OS | Linux x86_64, glibc 2.35 |
| GPU | NVIDIA RTX 4090, physical GPU 0 |
| Python | 3.10.22 |
| PyTorch / torchvision | 2.5.1+cu124 / 0.20.1+cu124 |
| AI2-THOR | 5.0.0 |
| AI2-THOR build | `f0825767cd50d69f666c7f282e54abfe58f1e917` |
| timm | 0.9.16 |
| CLIP package | clip-anytorch 2.6.0 |
| NumPy | 1.26.0 |

Navigation evaluation requires NVIDIA CUDA, Vulkan, and Xvfb. The simulator binaries target Linux x86_64. Archived metric verification and command validation also run on macOS with Python 3.9 or later.

Clone this repository, obtain the artifacts described below, and run commands from the repository root. On Ubuntu 22.04 with a working NVIDIA driver:

```bash
sudo apt-get update
sudo apt-get install -y python3.10-venv xvfb xauth libvulkan1 vulkan-tools libgl1

bash environment/install_env.sh
source .venv/bin/activate
python reproduce/prepare_assets.py
```

The installer creates `.venv` and uses the pinned runtime dependencies. `environment/reference_versions.json` records the reference environment; `environment/requirements-full-snapshot.lock` provides its complete package constraints. Python package installation requires access to a package index. Model initialization and simulator execution use the local artifacts.

## Data and checkpoints

The source repository and the complete reproduction archive have different contents. Evaluation requires policy checkpoints, calibration files, fixed episode manifests, iTHOR data, pretrained encoder caches, and the fixed simulator build.

**Artifact availability:** a public download endpoint for the IONav-specific artifacts has not yet been assigned. The full reproduction archive contains the files below; a source-only checkout is insufficient for complete evaluation. Extract the archive contents at the repository root while preserving the directory layout.

```text
.
├── AION/                         # Policy, controllers, training, and evaluation
│   └── data/                     # iTHOR offline data and text embeddings
├── weights/
│   ├── 18_4/                     # final_online.dat and final_training.pt
│   ├── 14_8/                     # final_online.dat and final_training.pt
│   └── original/                 # Corresponding AION baseline checkpoints
├── inputs/
│   ├── calibrations/{18_4,14_8}/  # Four calibrated controller files per split
│   ├── manifests/                # Fixed Seen/Unseen episodes and start states
│   └── materials/                # Seven RGB-D probes for network benchmarks
├── assets/
│   ├── cache/{clip,huggingface}/  # CLIP RN50 and DINOv2 initialization caches
│   └── ai2thor/releases/          # Fixed Linux simulator builds
├── environment/                  # Pinned dependencies and installer
├── reproduce/                    # Portable entry points and verification tools
├── reference/                    # Raw reference runs and analysis scripts
└── provenance/                   # Training records and source provenance
```

### Checkpoint identities

| Split | IONav checkpoint | SHA-256 |
|---|---|---|
| 18/4 | `weights/18_4/final_online.dat` | `8af29d4ab722b4a2a347b12d67c9824773470d0feb27bea008080e5650276ed8` |
| 14/8 | `weights/14_8/final_online.dat` | `ce2b9e00c37378365b4d9e29925f955fd151e9438843696a262d65debe2ebc0a` |

The baseline checkpoints are `weights/original/AION-g-18-4.dat` and `weights/original/AION-g-14-8.dat`. Complete registrations are stored in `reproduce/registered_inputs.json`.

Each split has its own `approach_calibrator.json`, `finish_calibrator.json`, `terminal_head.json`, and `terminal_range_guard.json`. The launchers verify the registered hashes and split-specific bindings. Keep checkpoints, calibration files, and episode manifests from the same split together.

`final_online.dat` is the evaluation state dictionary. `final_training.pt` retains the associated training state. The encoder caches are required by the model constructor even when a complete policy checkpoint is available.

## Evaluation

### Validate the configuration

These commands check the real checkpoint, calibration, manifest, and resource paths without starting CUDA or Unity:

```bash
python reproduce/evaluate.py --split 18/4 --profile ionav --dry-run
python reproduce/evaluate.py --split 14/8 --profile ionav --dry-run
```

The launcher sets the working directory and local cache paths. The original runtime files and the small path/device portability changes are recorded in `provenance/unmodified_runtime/`, `provenance/PORTABILITY_PATCHES.json`, and `provenance/PORTABILITY.patch`.

### Reproduce the reference comparison

Identify physical GPU 0 and use a fresh output directory for every run:

```bash
ionav_gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"

python reproduce/evaluate.py \
  --split 18/4 --subset unseen --profile original --seed 1 \
  --expected-gpu-uuid "$ionav_gpu_uuid" --measure-efficiency \
  --out runs/18_4_original_seed1

python reproduce/evaluate.py \
  --split 18/4 --subset unseen --profile ionav --seed 1 \
  --expected-gpu-uuid "$ionav_gpu_uuid" --measure-efficiency \
  --out runs/18_4_ionav_seed1

python reproduce/evaluate.py \
  --split 14/8 --subset unseen --profile ionav --seed 1 \
  --expected-gpu-uuid "$ionav_gpu_uuid" --measure-efficiency \
  --out runs/14_8_ionav_seed1

python reproduce/evaluate.py \
  --split 14/8 --subset unseen --profile original --seed 1 \
  --expected-gpu-uuid "$ionav_gpu_uuid" --measure-efficiency \
  --out runs/14_8_original_seed1
```

This reproduces the reference run order and produces four 600-episode evaluations. `--profile ionav` enables the complete controller; `--profile original` uses the corresponding AION checkpoint with the additional controllers disabled. The baseline launcher is registered for seed 1.

Use `--subset seen` for the fixed Seen manifests. IONav also accepts evaluation seeds 2 and 3; these reuse the same trained checkpoint. Both CUDA GPU 0 and Unity Vulkan device 0 must identify the intended physical GPU.

### Outputs

| Output | Contents |
|---|---|
| `results.json` | Aggregate metrics and integrity checks |
| `run_meta.json` | Configuration, device identity, and source/input hashes |
| `episodes/*.jsonl` | Episode trajectories and navigation metrics |
| `start_pose/*.jsonl` | Simulator-accepted start states |
| `efficiency/` | Step timings and worker summaries when timing is enabled |
| `status.json` | Completion status and process cleanup |

A complete evaluation contains 600 unique episodes, with 150 episodes per room type. Inspect the integrity status, input hashes, start-state validation, and worker completion before reporting results. Failures and simulator interruptions remain in the primary metric denominator.

## Efficiency benchmarks

`--measure-efficiency` enables the synchronized navigation timing used in the reference table. Step-level latency summaries exclude each worker's first 10 steps; episode action counts and active times include every step. Omitting the flag runs navigation evaluation without producing active-time measurements.

Single-model latency and PyTorch allocator memory are measured separately:

```bash
ionav_gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"

python reproduce/benchmark_network.py \
  --split 18/4 --profile ionav --warmup 50 --iterations 350 \
  --expected-gpu-uuid "$ionav_gpu_uuid" \
  --out runs/network/18_4_ionav
```

Add `--dry-run` to validate the command without GPU execution. Each invocation starts one model process and writes `result.json` and `samples.csv`. The benchmark uses the same seven real RGB-D observations, fixed nonzero recurrent probes, batch size one, and the original model precision.

For the full reference design, run AION, IONav, IONav, AION in sequence and repeat that order three times for each split:

```bash
set -e
ionav_gpu_uuid="$(nvidia-smi --id=0 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
for split in 18/4 14/8; do
  split_directory="${split/\//_}"
  for repetition in 1 2 3; do
    position=0
    for profile in original ionav ionav original; do
      position=$((position + 1))
      python reproduce/benchmark_network.py \
        --split "$split" --profile "$profile" \
        --warmup 50 --iterations 350 \
        --expected-gpu-uuid "$ionav_gpu_uuid" \
        --out "runs/network/${split_directory}/rep${repetition}_${position}_${profile}"
    done
  done
done
```

This produces 24 independent processes and 8,400 measured calls. For each split and method, report the mean and sample standard deviation across six process-level means. Repeated calls within one process are not independent experiments. Run the benchmark on an otherwise idle GPU and report its hardware and software configuration.

## Verify archived results

The full artifact archive includes the original records and a SHA-256 inventory. The following commands require only Python 3.9 or later:

```bash
python3 reproduce/verify_files.py --root .
python3 reproduce/recompute_metrics.py --root . --out outputs/archived_metrics

python3 reference/efficiency/scripts/analyze_fresh_efficiency.py \
  --root reference/efficiency --out outputs/archived_efficiency
```

The navigation verifier independently rebuilds SR, SPL, and action-weighted CR from episode logs and checks paired tasks, accepted start states, and action seeds. It writes `reproduction_metrics.csv` and `reproduction_metrics.json` and exits with a nonzero status on inconsistency.

The full efficiency analyzer also reconstructs timing and network statistics from the archived four navigation runs and 24 network processes. Its inputs include the original coordinator protocols; it is intended for `reference/efficiency/`, rather than a newly generated `runs/` directory. New evaluations provide their own `results.json`, timing logs, and network `result.json` files.

## Training and calibration

The evaluation checkpoints continue the matching AION-g initialization for 8,192 policy transitions. The recorded configuration uses four workers, a learning rate of `3e-5`, rollout length 128, an episode action budget of 50, and training seed 1. The 18/4 checkpoint uses the `control` arm without the experimental predictive auxiliary head.

Policy continuation is implemented in `AION/aiong_recovery/training.py`. Exact settings, optimizer/checkpoint provenance, and completed training budgets are recorded in:

- `provenance/train18_4/run_manifest.json`
- `provenance/train14_8/run_manifest.json`

Controller parameters were fitted and calibrated using the corresponding Seen development data. The supplied calibration JSON files preserve the parameters and their checkpoint bindings; the experiment records in `provenance/` document the original sequence. This release supports checkpoint-based evaluation and retains policy-continuation code, but does not provide a single portable workflow for refitting every controller component.

Use the supplied checkpoints and calibration parameters to reproduce the evaluation protocol above. Retraining or recalibrating produces a new model/controller configuration that requires its own evaluation. Asynchronous training and simulator execution do not guarantee byte-identical checkpoints.

## Evaluation protocol and metrics

The 18/4 and 14/8 splits refer to Seen/Unseen **target categories**. Each formal evaluation uses four room types, 150 episodes per type, the registered start states, local inference, categorical action sampling, and a 100-action budget. Feature caching and shared inference are disabled for the reference configuration.

**Inputs.** Target visibility and target bounding boxes are provided by iTHOR annotations. Ground-truth distance and motion information are recorded for supervision and evaluation diagnostics. The additional controller decisions use the observed cues and their history; the implementation does not include a learned detector benchmark.

**Success rate (SR).** Success requires a `Done` action when the original environment criteria hold: target visibility, a 3D distance of at most 1.5 m from the agent to the reference coordinates encoded in the target `objectId`, and a union target-box center inside the original center region. `threshold=0.8` corresponds to normalized x and y coordinates in `[0.1, 0.9]`.

**SPL.** The release retains `AION/runners/train_util.py::compute_spl`. Failed episodes receive zero. For a successful episode, the implementation uses the shortest grid path from its actual start state to its actual final state and the number of translation actions:

```text
SPL_i = success_i * best_path_length_i / max(best_path_length_i, move_steps_i)
SPL   = mean(SPL_i)
CR    = sum(total_collision_i) / sum(ep_length_i)
```

Zero-length and nonfinite-path cases follow the retained function. This SPL implementation uses the episode endpoint and translation count; it is not interchangeable with every target-geodesic SPL implementation used by other navigation benchmarks.

**Collision rate (CR).** CR is the action-weighted legacy collision count divided by the total action count, including turns and `Done`. It is not an average of per-episode ratios or the translation-only physical obstruction rate.

**Timing and memory.** Audited navigation time includes simulation and synchronization under four-worker execution. The isolated network benchmark excludes model loading, agent preprocessing, host-to-device transfer, sampling, controllers, Unity, and logging. Its allocator memory excludes the simulator and driver. Improvements in mean navigation cost do not imply that every tail-latency metric improves.

**Repeatability.** Unity/CUDA reruns may produce small observation and trajectory differences even with the same checkpoints, manifests, and action seeds. Preserve each run's metadata and report the measured result. Evaluation seeds are not independent training seeds. The 18/4 Seen development history and reuse of the Unseen benchmark across experiments are documented in the archived protocols.

## Acknowledgments and licensing

This project builds on [AION](https://github.com/Zichen-Yan/AION), upstream commit `3aac2c500fc6c036372fdb22ff9b96a3a4864534`, and uses AI2-THOR, DINOv2, timm, and OpenAI CLIP. Upstream attribution and citation information are retained in `provenance/UPSTREAM_README.md`.

Third-party code, data, model weights, and simulator binaries retain their respective terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No project-wide license is specified in this repository.
