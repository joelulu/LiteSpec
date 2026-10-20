# Qwen3.5 image/text DLite direct SFT on multiple GPUs

This is the same CE + probability-L1 direct-SFT objective as the original VLM
reference entry point. It is not the paper's two-stage recipe or a DFlash baseline.
Each process loads one frozen target and one trainable draft. Only the draft is
inside DistributedDataParallel. The target's hidden states and logits are computed
locally with no gradients; no SGLang service is used during training.

## Start

Stop any active data-generation client first. Stop this workspace's model servers
with `bash scripts/data/stop_qwen35_vlm.sh` and check `nvidia-smi`.

```bash
source .venv-vlm/bin/activate
TRAIN_DATA="../data/finevision_qwen35_sglang_20000/train.jsonl" \
OUTPUT_DIR="../runs/dlite_vlm_20k_ddp" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
bash scripts/dlite/run_training_vlm.sh
```

The launcher uses `.venv-vlm/bin/python` by default. Override `PYTHON_BIN` if needed.
`TARGET_MODEL` defaults to `$HOME/LMM/lmm_model/Qwen3.5-4B`. It prints the resolved
data and output paths. It does not install dependencies or modify environments.

Defaults: 2 draft layers, 4 selected target layers, sequential rank 64, block size 8,
2 sampled anchors per image, length 4096, learning rate 1e-4, gradient clipping 1.0,
one epoch, at most 1000 optimizer updates, checkpoint every 100 updates. Each GPU
processes one image per update (global batch 8 on eight GPUs). `MAX_SAMPLES=0`
loads all records. `MAX_STEPS` changes the update cap; the one-epoch limit still
applies. Use `CONFIG` with a modified YAML for additional epochs or other settings.
The initial 1000-update run is a pipeline/learning check, not a full reproduction.
Validation data is not used as training data or automatically evaluated.

Each epoch is deterministically shuffled and sharded. If the dataset size is not
divisible by world size, at most world_size - 1 samples are repeated to complete
the final global batch. Rank zero reports the padding, global mean loss, gradient
norm, step time, and global number of processed samples. The latter includes repeats.

## Resume

Use the same data, configuration, world size and output directory:

```bash
OUTPUT_DIR="../runs/dlite_vlm_20k_ddp" \
bash scripts/dlite/run_training_vlm.sh --resume ../runs/dlite_vlm_20k_ddp/last
```

The `last` directory contains inference-compatible draft weights/config, optimizer,
rank-local data cursor and per-rank Python/Torch/CUDA RNG states. Only rank zero
writes. All ranks participate in checkpoint collection. Changing world size or
configuration when resuming is rejected. An uncompleted step is replayed from the
last completed checkpoint. Single-process checkpoints from the old entry point
remain readable in single-process mode.

## Verification and limits

`tests/test_vlm_training.py` covers original single-device image backward and resume.
`tests/test_vlm_distributed.py` checks eight-rank sharding and launches real two-process
CPU/Gloo image training with a tiny Qwen3.5 target, comparing interrupted/resumed
weights, optimizer and per-rank RNG with an uninterrupted run. This does not replace
the required H100/NCCL launch check on the user's machine. The trainer intentionally
does not bypass sample errors or retry failed ranks independently; torchrun aborts
the job so all ranks restart from the same checkpoint.

In the implementation environment, CPU image backward, single-process resume,
objective-wrapper loss/gradient parity and eight-rank shard coverage passed. The
two-process test is skipped specifically when the execution sandbox rejects Gloo
TCP device creation (`Operation not permitted`); it must be run on an unrestricted
training host to verify communication and distributed resume.
