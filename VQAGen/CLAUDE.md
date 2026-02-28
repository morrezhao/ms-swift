# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VQAGen generates visual question-answering pairs for 3D scene understanding benchmarks (VSIBench). It uses an LLM to produce Python code that computes answers from scene metadata, then executes that code in a sandbox to get verified ground-truth values. Supports 10 question types: object counting, distances, sizes, room area, relative directions (3 variants), and relative distances (3 variants).

## Commands

```bash
# Single dataset (uses all scenes from metadata when no --split_path)
python -m tasks.code_qa_generator \
    --processed_data_path /upfs/enhan/data/processed_data/ScanNetpp \
    --dataset scannetpp \
    --split_type train \
    --output_dir data/qa_output \
    --llm_backend vllm \
    --llm_base_url http://localhost:8000/v1 \
    --llm_model Qwen/Qwen3-32B \
    --use_llm_mc

# All 3 datasets
python -m tasks.code_qa_generator \
    --datasets scannet:/upfs/enhan/data/processed_data/ScanNet \
    --datasets scannetpp:/upfs/enhan/data/processed_data/ScanNetpp \
    --datasets arkitscenes:/upfs/enhan/data/processed_data/ARKitScenes \
    --split_type train \
    --output_dir data/qa_output \
    --llm_backend vllm --llm_model Qwen/Qwen3-32B \
    --use_llm_mc

# Convert to GRPO training format
python -m utils.format_qa \
    --datasets scannet:data/qa_output/train/qa_code_generated_scannet.json:/upfs/enhan/data/processed_data/ScanNet/color/train \
    --datasets scannetpp:data/qa_output/train/qa_code_generated_scannetpp.json:/upfs/enhan/data/processed_data/ScanNetpp/color/train \
    --datasets arkitscenes:data/qa_output/train/qa_code_generated_arkitscenes.json:/upfs/enhan/data/processed_data/ARKitScenes/color/train \
    --output_path data/qa_output/train_grpo.json --shuffle

# Start vLLM server (prerequisite for vllm backend)
vllm serve Qwen/Qwen3-32B --port 8000 --tensor-parallel-size 4
```

Backends: `--llm_backend vllm` (high-perf local), `ollama` (simple local), `cockpit` (API gateway for GPT/Claude/Gemini).

## Architecture

### Pipeline Flow

```
code_qa_generator.py (entry point)
  │
  ├─ LLM call: system prompt (code_prompts.py) + scene context (context_formatter.py)
  │  → LLM returns JSON with question_type, question, code, options
  │
  ├─ Parse: code_output_parser.py extracts CodeGeneratedQA objects
  │
  ├─ Execute: script_executor.py runs compute_answer() in sandboxed subprocess
  │  └─ Sandbox provides: np, math, scene data, helper functions (direction/distance)
  │
  ├─ (Optional) Validate: qa_validator.py → LLM accept/reject check
  │
  └─ Format: attach MC options, balance A/B/C/D distribution → output JSON
```

### Key Design Decisions

- **Code-as-answer**: LLM writes `compute_answer()` Python functions instead of direct answers. Code is executed against scene metadata for verifiable ground truth.
- **Sandboxed execution**: `script_executor.py` runs LLM-generated code in a subprocess with restricted builtins, timeout (30s default), and pre-loaded helper functions. Import statements are stripped automatically.
- **MC options from LLM**: For direction/distance question types, the LLM provides options directly in its output. Numeric types (counting, size, area) don't need MC options. QAs are discarded if the LLM fails to provide valid options.
- **Two-tier validation**: First, code execution must succeed. Then optionally, a second LLM call (`--use_llm_mc`) validates question quality (accept/reject).

### Question Types

Defined in `code_prompts.py` as `VALID_QUESTION_TYPES`:
- **Numeric** (no MC options): `object_counting`, `object_abs_distance`, `object_size_estimation`, `room_size_estimation`
- **MC** (LLM provides options): `object_rel_direction_v1/v2/v3`, `object_rel_distance_v1/v2/v3`

Direction types have fixed option sets (e.g., v1 = front-left/front-right/back-left/back-right). Distance v1/v2 use object category names as options.

### Sandbox Helper Functions (script_executor.py)

Functions available to LLM-generated code — these are defined inside `_execute_in_process()`:
- `get_rel_direction_quadrant/lr/lrb()` — relative direction using OBB vertices and angle calculation (requires open3d)
- `get_closest_among_choices()` — find closest object using `cal_3d_bbox_distance_between_categories`
- `find_closest_object/find_farthest_object()` — centroid-based distance
- `euclidean_distance()`, `get_centroid()`, `get_all_centroids()`

### Scene Metadata Format

Stored in `metadata/` as JSON. Each scene has:
- `object_counts`: `{"chair": 2, "table": 1, ...}`
- `object_bboxes`: `{"chair": [{"centroid": [x,y,z], "axesLengths": [l,w,h], "normalizedAxes": [...]}]}`
- `room_size`: float (sqm), `room_center`: `[x,y,z]`

## Key Dependencies

numpy, open3d, torch, scipy, tqdm, openai, requests

### metadata_path:
ScanNet: /upfs/enhan/data/processed_data/ScanNet/metadata/[split_type]/scannet_metadata_[split_type].json
ScanNet++: /upfs/enhan/data/processed_data/ScanNetpp/metadata/[split_type]/scannetpp_metadata_[split_type].json
ARKitScenes: /upfs/enhan/data/processed_data/ARKitScenes/metadata/[split_type]/arkitscenes_metadata_[split_type].json

### sampled_frames_path:
ScanNet: /upfs/enhan/data/processed_data/ScanNet/color/[split_type]/[scene_id]/[frame_id].jpg
ScanNet++: /upfs/enhan/data/processed_data/ScanNetpp/color/[split_type]/[scene_id]/[frame_id].jpg
ARKitScenes: /upfs/enhan/data/processed_data/ARKitScenes/color/[split_type]/[scene_id]/[frame_id].jpg
each with 32 frames, frame_ids are from small to large, but not necessarily consecutive starting from 0.
