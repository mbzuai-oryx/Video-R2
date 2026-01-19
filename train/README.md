# Training

This folder contains the two-stage post-training pipeline used to build Video-R2:

1. **Stage 1: Timestamp-aware SFT** starting from **Qwen2.5-VL-Instruct**
2. **Stage 2: GRPO** guided by Temporal Alignement Reward **(TAR)**

The main entrypoints are:

- `src/train/train_sft.py`
- `src/train/train_grpo.py`

Sample multi-GPU launch scripts are provided in `scripts/`:

- `scripts/train_sft.sh`
- `scripts/train_grpo.sh`

## 1) Environment

```bash
conda create -n video-r2 python=3.12 -y
conda activate video-r2
pip install -U pip

# We use torch v2.7.0, torchvision v0.22.0 and transformers v2.51.1 in the development of Video-R2
# Please see requirements.txt and environment.yml for all requirements
pip install -r requirements.txt

# Further, we recommend installing flash-attn v2.7.4.post1 or v2.8.3 for training
```

## 2) Download and Arrange the Dataset

We release the datasets used in Video-R2 development on Hugging Face:

- https://huggingface.co/datasets/MBZUAI/Video-R2-Dataset

Download:

```bash
hf download MBZUAI/Video-R2-Dataset --repo-type dataset
```

Arrange (or symlink) into a local layout that matches the paths you will pass to the scripts:

```text
data/
  video-r2-sft-dataset.json
  video-r2-grpo-dataset.json
  videos/
    <video files referenced in the JSON, extract all the .tar files>
  subtitles/            # optional
    <subtitle files matched by video stem, extract subtitles.tar>
```

Important paths used by the scripts:

- `DATA_PATH`: one of the JSON files above
- `DATA_FOLDER`: folder containing videos (passed to the code as `--image_folder`)
- `SUBTITLES_FOLDER`: optional folder containing `.srt` files

## 3) Stage 1: Timestamp-aware SFT (from Qwen2.5-VL-Instruct)

We start from the instruction-tuned Qwen2.5-VL model:

- `Qwen/Qwen2.5-VL-7B-Instruct`

1) Edit `scripts/train_sft.sh`:

- `MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"`
- `DATA_PATH` (SFT JSON)
- `DATA_FOLDER` (videos folder)
- `SUBTITLES_FOLDER` (optional)
- `OUTPUT_DIR`

2) Run:

```bash
cd train
bash scripts/train_sft.sh
```

Notes:

- The script uses LoRA by default.
- Video sampling is controlled by `--fps` and `FPS_MAX_FRAMES`.
- The code supports overlaying timestamps and subtitles on frames via env vars.

## 4) Merge LoRA weights (optional)

Merge the LoRA checkpoints after SFT:

```bash
python src/merge_lora_weights.py \
  --base_model <base_model_id_or_path> \
  --lora_model <lora_checkpoint_path> \
  --output_dir <merged_output_dir>
```

## 5) Stage 2: GRPO

We perform GRPO starting from the SFT checkpoint. GRPO is configured with a mixture of reward functions, including Temporal Alignment Reward (TAR).

1) Edit `scripts/train_grpo.sh`:

- `MODEL_NAME` should point to the SFT merged checkpoints
- `DATA_PATH` (GRPO JSON)
- `DATA_FOLDER` (videos folder)
- `SUBTITLES_FOLDER` (optional)
- `OUTPUT_DIR`

2) Serve a judge / parsing model

TAR uses an LLM judge served behind an OpenAI-compatible API. An example vLLM launcher is provided:

```bash
cd train
bash serve_llm/serve_qwen3.sh
```

You can test the server:

```bash
python serve_llm/test_vllm_client.py
```

3) Run GRPO:

```bash
cd train
bash scripts/train_grpo.sh
```

Key knobs:

- `--num_generations`: rollouts per prompt
- `--beta`: KL coefficient
- `--max_prompt_length`, `--max_completion_length`: token budgets
- TAR parameters: `--buffer_seconds`, `--similarity_threshold`


## Citation ✏️

If you find Video-R2 helpful, please cite:

```bibtex
@article{maaz2025video-r2,
  title={Video-R2: Reinforcing Consistent and Grounded Reasoning in Multimodal Language Models},
  author={Maaz, Muhammad and Rasheed, Hanoona and Khan, Fahad Shahbaz and Khan, Salman},
  journal={arXiv preprint arXiv:2511.23478},
  year={2025}
}
```

---

[<img src="../images/IVAL_logo.png" width="200" height="100">](https://www.ival-mbzuai.com)
[<img src="../images/Oryx_logo.png" width="100" height="100">](https://github.com/mbzuai-oryx)
[<img src="../images/MBZUAI_logo.png" width="360" height="85">](https://mbzuai.ac.ae)
