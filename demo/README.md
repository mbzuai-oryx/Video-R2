# Demo

This folder provides a simple Gradio UI to run **Video-R2** on a single video using Hugging Face Transformers.

- Model: https://huggingface.co/MBZUAI/Video-R2

## 1) Install

```bash
conda create -n video-r2 python=3.12 -y
conda activate video-r2
pip install -U pip

# We use torch v2.7.0, torchvision v0.22.0 and transformers v2.51.1 in the development of Video-R2
# Please see requirements.txt and environment.yml for all requirements
pip install -r requirements.txt
```

## 2) Run

From the repository root:

```bash
cd demo
python gradio_demo.py --ckpt MBZUAI/Video-R2 --port 7860
```

Open the printed URL in your browser.

## 3) Notes

- If you hit OOM, reduce the number of sampled frames in the UI.
- If your browser cannot play the uploaded video, convert it to MP4.

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

[<img src="../images/IVAL_logo.png" width="242" height="121">](https://www.ival-mbzuai.com)
[<img src="../images/Oryx_logo.png" width="121" height="121">](https://github.com/mbzuai-oryx)
[<img src="../images/MBZUAI_logo.png" width="360" height="121">](https://mbzuai.ac.ae)
