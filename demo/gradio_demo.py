import argparse
import gradio as gr
import shutil, os, tempfile
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# -------------------------
# Parse command-line args
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help="Model checkpoint (e.g., MBZUAI/Video-R2)")
parser.add_argument("--port", type=int, default=7860, help="Port to run Gradio app on")
args = parser.parse_args()

# -------------------------
# Load Model & Processor
# -------------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.ckpt,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(args.ckpt)


# -------------------------
# Inference Function
# -------------------------
def run_inference(video_upload, video_path, subtitles_path, question, pre_text, post_text, nframes):
    # Prefer uploaded video if provided, else fallback to manual path
    if video_upload is not None and video_upload != "":
        video_input_path = video_upload
    elif video_path is not None and video_path.strip() != "":
        video_input_path = video_path.strip()
    else:
        return "❌ No video provided. Please upload a video or specify a path."

    # Build full question string dynamically
    full_question = f"{pre_text}\n{question}\n{post_text}"

    # Build video content (optionally include subtitles if provided)
    video_content = {
        "type": "video",
        "video": video_input_path,
        "nframes": int(nframes),
        "max_pixels": 200704,
    }
    if subtitles_path and subtitles_path.strip() != "":
        video_content["subtitles"] = subtitles_path.strip()

    # Messages
    messages = [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": full_question},
            ],
        }
    ]

    # Prepare prompt
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    # Pack inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# -------------------------
# Video Preview Function
# -------------------------
def preview_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return None
    tmp_dir = tempfile.gettempdir()
    dst = os.path.join(tmp_dir, os.path.basename(video_path))
    shutil.copy(video_path, dst)
    return dst


# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Video-R2 Video Inference Demo (Transformers)")
    gr.Markdown(f"**Loaded checkpoint:** `{args.ckpt}`")

    with gr.Row():
        with gr.Column():
            video_upload = gr.Video(label="Upload a video")  # Option 1: upload
            video_path = gr.Textbox(label="Or enter video path", placeholder="/path/to/video.mp4 or file:///path/to/video.mp4")  # Option 2: manual path
            subtitles_path = gr.Textbox(label="Subtitles path (optional)", placeholder="/path/to/subtitles.srt or .vtt")
            preview_btn = gr.Button("Preview Video (from path)")
            video_preview = gr.Video(label="Video Preview (Path)", interactive=False)

            question = gr.Textbox(label="Question", placeholder="Enter your question")
            pre_text = gr.Textbox(
                label="Pre-Text",
                value=""
            )
            post_text = gr.Textbox(label="Post-Text", 
                                   value="Please think about this question as if you were a human pondering deeply. Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions. It's encouraged to include self-reflection or verification in the reasoning process. Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. Ensure to include the thinking process inside <think> and </think> tags and the final answer inside <answer> and </answer> tags.")
            nframes = gr.Slider(1, 768, value=32, step=1, label="Number of Video Frames")
            run_btn = gr.Button("Run Inference")

        with gr.Column():
            output_box = gr.Textbox(label="Model Output", lines=12)

    # Wire up functions
    preview_btn.click(fn=preview_video, inputs=[video_path], outputs=[video_preview])

    run_btn.click(
        fn=run_inference,
        inputs=[video_upload, video_path, subtitles_path, question, pre_text, post_text, nframes],
        outputs=[output_box],
    )

demo.launch(server_port=args.port)
