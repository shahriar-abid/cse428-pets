"""Gradio live demo for CSE428 pet segmentation + breed classification.

Loads saved checkpoints (self-describing: cfg + classes are embedded), builds
the model from them, and serves a polished UI with image upload -> mask overlay
+ top-3 breeds with confidence. Lets you switch between U-Net and
Attention U-Net live.

Run:
    python gradio_demo.py                      # local UI at http://127.0.0.1:7860
    python gradio_demo.py --share              # + free 72h public link
    python gradio_demo.py --ckpt path/to/best.pth   # single-model mode
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Reuse the exact model code the notebook uses (no duplication).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models import build_model  # noqa: E402

# Default checkpoint paths (used only when a model isn't picked by name).
DEFAULT_CKPTS = {
    "attention_unet": "data/attention_unet/checkpoints/best.pth",
    "unet": "data/unet/checkpoints/best.pth",
}

MODELS = {}  # name -> (model, img_size, classes, threshold)


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    img_size = int(ck["cfg"].get("data", {}).get("img_size", 256))
    classes = list(ck.get("classes") or [f"class_{i}" for i in range(37)])
    threshold = float(ck["cfg"].get("model", {}).get("seg_threshold", 0.5))
    name = ck["cfg"]["model"]["name"]
    return model, img_size, classes, threshold, name


def get_model(name: str, device):
    """Load (once) and cache the requested model."""
    key = _model_key(name)
    if key in MODELS:
        return MODELS[key]
    ckpt = DEFAULT_CKPTS.get(key)
    if ckpt is None or not Path(ckpt).is_file():
        raise gr.Error(f"checkpoint not found: {ckpt}")
    MODELS[key] = load_model(ckpt, device)
    print(f"loaded {key} from {ckpt}")
    return MODELS[key]


def _model_key(name: str) -> str:
    """Map UI labels / paths to the canonical checkpoint key."""
    label_map = {"Attention U-Net": "attention_unet", "U-Net": "unet"}
    return label_map.get(name, name)  # a path or raw key passes through


@torch.no_grad()
def predict(img, model_name: str):
    if img is None:
        raise gr.Error("Please upload an image first.")
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                else "cpu"))
    model, img_size, classes, threshold, _ = get_model(model_name, device)

    img = img.convert("RGB")
    x = TF.resize(img, [img_size, img_size], antialias=True)
    x = TF.to_tensor(x).unsqueeze(0).to(device)
    out = model(x)

    # segmentation mask (resized back to the original image size)
    mask = (torch.sigmoid(out["seg_logits"]) > threshold)[0, 0].cpu().numpy()
    mask = TF.resize(torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0),
                     [img.size[1], img.size[0]],
                     interpolation=TF.InterpolationMode.NEAREST)[0, 0].numpy() > 0.5

    # classification: top-3 breeds
    probs = torch.softmax(out["cls_logits"], dim=1)[0]
    topk = torch.topk(probs, k=3)
    top3 = {classes[int(i)]: float(p) for p, i in zip(topk.values, topk.indices)}

    # overlay mask in red on the original image
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.copy()
    arr[mask > 0] = 0.5 * arr[mask > 0] + 0.5 * np.array([1.0, 0.0, 0.0])

    top_breed = classes[int(topk.indices[0])]
    return np.clip(arr, 0, 1), top3, f"{top_breed} ({topk.values[0].item():.0%})"


CSS = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
#demo-header { text-align: center; margin-bottom: 4px; }
#demo-header h1 { font-size: 1.7rem; margin-bottom: 2px; }
#demo-header p { color: #6b7280; margin-top: 0; }
.badge { display:inline-block; background:#f3f4f6; border:1px solid #e5e7eb;
         border-radius:999px; padding:2px 12px; font-size:0.8rem; color:#374151; }
footer { display:none !important; }
"""


def main():
    ap = argparse.ArgumentParser(description="Gradio demo - pet segmentation + breed classification")
    ap.add_argument("--ckpt", default=None,
                    help="Path to a saved best.pth (overrides the model dropdown)")
    ap.add_argument("--share", action="store_true", help="Create a free 72h public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                                else "cpu"))

    # Validate checkpoints up front so failures are loud and early.
    single_ckpt = None
    if args.ckpt:
        if not Path(args.ckpt).is_file():
            sys.exit(f"checkpoint not found: {args.ckpt}")
        single_ckpt = args.ckpt
        MODELS["single"] = load_model(single_ckpt, device)
        print(f"loaded {MODELS['single'][4]} from {single_ckpt}")
    else:
        available = [n for n, p in DEFAULT_CKPTS.items() if Path(p).is_file()]
        if not available:
            sys.exit("no checkpoints found under data/*/checkpoints/best.pth - "
                     "run the notebook in train mode first.")
        for n in available:
            get_model(n, device)  # preload + verify

    # A few example images straight from the dataset (if present).
    examples = []
    img_dir = Path("data/oxford-iiit-pet/images")
    if img_dir.is_dir():
        examples = sorted(img_dir.glob("*.jpg"))[:6]

    with gr.Blocks(title="CSE428 Pet Demo") as demo:
        gr.HTML(
            "<div id='demo-header'>"
            "<h1>🐾 Pet Segmentation &amp; Breed Classification</h1>"
            "<p>CSE428 · Oxford-IIIT Pet · U-Net &amp; Attention U-Net with a joint "
            "classifier head</p>"
            "<span class='badge'>no training — instant inference from a saved checkpoint</span>"
            "</div>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Upload a pet photo")
                if single_ckpt:
                    fixed_label = f"Model: {MODELS['single'][4]}"
                    gr.Markdown(f"**{fixed_label}**")
                    model_sel = gr.State("single")
                else:
                    model_sel = gr.Radio(
                        ["Attention U-Net", "U-Net"],
                        value="Attention U-Net",
                        label="Model",
                    )
                btn = gr.Button("Segment & Classify", variant="primary")
                if examples:
                    gr.Examples(
                        examples=[str(p) for p in examples],
                        inputs=inp,
                        label="Try an example",
                        cache_examples=False,
                    )
            with gr.Column(scale=1):
                out_overlay = gr.Image(type="numpy", label="Segmentation overlay")
                out_top = gr.Label(num_top_classes=3, label="Top-3 predicted breeds")
                out_headline = gr.Markdown("_Upload an image to see the predicted breed._")

        gr.HTML(
            "<div style='text-align:center; margin-top:18px; padding-top:12px; "
            "border-top:1px solid #e5e7eb; color:#6b7280; font-size:0.85rem; line-height:1.7;'>"
            "<b style='color:#374151;'>Group Members</b><br>"
            "Md Al Shahriar Abid (23301613) &nbsp;·&nbsp; Tasnuva Rahman (23301505)<br>"
            "CSE428 · Section 04 · BRAC University"
            "</div>"
        )

        btn.click(
            predict,
            inputs=[inp, model_sel],
            outputs=[out_overlay, out_top, out_headline],
        )
        inp.change(
            predict,
            inputs=[inp, model_sel],
            outputs=[out_overlay, out_top, out_headline],
        )
        if not single_ckpt:
            model_sel.change(
                predict,
                inputs=[inp, model_sel],
                outputs=[out_overlay, out_top, out_headline],
            )

    print(f"ready | device {device} | models: {list(MODELS)}")
    demo.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
