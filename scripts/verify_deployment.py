"""Pre-demo verification: checkpoints + the generated faculty-demo notebook.

Run before the viva (or before handing the demo files over):

    .venv/bin/python scripts/verify_deployment.py [--checkpoint PATH]...

Checks:
  1. every checkpoint is self-describing (cfg + 37 breed names) and loads;
  2. the model rebuilds from the embedded cfg and the forward pass returns
     seg_logits + cls_logits at the expected shapes;
  3. predict() runs on CPU on a sample image and writes the overlay PNG;
  4. notebooks/faculty_demo.ipynb is standalone: no repo/dataset/training
     references, no forbidden imports/patterns.

Exit code is 0 only if everything passes.
"""

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

from scripts.predict import get_preprocessing, load_model, make_overlay, predict

NOTEBOOK_PATH = os.path.join(ROOT, "notebooks", "faculty_demo.ipynb")
PRESENTATION_NOTEBOOK_PATH = os.path.join(ROOT, "notebooks", "presentation_notebook.ipynb")

# Patterns that must NEVER appear in the faculty demo notebook (word-boundary
# checks so innocuous substrings like "digit" don't trip).
FORBIDDEN_PATTERNS = [
    r"\bsrc\b",
    r"\bTrainer\b",
    r"\bget_loaders\b",
    r"\bget_datasets\b",
    r"OxfordIIITPet",
    r"\bkaggle\b",
    r"subprocess",
    r"os\.system",
    r"\bgit\b",
    # shell escapes (!git, !pip ...) — a line starting with '!'; a '!' inside
    # '!=' (comparison) is fine.
    r"^\s*!",
    # shell magics (%cd, %pip ...) — a line starting with '%'; a trailing '%'
    # inside format strings like ":.2%" is fine, so only match at line start.
    r"^\s*%",
]

# Only these modules may be imported by the notebook code cells.
ALLOWED_IMPORTS = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchvision",
    "torchvision.transforms.functional",
    "PIL",
    "numpy",
    "matplotlib",
    "ipywidgets",
    "IPython",
    "pathlib",
    "os",
    "json",
    "glob",
    "torchvision",
    "pandas",
]

# Required content markers (must appear in the notebook).
REQUIRED_MARKERS = [
    "best.pth",
    "build_model",
    "IMG_SIZE",
    "THRESHOLD",
    "predicted breed",
    "No training",
    "Upload",
]

# Markers for the presentation notebook (superset: results/history/grids).
PRESENTATION_MARKERS = [
    "best.pth",
    "build_model",
    "3×3",
    "results_table",
    "history_plot",
    "predicted breed",
    "No training",
]


def verify_checkpoint(path):
    print(f"\n=== checkpoint: {path} ===")
    device = torch.device("cpu")
    model, classes, cfg, ckpt = load_model(path, device)
    assert "model" in cfg, "checkpoint cfg has no 'model' section"
    assert len(classes) == 37, f"expected 37 breeds, got {len(classes)}"
    print(f"  model: {cfg['model']['name']} | classes: {len(classes)}")
    img_size, threshold = get_preprocessing(cfg)
    print(f"  img_size: {img_size} | threshold: {threshold}")

    # forward pass at the checkpoint's own image size
    x = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        out = model(x)
    assert out["seg_logits"].shape == (1, 1, img_size, img_size)
    assert out["cls_logits"].shape == (1, 37)
    print(f"  forward ok: seg {tuple(out['seg_logits'].shape)} cls {tuple(out['cls_logits'].shape)}")

    # end-to-end predict on a tiny synthetic image
    tmp_img = os.path.join(ROOT, "reports", "_verify_input.png")
    tmp_out = os.path.join(ROOT, "reports", "_verify_overlay.png")
    Image.fromarray(np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)).save(tmp_img)
    mask, label, conf, _ = predict(model, tmp_img, device, img_size, threshold)
    assert mask.shape == (200, 300), f"mask not resized to original size: {mask.shape}"
    make_overlay(tmp_img, mask, tmp_out)
    assert os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0
    assert 0 <= label < len(classes)
    print(f"  predict ok: label={label} ({classes[label]}) conf={conf:.2%} "
          f"mask={tuple(mask.shape)} overlay written")
    print(f"  PASSED")
    return True


def verify_notebook(path):
    print(f"\n=== faculty demo notebook: {path} ===")
    with open(path) as f:
        nb = json.load(f)
    code_src = "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    )
    all_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])

    for pat in FORBIDDEN_PATTERNS:
        assert not re.search(pat, code_src), f"forbidden pattern in code: {pat!r}"
    print("  no forbidden patterns in code cells")

    for line in code_src.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            # the import target is the module path (drop ' as' alias and commas)
            target = s.split()[1].split(".")[0].rstrip(",")
            assert target in ALLOWED_IMPORTS, f"disallowed import: {s!r}"
    print("  all imports are from the allowed set")

    # the faculty demo needs the upload/THRESHOLD markers; the presentation
    # notebook is a superset (it has results/history/grids instead)
    markers = REQUIRED_MARKERS if os.path.basename(path) == "faculty_demo.ipynb" else PRESENTATION_MARKERS
    for marker in markers:
        assert marker.lower() in all_src.lower(), f"missing required marker: {marker!r}"
    print("  required markers present")

    # model classes must be inlined (not imported)
    assert "class UNet(nn.Module):" in code_src
    assert "class AttentionUNet(UNet):" in code_src
    assert "def build_model(cfg):" in code_src
    print("  model classes inlined (UNet / AttentionUNet / build_model)")
    print("  PASSED")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="checkpoint(s) to verify; defaults to all outputs/*/checkpoints/best.pth")
    ap.add_argument("--no-notebook", action="store_true",
                    help="skip notebook self-containment checks")
    args = ap.parse_args()

    ckpts = args.checkpoint or sorted(
        glob.glob(os.path.join(ROOT, "outputs", "*", "checkpoints", "best.pth"))
    )
    if not ckpts:
        print("no checkpoints found — pass --checkpoint PATH or train first")
        sys.exit(2)

    ok = all(verify_checkpoint(c) for c in ckpts)
    if not args.no_notebook:
        ok = verify_notebook(NOTEBOOK_PATH) and ok
        # the presentation notebook shares the same self-containment rules
        ok = verify_notebook(PRESENTATION_NOTEBOOK_PATH) and ok

    print("\n" + ("ALL DEPLOYMENT CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
