"""Gradio web UI for classifying images with a model trained by train.py.

Usage:
    python app.py

Run `python train.py` first so that custom_cnn.pth exists.
"""

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

CHECKPOINT_PATH = Path("custom_cnn.pth")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Same transforms used for validation during training.
VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def get_device() -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path):
    """Load the checkpoint dict, tolerating older torch versions without weights_only."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(path: Path, device: torch.device):
    """Rebuild ResNet-18 for the saved number of classes and load the trained weights."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path.resolve()}\n"
            "Run `python train.py` first to create custom_cnn.pth."
        )

    checkpoint = load_checkpoint(path)
    class_names = checkpoint["class_names"]

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    return model, class_names


DEVICE = get_device()
MODEL, CLASS_NAMES = load_model(CHECKPOINT_PATH, DEVICE)


def predict(image: Image.Image) -> dict[str, float]:
    """Return a mapping of every class name to its predicted probability."""
    if image is None:
        return {}

    image = image.convert("RGB")
    tensor = VAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1)[0]

    return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs.cpu())}


def main() -> None:
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload an image"),
        outputs=gr.Label(label="Prediction"),
        title="Image Classifier",
        description=f"Trained on {len(CLASS_NAMES)} classes: {', '.join(CLASS_NAMES)}",
    )
    demo.launch()


if __name__ == "__main__":
    main()
