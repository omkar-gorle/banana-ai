"""Grad-CAM explainability for BananaCNN.

Grad-CAM highlights which image regions influenced the model's prediction.
It is a *visualization tool*, not proof of the model's reasoning.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from banana_ai.ml.model import BananaCNN


class GradCAM:
    """Generate Grad-CAM heatmaps for BananaCNN predictions.

    Uses the final convolutional layer (the last Conv2d inside
    ``model.features``) by default.
    """

    def __init__(self, model: BananaCNN, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.model.eval()

        # Identify the target layer: last Conv2d in model.features
        if target_layer is None:
            for module in reversed(list(model.features.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
        if target_layer is None:
            raise ValueError("Could not find a Conv2d layer in model.features")

        self.target_layer = target_layer
        self._gradients = None
        self._activations = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Generate a Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed image tensor of shape (1, 3, H, W).
        class_idx : int or None
            Target class index. If None, uses the predicted class.

        Returns
        -------
        np.ndarray
            Heatmap of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backpropagate from the target class score
        target = output[0, class_idx]
        target.backward()

        # Global average pool the gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def cleanup(self):
        """Remove registered hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def generate_gradcam_overlay(
    image_path: str,
    model: BananaCNN,
    device: torch.device,
    image_size: int = 224,
    class_idx: int | None = None,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Generate a Grad-CAM overlay for a banana image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    model : BananaCNN
        Trained model.
    device : torch.device
        Device to run inference on.
    image_size : int
        Input image size.
    class_idx : int or None
        Target class. If None, uses the predicted class.
    alpha : float
        Blending factor for the overlay.

    Returns
    -------
    tuple of (overlay, heatmap, predicted_class_idx, confidence)
    """
    import cv2

    # Load and preprocess
    original = Image.open(image_path).convert("RGB")
    original_resized = original.resize((image_size, image_size))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    input_tensor = transform(original).unsqueeze(0).to(device)

    # Generate Grad-CAM
    grad_cam = GradCAM(model)
    try:
        heatmap = grad_cam.generate(input_tensor, class_idx=class_idx)
    finally:
        grad_cam.cleanup()

    # Get prediction info
    with torch.no_grad():
        probs = torch.softmax(model(input_tensor), dim=1)[0]
    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (image_size, image_size))

    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original
    original_np = np.array(original_resized)
    overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_np)

    return overlay, heatmap_resized, pred_idx, confidence


def save_gradcam_example(
    image_path: str,
    model_path: str,
    output_dir: str = "reports/explainability",
) -> dict:
    """Generate and save a Grad-CAM visualization example.

    Returns a dict with the results and file paths.
    """
    import cv2

    from banana_ai.ml.device import configure_cpu_threads, get_device
    from banana_ai.ml.model import CLASS_NAMES

    configure_cpu_threads()
    device = get_device()

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    model = BananaCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    overlay, heatmap, pred_idx, confidence = generate_gradcam_overlay(
        image_path=image_path,
        model=model,
        device=device,
        image_size=checkpoint.get("image_size", 224),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    overlay_path = output_path / "gradcam_overlay.png"
    heatmap_path = output_path / "gradcam_heatmap.png"

    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(heatmap_path),
        cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET),
    )

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "overlay_path": str(overlay_path),
        "heatmap_path": str(heatmap_path),
        "note": (
            "Grad-CAM is a visualization of influential regions, "
            "not proof of the model's reasoning."
        ),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanation.")
    parser.add_argument("--image", required=True, help="Path to banana image.")
    parser.add_argument(
        "--model", default="models/banana_cnn_best.pt", help="Model checkpoint."
    )
    parser.add_argument(
        "--output-dir", default="reports/explainability", help="Output directory."
    )
    args = parser.parse_args()

    result = save_gradcam_example(args.image, args.model, args.output_dir)
    print(json.dumps(result, indent=2))
