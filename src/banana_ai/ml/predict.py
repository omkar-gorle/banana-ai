import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from banana_ai.ml.model import BananaCNN
from banana_ai.ml.device import get_device
from banana_ai.services.shelf_life import prototype_estimate


def load_model(model_path: str):
    device = get_device()

    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parents[3] / checkpoint_path

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = BananaCNN(num_classes=len(checkpoint["class_names"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((checkpoint.get("image_size", 224), checkpoint.get("image_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, transform, checkpoint["class_names"], device


def predict_image(image_path: str, model, transform, class_names, device):
    with Image.open(image_path).convert("RGB") as image:
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    index = int(probabilities.argmax().item())
    stage = class_names[index]
    confidence = float(probabilities[index].item())

    estimate = prototype_estimate(stage)

    probabilities_dict = {
        class_names[i]: float(probabilities[i].item())
        for i in range(len(class_names))
    }

    return {
        "stage": stage,
        "confidence": confidence,
        "probabilities": probabilities_dict,
        "estimated_days_left": estimate,
    }


def predict(image_path: str, model_path: str = "models/banana_cnn_v2.pt"):
    """Run inference using the production V2 checkpoint by default."""
    model, transform, class_names, device = load_model(model_path)
    return predict_image(image_path, model, transform, class_names, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--model",
        default="models/banana_cnn_v2.pt",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(args.image)

    result = predict(args.image, args.model)

    print(f"\nPrediction: {result['stage']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Prototype days-left estimate: {result['estimated_days_left']}")
    print("\nClass probabilities:")
    for name, probability in result["probabilities"].items():
        print(f"  {name:10s}: {probability:.2%}")
