import torch

from banana_ai.ml.device import (
    configure_cpu_threads,
    get_device,
    get_device_name,
    get_device_report,
)
from banana_ai.ml.model import BananaCNN


def test_device_detection_does_not_crash():
    assert get_device().type in {"cpu", "cuda", "xpu"}


def test_cpu_fallback_is_available():
    assert torch.device("cpu").type == "cpu"


def test_forward_pass_on_selected_device():
    device = get_device()
    model = BananaCNN(num_classes=4).to(device)
    output = model(torch.randn(1, 3, 224, 224, device=device))
    assert output.shape == (1, 4)


def test_checkpoint_save_load_with_map_location(tmp_path):
    path = tmp_path / "checkpoint.pt"
    model = BananaCNN(num_classes=4)
    torch.save({"model_state_dict": model.state_dict()}, path)
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    restored = BananaCNN(num_classes=4)
    restored.load_state_dict(checkpoint["model_state_dict"])


def test_device_report_has_valid_information():
    report = get_device_report()
    assert report["selected_device"] in {"cpu", "cuda", "xpu"}
    assert isinstance(report["cuda_available"], bool)
    assert isinstance(report["xpu_available"], bool)
    assert report["torch_cpu_threads"] >= 1
    assert get_device_name()


def test_cpu_thread_configuration(monkeypatch):
    monkeypatch.setenv("BANANA_TORCH_THREADS", "4")
    assert configure_cpu_threads() == 4
