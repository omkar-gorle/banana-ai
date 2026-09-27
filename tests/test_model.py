import torch

from banana_ai.ml.model import BananaCNN


def test_model_output_shape():
    model = BananaCNN(num_classes=4)
    x = torch.randn(2, 3, 224, 224)

    output = model(x)

    assert output.shape == (2, 4)
