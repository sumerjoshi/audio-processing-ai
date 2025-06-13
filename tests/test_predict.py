import torch
import pytest
from model.pretrained.dual_head_cnn14 import DualHeadCnn14
from predict import predict_one


@pytest.fixture()
def dummy_input():
    return torch.randn(1, 32000)


@pytest.fixture()
def model():
    model = DualHeadCnn14(16000, 1024, 320, 64, 50, 8000, 527, False)
    model.eval()
    return model


def test_predict_one_output_shape(model, dummy_input):
    _, tag_probs = predict_one(model, dummy_input)
    assert tag_probs.shape == (527,)


def test_predict_output_range(model, dummy_input):
    prob, _ = predict_one(model, dummy_input)
    assert prob >= 0 and prob <= 1
