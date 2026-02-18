import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
snn = pytest.importorskip("snntorch")

import numpy as np
from fugu.backends.snntorch_backend import build_fugu_network, group_torch_layers


num_inputs = 100
num_hidden = 1000
num_outputs = 100
num_steps = 25
beta = 0.95


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)


def test_group_torch_layers_extracts_two_blocks():
    model = Net()
    blocks = group_torch_layers(model)

    assert len(blocks) == 2
    assert blocks[0]["output_size"] == num_hidden
    assert blocks[1]["output_size"] == num_outputs
    assert blocks[0]["is_output"] is False
    assert blocks[1]["is_output"] is True


def test_build_fugu_network_returns_scaffold():
    model = Net()
    blocks = group_torch_layers(model)
    spikes = torch.zeros((num_inputs, num_steps), dtype=torch.int32).numpy()

    scaffold = build_fugu_network(blocks, spikes)

    assert scaffold is not None
    assert "Output" in scaffold.name_to_tag


def test_weight_and_bias_shapes_match_layer_sizes():
    model = Net()
    blocks = group_torch_layers(model)

    assert blocks[0]["weights"].shape == (num_hidden, num_inputs)
    assert blocks[1]["weights"].shape == (num_outputs, num_hidden)
    assert blocks[0]["biases"].shape == (num_hidden,)
    assert blocks[1]["biases"].shape == (num_outputs,)

def test_beta_values_are_correctly_extracted():
    model = Net()
    blocks = group_torch_layers(model)

    assert np.allclose(blocks[0]["beta"], beta)
    assert np.allclose(blocks[1]["beta"], beta)

def test_weight_values_are_correctly_extracted():
    model = Net()
    blocks = group_torch_layers(model)

    assert np.allclose(blocks[0]["weights"], model.fc1.weight.detach().cpu().numpy())
    assert np.allclose(blocks[1]["weights"], model.fc2.weight.detach().cpu().numpy())

