import numpy as np
import pytest

from fugu.simulators.SpikingNeuralNetwork.neuron import GeneralNeuron

# Ensure deterministic random outcomes for probabilistic spiking
np.random.seed(0)


def test_general_neuron_custom_lambda_blocks_spike():
    # Bias large enough to exceed threshold if default logic used
    n = GeneralNeuron(
        name="n",
        threshold=1.0,
        reset_voltage=0.0,
        leakage_constant=1.0,
        voltage=0.0,
        bias=10.0,
        p=1.0,
        record=False,
        compartment={"name": "RecurrentInhibition", "tau_syn": 1.0, "dt": 1.0},
        spike_thresh_lambda=lambda v: False,
    )

    n.update_state()
    assert n.spike is False, "Custom lambda should block spiking even above threshold"


def test_general_neuron_custom_lambda_allows_spike():
    n = GeneralNeuron(
        name="n2",
        threshold=1000.0,  # very high to ensure default would not spike
        reset_voltage=0.0,
        leakage_constant=1.0,
        voltage=0.0,
        bias=0.5,  # any value
        p=1.0,
        record=False,
        compartment={"name": "RecurrentInhibition", "tau_syn": 1.0, "dt": 1.0},
        spike_thresh_lambda=lambda v: True,  # force spike decision
    )

    n.update_state()
    assert n.spike is True, "Custom lambda should allow spiking regardless of threshold"
