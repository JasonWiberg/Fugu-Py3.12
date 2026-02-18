
from fugu import Scaffold
from fugu.bricks.input_bricks import Vector_Input
from fugu.bricks.dense_bricks import dense_layer_1d
import numpy as np
import snntorch as snn
import torch.nn as nn

def group_torch_layers(model):
    blocks = []
    pending = {}
    for m in model.children():

        if isinstance(m, nn.Linear):
            pending = {"weights": m.weight.detach().cpu().numpy(),
                       "biases": (m.bias.detach().cpu().numpy() if m.bias is not None else None),
                       "output_size": m.out_features}
        elif hasattr(snn, "Leaky") and isinstance(m, snn.Leaky):
            beta = float(m.beta)
            blocks.append({
                **pending,
                "beta": beta,                  # store beta, not "decay"
                "threshold": 1.0,              # or m.threshold if present
                "is_output": False
            })
            pending = {}
        # ignore other modules, or handle Flatten, Dropout, etc. as needed
    if blocks:
        blocks[-1]["is_output"] = True
    return blocks

def build_fugu_network(layer_dicts, input_data):
    sc = Scaffold()

    spikes = np.asarray(input_data, dtype=int)
    assert spikes.ndim == 2

    sc.add_brick(Vector_Input(spikes, coding="binary-L", name="Input", time_dimension=True))

    for i, layer in enumerate(layer_dicts):
        out_sz = int(layer["output_size"])
        W = np.asarray(layer["weights"], dtype=float)          # (out, in)
        b = layer["biases"]
        if b is None:
            b = np.zeros(out_sz, dtype=float)
        else:
            b = np.asarray(b, dtype=float).reshape(-1)
        T = layer["threshold"]
        T = np.full(out_sz, float(T), dtype=float) if np.isscalar(T) else np.asarray(T, dtype=float).reshape(-1)

        beta = float(layer["beta"])
        decay = 1.0 - beta                                  

        brick = dense_layer_1d(
            output_shape=(out_sz,),
            weights=W,
            thresholds=T,
            biases=b,
            decay=decay,
            name=("Output" if layer["is_output"] else f"Layer{i}"),
        )
        sc.add_brick(brick, input_nodes=[-1], output=layer["is_output"])

    sc.lay_bricks()
    return sc

def get_output_neuron_numbers(fugu_net):
    output_neuron_numbers = []
    for node, attrs in fugu_net.graph.nodes(data=True):
        if "Output" in node and "begin" not in node and "complete" not in node:
            output_neuron_numbers.append(attrs["neuron_number"])
    return output_neuron_numbers