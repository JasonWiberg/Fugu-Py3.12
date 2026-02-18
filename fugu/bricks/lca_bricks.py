from fugu.bricks import Brick
from fugu.scaffold.port import ChannelSpec, PortSpec, PortUtil
import numpy as np

class LCABrick(Brick):
    def __init__(
        self,
        Phi,
        input_signal=None,
        dt=1e-3,
        tau_syn=1.0,
        threshold=1.0,
        spike_prob=1.0,
        leakage_constant=1.0,
        lam=0.1,
        reset_voltage=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.Phi = np.array(Phi, dtype=float)
        self.input_signal = None if input_signal is None else np.array(input_signal, dtype=float).reshape(-1)
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        self.tau_syn = float(tau_syn)
        if self.tau_syn <= 0:
            raise ValueError("tau_syn must be positive.")
        self.threshold = float(threshold)
        self._lam_bias = float(lam)
        self.spike_prob = float(spike_prob)
        if self.spike_prob < 0 or self.spike_prob > 1:
            raise ValueError("spike_prob must be within [0, 1].")
        self.leakage_constant = float(leakage_constant)
        if self.leakage_constant < 0 or self.leakage_constant > 1:
            raise ValueError("leakage_constant must be within [0, 1].")
        self.reset_voltage = float(reset_voltage)
        self.bias_vector = None
        self.supported_codings = ['Raster', 'Undefined']

    @classmethod
    def input_ports(cls) -> dict[str, PortSpec]:
        """LCABrick doesn't require inputs - it generates sparse codes."""
        return {}

    @classmethod  
    def output_ports(cls) -> dict[str, PortSpec]:
        """LCABrick outputs sparse activation codes."""
        port = PortSpec(name='output')
        port.channels['data'] = ChannelSpec(name='data', coding=['Raster'])
        port.channels['complete'] = ChannelSpec(name='complete')
        return {port.name: port}

    def normalize_columns(self, A):
        """ Normalize columns of A to unit norm. """
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        return A / norms
                

    def build2(self, graph, inputs: dict = {}):
        Phi = self.normalize_columns(self.Phi)
        N = Phi.shape[1]
        M = Phi.shape[0]

        if self.input_signal is None:
            raise ValueError("input_signal must be provided to compute LCA biases.")

        input_vector = np.array(self.input_signal, dtype=float).reshape(-1)
        if input_vector.size != M:
            raise ValueError(
                f"input_signal length {input_vector.size} does not match dictionary height {M}."
            )

        feedforward_drive = Phi.T @ input_vector
        # Scale feedforward drive once by dt and fold lambda into bias
        scaled_bias = self.dt * (feedforward_drive - self._lam_bias)
        self.bias_vector = scaled_bias


        # Create complete control node
        complete_node_name = self.generate_neuron_name('complete')
        graph.add_node(
            complete_node_name,
            index=-1,
            threshold=0.0,
            decay=0.0,
            p=1.0,
            potential=0.0,
        )
        comp = {"name":"RecurrentInhibition", 
                "tau_syn": 1,
                "dt":self.dt}

        neuron_names = []
        for i in range(N):
            neuron_name = self.generate_neuron_name(f"neuron_{i}")
            neuron_names.append(neuron_name)
            graph.add_node(
                neuron_name,
                index=i,
                threshold=self.threshold,
                reset_voltage=self.reset_voltage,
                leakage_constant=self.leakage_constant,
                potential=0.0,
                bias=float(scaled_bias[i]),
                p=self.spike_prob,
                neuron_type='GeneralNeuron',
                compartment=comp,
            )

        W = Phi.T @ Phi
        np.fill_diagonal(W, 0.0)

        scaled_weights = self.dt * W
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                weight = scaled_weights[i, j]
                if weight > 0:
                    graph.add_edge(neuron_names[i], neuron_names[j], weight=weight, delay=1)

        result = PortUtil.make_ports_from_specs(LCABrick.output_ports())
        output_port = result['output']

        data_channel = output_port.channels['data']
        data_channel.neurons = neuron_names

        complete_channel = output_port.channels['complete']
        complete_channel.neurons = [complete_node_name]

        return result
