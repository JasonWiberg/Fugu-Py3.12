#!/usr/bin/env python3
"""Utilities for converting snnTorch fully connected nets into Fugu scaffolds."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

import torch.nn as nn

try:  # snnTorch is an optional dependency
	import snntorch as snn
except ImportError:  # pragma: no cover - soft dependency
	snn = None  # type: ignore

from fugu import Scaffold
from fugu.bricks.dense_bricks import dense_layer_1d
from fugu.bricks.input_bricks import Vector_Input

from .snn_backend import snn_Backend


def _require_snntorch():
	if snn is None:
		raise ImportError(
			"snntorch is required for snntorch_Backend. Install it via `pip install snntorch`."
		)


def group_torch_layers(model: nn.Module) -> List[Dict[str, Any]]:
	"""Extract linear + snnTorch leaky blocks from a sequential model."""

	_require_snntorch()

	blocks: List[Dict[str, Any]] = []
	pending: Dict[str, Any] = {}

	for module in model.children():
		if isinstance(module, nn.Linear):
			pending = {
				"weights": module.weight.detach().cpu().numpy(),
				"biases": (
					module.bias.detach().cpu().numpy() if module.bias is not None else None
				),
				"output_size": module.out_features,
			}
		elif hasattr(snn, "Leaky") and isinstance(module, snn.Leaky):
			if not pending:
				raise ValueError("Encountered snn.Leaky without a preceding Linear layer.")
			beta = float(module.beta)
			blocks.append(
				{
					**pending,
					"beta": beta,
					"threshold": 1.0,
					"is_output": False,
				}
			)
			pending = {}
		# Additional snnTorch module types (Flatten, Dropout, etc.) can be inserted here.

	if blocks:
		blocks[-1]["is_output"] = True

	return blocks


def build_fugu_network(
	layer_dicts: Sequence[Dict[str, Any]],
	input_data: np.ndarray,
	*,
	input_name: str = "Input",
	input_coding: str = "binary-L",
) -> Scaffold:
	"""Construct a Fugu scaffold mirroring the provided snnTorch layers."""

	sc = Scaffold()

	spikes = np.asarray(input_data, dtype=int)
	if spikes.ndim != 2:
		raise ValueError(
			"input_data must be a 2D array shaped (n_inputs, timesteps) for Vector_Input"
		)

	sc.add_brick(
		Vector_Input(spikes, coding=input_coding, name=input_name, time_dimension=True)
	)

	for idx, layer in enumerate(layer_dicts):
		out_sz = int(layer["output_size"])
		weights = np.asarray(layer["weights"], dtype=float)
		biases = layer.get("biases")
		if biases is None:
			biases = np.zeros(out_sz, dtype=float)
		else:
			biases = np.asarray(biases, dtype=float).reshape(-1)

		threshold = layer.get("threshold", 1.0)
		if np.isscalar(threshold):
			thresholds = np.full(out_sz, float(threshold), dtype=float)
		else:
			thresholds = np.asarray(threshold, dtype=float).reshape(-1)

		beta = float(layer["beta"])
		decay = 1.0 - beta

		brick = dense_layer_1d(
			output_shape=(out_sz,),
			weights=weights,
			thresholds=thresholds,
			biases=biases,
			decay=decay,
			name=("Output" if layer.get("is_output") else f"Layer{idx}"),
		)
		sc.add_brick(brick, input_nodes=[-1], output=bool(layer.get("is_output")))

	sc.lay_bricks()
	return sc


def get_output_neuron_numbers(fugu_net: Scaffold) -> List[int]:
	"""Return neuron numbers corresponding to the scaffold's Output brick."""

	output_neuron_numbers: List[int] = []
	for node, attrs in fugu_net.graph.nodes(data=True):
		if "Output" in node and "begin" not in node and "complete" not in node:
			output_neuron_numbers.append(attrs["neuron_number"])
	return output_neuron_numbers


class snntorch_Backend(snn_Backend):

	def __init__(
		self,
		*,
		input_name: str = "Input",
		input_coding: str = "binary-L",
	):
		"""Store trained snnTorch model; conversion happens during ``compile``.

		Args:
			model: Trained snnTorch network to mirror in Fugu.
			input_name: Name of the Fugu input brick.
			input_coding: Coding for the input spikes (passed to ``Vector_Input``).
		"""
		_require_snntorch()
		super().__init__()
		self.model: nn.Module = None
		self.input_name = input_name
		self.input_coding = input_coding
		self.scaffold: Scaffold = None
		self._input_tag: str = None
		self._num_inputs: int = None
		self._num_steps: int = None
		self._output_neuron_numbers: List[int] = None

	def group_torch_layers(self, model: nn.Module) -> List[Dict[str, Any]]:
		"""Thin instance wrapper around the module-level helper."""
		return group_torch_layers(model)

	def _build_scaffold_from_model(self, num_steps: int) -> Scaffold:
		"""Converts the stored snnTorch model into a Fugu scaffold."""

		if self.model is None:
			raise RuntimeError(
				"snntorch_Backend.compile must be called with a model before building the scaffold."
			)

		layer_dicts = self.group_torch_layers(self.model)
		if not layer_dicts:
			raise ValueError("Provided model does not contain any supported snnTorch layers.")

		first_layer_weights = np.asarray(layer_dicts[0]["weights"], dtype=float)
		self._num_inputs = int(first_layer_weights.shape[1])
		self._num_steps = int(num_steps)

		spikes_init = np.zeros((self._num_inputs, self._num_steps), dtype=int)

		scaffold = build_fugu_network(
			layer_dicts,
			spikes_init,
			input_name=self.input_name,
			input_coding=self.input_coding,
		)
		self.scaffold = scaffold
		self._input_tag = scaffold.name_to_tag[self.input_name]
		self._output_neuron_numbers = get_output_neuron_numbers(scaffold)
		return scaffold

	def compile(self, model: nn.Module, compile_args: Optional[Dict[str, Any]] = None):
		"""Compile a snnTorch ``model`` by first converting it to a Fugu scaffold.

		Args:
			model: Trained snnTorch network to mirror in Fugu.
			compile_args: Backend compile arguments. Must include ``num_steps`` for
				the temporal dimension used when building the Fugu scaffold.
		"""

		self.model = model
		compile_args = dict(compile_args or {})
		num_steps = int(compile_args.pop("num_steps", 1))
		scaffold = self._build_scaffold_from_model(num_steps)
		super().compile(scaffold, compile_args)

	def reset(self): 
		"""Reset network state between runs, as done explicitly in the notebook.

		The underlying ``snn_Backend.reset`` rebuilds the neural network from the
		stored Fugu circuit, ensuring each call to ``run`` starts from a clean state.
		"""
		super().reset()

	def run(
		self,
		spikes: np.ndarray,
		n_steps: int,
		return_potentials: bool = False,
	): 
		"""Run the compiled network for ``n_steps`` using externally provided spikes.

		The user is responsible for all preprocessing (e.g., frame-to-spike
		conversion) and must supply a spike matrix compatible with the input brick.
		"""

		if self.scaffold is None or self._input_tag is None:
			raise RuntimeError("snntorch_Backend.compile must be called before run().")

		self.set_properties(
			{
				self._input_tag: {
					"spike_vector": np.asarray(spikes, dtype=int),
					"time_dimension": True,
				}
			}
		)
		# Match the explicit reset-before-run pattern from the notebook.
		self.reset()
		return super().run(n_steps=n_steps, return_potentials=return_potentials)

	def get_output_neuron_numbers(self) -> List[int]:
		"""Return cached output neuron numbers after compilation."""
		if self._output_neuron_numbers is None:
			if self.scaffold is None:
				raise RuntimeError("snntorch_Backend.compile must be called before querying outputs.")
			self._output_neuron_numbers = get_output_neuron_numbers(self.scaffold)
		return list(self._output_neuron_numbers)


__all__ = [
	"snntorch_Backend",
	"group_torch_layers",
	"build_fugu_network",
	"get_output_neuron_numbers",
]

