from collections import deque
from warnings import warn

from typing import Optional, Dict, Any
import fugu.simulators.SpikingNeuralNetwork as snn

from .backend import Backend, PortDataIterator
from ..utils.export_utils import results_df_from_dict
from ..utils.misc import CalculateSpikeTimes
from .snn_backend import snn_Backend
import numpy as np

class gsearch_Backend(snn_Backend):
	"""
		Backend extension implementing Loihi graph-search runtime operations.
	"""
	def compile(self, scaffold, compile_args):
		super().compile(scaffold, compile_args)
		# Optional debug flag
		import os
		self._gs_debug = bool(os.getenv('FUGU_GS_DEBUG'))
		
		# Store original backward edge weights for readout logic
		bundle = self.fugu_graph.graph.get('loihi_gs', {})
		if 'w_backward_original' not in bundle:
			w_backward_original = {}
			for u, v, data in self.fugu_graph.edges(data=True):
				if data.get('direction') == 'backward':
					# Edge (u, v) is a backward edge, store with key (u, v)
					w_backward_original[(u, v)] = data.get('weight', 0)
			bundle['w_backward_original'] = w_backward_original
			self.fugu_graph.graph['loihi_gs'] = bundle
		
		# Initialize spike_time tracking for all neurons (when they first spiked)
		self.spike_time = {name: -1 for name in self.fugu_graph.nodes()}
		self.current_timestep = 0
	
	def zero_backward_edges(self, edges_to_zero: list[tuple[str, str]]) -> int:
		"""Zero out backward edges in both the graph and the neural network.
		
		Args:
		    edges_to_zero: List of (post, pre) tuples representing backward edges post<-pre.
		                   The graph stores this as edge (post, pre) with direction='backward'.
		
		Returns:
		    Number of edges successfully zeroed.
		"""
		zeroed_count = 0
		for post, pre in edges_to_zero:
			# The graph stores backward edge post<-pre as (post, pre) with direction='backward'
			if self.fugu_graph.has_edge(post, pre):
				edge_data = self.fugu_graph[post][pre]
				if edge_data.get('direction') == 'backward' and edge_data.get('weight', 0) > 0:
					# Zero the weight in the graph
					edge_data['weight'] = 0.0
					
					# Also zero the synapse in the neural network simulator
					# The synapse in nn is from post to pre
					if hasattr(self, 'nn') and hasattr(self.nn, 'syns'):
						syn_key = (post, pre)
						if syn_key in self.nn.syns:
							self.nn.syns[syn_key].wght = 0.0
					
					zeroed_count += 1
		
		return zeroed_count
	
	def remaining_backward_edges(self) -> list[tuple[str, str]]:
		"""Return list of backward edges that still have non-zero weight.
		
		Returns:
		    List of (pre, post) tuples for backward edges with weight > 0.
		"""
		remaining = []
		for u, v, data in self.fugu_graph.edges(data=True):
			if data.get('direction') == 'backward' and data.get('weight', 0) > 0:
				remaining.append((u, v))
		return remaining
	

	
	def readout_next_hop(self):
		"""Return next hop by checking which BACKWARD edge was pruned.

		Algorithm from standalone implementation (Lines 14-20):
		For current node i, find node j such that:
		1. There's a forward edge i to j (direction='forward', weight=1)
		2. The backward edge j to i was PRUNED (originally had weight>0, now weight=0)

		This method inspects `self.current_hop` to determine current node.

		Returns:
		    The next neuron name (str) or None if no valid hop.
		"""
		current = getattr(self, 'current_hop', None)
		if current is None or current not in self.fugu_graph:
			return None
		
		# Get bundle to access original backward edge weights
		bundle = self.fugu_graph.graph.get('loihi_gs', {})
		w_backward_original = bundle.get('w_backward_original', {})
		
		# Look for forward edges from current (ignore forward weight; only direction matters)
		# Iterate deterministically by sorting on neighbor name
		neighbors = []
		for _, nxt, data in self.fugu_graph.out_edges(current, data=True):
			if data.get('direction') == 'forward':
				neighbors.append(nxt)
		for nxt in sorted(neighbors):
			# Check if backward edge nxt to current was pruned
			backward_key = (nxt, current)
			if backward_key in w_backward_original and w_backward_original[backward_key] > 0:
				if self.fugu_graph.has_edge(nxt, current):
					current_weight = self.fugu_graph[nxt][current].get('weight', 0)
					if current_weight == 0:
						return nxt

		return None

	def reconstruct_path(self) -> list[str]:
		"""
		Reconstruct a path by walking forward hops using pruned-backward-edge signals.

		This mirrors the original readout: starting at source, at each step choose the
		unique forward edge whose paired backward edge was pruned (weight==0 now).
		No BFS over the remaining graph is performed; we only follow next hops until
		reaching the destination or stalling. Returns [src..dst] or [] if unreachable.
		"""
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		src = bundle.get('source_neuron')
		dst = bundle.get('destination_neuron')
		if not src or not dst or src not in self.fugu_graph or dst not in self.fugu_graph:
			return []

		path: list[str] = [src]
		self.current_hop = src
		visited = set([src])
		while path[-1] != dst:
			if path[-1] == dst:
				return path
			nxt = self.readout_next_hop()
			if nxt is None:
				break
			path.append(nxt)
			if nxt in visited:
				# Detected a loop; abort
				raise RuntimeError("Cycle located in readout.")
			visited.add(nxt)
			self.current_hop = nxt

		# If we exited without reaching dst, report unreachable
		return path if path and path[-1] == dst else []



	def prune_step(self) -> dict[str, Any]:
		"""Perform one graph-search pruning step (Lines 7-12 of algorithm).

		For each neuron i that spiked at current timestep k:
		  For each fanin j with backward edge j to i (edge stored as (j,i) with direction='backward'):
		    If j spiked at timestep (k - d_{j,i}), prune edge j to i
		
		Returns diagnostics: {'zeroed': int, 'remaining': int, 'source_spiked': bool}.
		"""
		# Determine current timestep index from spike histories
		# After self.nn.step(), each neuron's spike_hist has been appended for this step.
		current_index = None
		if self.nn.nrns:
			# Take the length of any neuron's history as reference (they are advanced in lockstep)
			any_neuron = next(iter(self.nn.nrns.values()))
			current_index = len(any_neuron.spike_hist) - 1
		else:
			current_index = -1

		# Get neurons that spiked THIS timestep (regardless of whether it's their first spike)
		newly_spiked = [name for name, neuron in self.nn.nrns.items() if neuron.spike_hist and neuron.spike_hist[-1]]
		
		edges_to_prune: list[tuple[str, str]] = []
		
		# Algorithm 1 Lines 7-9: for all j fanin to i, if s_i[t+1]=1 and s_j[t-d_{j,i}]=1, prune j to i edge
		# In our naming: i = neuron that just spiked, j = fanin neuron
		# Backward edge j to i is stored in graph as (j, i) with direction='backward'
		for i in newly_spiked:
			# Get all incoming backward edges: (j, i) where j is fanin to i
			for j, _, edge_data in self.fugu_graph.in_edges(i, data=True):
				if edge_data.get('direction') != 'backward':
					continue
				
				delay = int(edge_data.get('delay', 1))
				j_neuron = self.nn.nrns.get(j)
				if not j_neuron:
					continue
				
				# Check if j spiked at (current_index - delay)
				check_index = current_index - delay
				did_spike = False
				if check_index >= 0 and check_index < len(j_neuron.spike_hist):
					did_spike = j_neuron.spike_hist[check_index]
					if did_spike:
						# Prune backward edge j-i
						edges_to_prune.append((j, i))
		
		
		zeroed = self.zero_backward_edges(edges_to_prune)
		# Only compute remaining count if debug mode is on (expensive for large graphs)
		remaining = len(self.remaining_backward_edges()) if self._gs_debug else -1
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		src = bundle.get('source_neuron')
		source_spiked = (src in newly_spiked) if src else False
		if self._gs_debug:
			print(f"[STEP] t={current_index}: zeroed={zeroed} remaining_backward={remaining} source_spiked={source_spiked}")
		return {'zeroed': zeroed, 'remaining': remaining, 'source_spiked': source_spiked}

	def run(self, n_steps: Optional[int] = None):
		"""Run the Loihi graph-search wavefront until source spikes or timeout.

		Algorithm (simplified):
		 1. Inject an initial spike at the destination neuron.
		 2. Iterate simulation steps:
		    - advance network one step (self.nn.step)
		    - prune backward edges whose post neuron spiked this step
		    - stop if source neuron has spiked
		 3. Reconstruct forward path (source -> destination) via remaining forward edges.

		Args:
		  n_steps: Optional cap on number of simulation iterations. If None, a heuristic
		           limit is chosen (10x number of neurons).

		Returns:
		  dict with keys:
		    'path'              : list[str] path of neuron names source->destination (may be empty)
		    'steps'             : int steps executed
		    'source_spiked'     : bool whether source spiked during run
		    'remaining_backward': int count of backward edges left after pruning
		"""
		bundle = self.fugu_graph.graph.get('loihi_gs') or {}
		dst = bundle.get('destination_neuron')
		src = bundle.get('source_neuron')
		if not src or not dst:
			return {'path': [], 'steps': 0, 'source_spiked': False, 'remaining_backward': len(self.remaining_backward_edges())}
		if dst not in self.nn.nrns or src not in self.nn.nrns:
			return {'path': [], 'steps': 0, 'source_spiked': False, 'remaining_backward': len(self.remaining_backward_edges())}

		# Destination already initialized with v=1.5 and v_prev=0.0 from brick
		# First step: v_prev=0.0 < 1.0, v=1.5 >= 1.0 - spike
		# After spike: v=1.5 (reset), v_prev=1.5
		# Future steps: v_prev=1.5 >= 1.0 - no more spikes
		self.spike_time[dst] = 0  # Will spike at algorithm time 0

		limit = int(n_steps) if n_steps is not None else max(10, 10 * len(self.fugu_graph.nodes))
		self.current_timestep = 0  # simulator time
		self.algo_offset = 0       # algorithm time corresponds to first spike times directly
		source_spiked = False

		# Perform initial propagation step (destination will spike naturally)
		self.current_timestep += 1
		self.nn.step()
		# Initial propagation step (destination should have spiked)
		last_diag = self.prune_step()
		source_spiked = last_diag.get('source_spiked', False)

		# Main loop
		while not source_spiked and self.current_timestep < limit:
			self.current_timestep += 1
			self.nn.step()
			last_diag = self.prune_step()
			source_spiked = last_diag.get('source_spiked', False)
		
		if source_spiked and self.current_timestep < limit:
			self.current_timestep += 1
			self.nn.step()
			self.prune_step()


		path = self.reconstruct_path() if source_spiked else []
		# Compute cost: sum of backward edge delays (post,pre) equals original edge costs
		if path:
			backward_sum = 0
			for i in range(len(path) - 1):
				pre = path[i]
				post = path[i+1]
				if self.fugu_graph.has_edge(post, pre):
					backward_sum += int(self.fugu_graph[post][pre].get('delay', 0))
			cost = backward_sum
		else:
			cost = float('inf')
		
		return {
			'path': path,
			'steps': self.current_timestep,
			'source_spiked': source_spiked,
			'remaining_backward': len(self.remaining_backward_edges()) if self._gs_debug else -1,
			'cost': cost,
		}