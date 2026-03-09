#!/usr/bin/env python3
"""
FOR TESTING PURPOSES. Source of ground truth for the Fugu implementation and the accompanying test
harness, test_fugu_loihi.py

Direct implementation of the Loihi graph search algorithm from [1].

[1] M. Davies et al., "Advancing Neuromorphic Computing With Loihi: A Survey of Results and Outlook," 
in Proceedings of the IEEE, vol. 109, no. 5, pp. 911-934, May 2021, doi: 10.1109/JPROC.2021.3067593.

This implements the neuromorphic wavefront propagation algorithm that finds
shortest paths in weighted directed graphs using spiking neural network dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class LoihiGraphSearch:
    """
    
    The algorithm uses wavefront propagation to find shortest paths.
    Key features:
    - Parallel wavefront advancement using neuron dynamics
    - Backward edge pruning to trace shortest path
    - Requires fanout constraint: nodes with fanout > 1 must have all outgoing edges with cost=1
    """
    
    def __init__(self, adj_list: Dict[int, List[Tuple[int, int]]], source: int, destination: int):
        """
        Initialize Loihi graph search.
        
        Args:
            adj_list: Adjacency list {node: [(neighbor, cost), ...]}
            source: Source node
            destination: Destination node
        """
        self.adj_list = adj_list
        self.source = source
        self.destination = destination
        
        # Build node list
        self.nodes = list(adj_list.keys())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # Algorithm state (Line 1)
        self.v = np.zeros(self.n_nodes, dtype=np.float64)  # neuron potential
        self.s = np.zeros(self.n_nodes, dtype=np.int32)    # spike state
        self.spike_time = np.full(self.n_nodes, -1, dtype=np.int32)  # when each neuron first spiked
        
        # Synaptic weights and delays (Line 2)
        # w_forward[i][j] = forward weight i->j
        # w_backward[j][i] = backward weight j->i  
        # d_forward[i][j] = forward delay (always 0 in Loihi)
        # d_backward[j][i] = backward delay = cost - 1
        self.w_forward = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        self.w_backward = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float64)
        self.d_forward = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int32)
        self.d_backward = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int32)
        
        self._initialize_graph()
        
        # Don't initialize spike history here - it will be done in run()
        
    def _initialize_graph(self):
        """Initialize synaptic weights and delays (Algorithm 1, Line 2).
        
        NOTE: The paper uses delay d_{i,j} = c_{i,j} - 1 (can be 0).
        But simulation requires minimum delay=1, so we use d_{i,j} = c_{i,j}.
        This shifts all delays by +1 but preserves relative timing.
        """
        for node, neighbors in self.adj_list.items():
            i = self.node_to_idx[node]
            
            for neighbor, cost in neighbors:
                j = self.node_to_idx[neighbor]
                
                # Forward synapse i -> j: weight=1, delay=1 (minimum)
                self.w_forward[i][j] = 1.0
                self.d_forward[i][j] = 1
                
                # Backward synapse j -> i: weight=1, delay=cost (not cost-1!)
                # Matrix convention: w_backward[target][source]
                self.w_backward[i][j] = 1.0  # from j to i
                self.d_backward[i][j] = cost  # Use cost directly (min delay=1)
        
        self.w_backward_original = self.w_backward.copy()
    
    def advance_wavefront(self, t: int):
        """
        Execute one timestep of wavefront propagation (ADVANCEWAVEFRONT procedure).
        
        Args:
            t: Current timestep
        """
        # Store previous potential for threshold crossing check
        v_prev = self.v.copy()
        
        # Line 5: Update neuron potentials
        v_new = self.v.copy()
        
        for i in range(self.n_nodes):
            # Sum inputs from all fanin with delays
            input_current = 0.0
            
            for j in range(self.n_nodes):
                if self.w_backward[i][j] > 0:
                    delay = self.d_backward[i][j]
                    # Get spike from t - delay timesteps ago
                    if t >= delay:
                        time_idx = t - delay
                        if time_idx < len(self.spike_history):
                            s_delayed = self.spike_history[time_idx][j]
                            input_current += self.w_backward[i][j] * s_delayed
            
            v_new[i] = self.v[i] + input_current
        
        self.v = v_new
        
        # Line 6: Update spike states (spike when crossing threshold)
        # s_i[t+1] = 1 if v_i[t+1] >= 1 and v_i[t] < 1
        s_new = self.s.copy()
        newly_spiked = []  # Track which neurons spike THIS timestep
        for i in range(self.n_nodes):
            if self.v[i] >= 1.0 and v_prev[i] < 1.0:  # Threshold crossing
                s_new[i] = 1
                self.spike_time[i] = t  # Record when this neuron spiked
                newly_spiked.append(i)
        
        self.s = s_new
        
        # Lines 7-12: Prune backward edges (only for neurons that JUST spiked)
        for i in newly_spiked:
            # For all j in fanin to i
            for j in range(self.n_nodes):
                if self.w_backward[i][j] > 0:
                    delay = self.d_backward[i][j]
                    # Check if neuron j spiked EXACTLY at time t - delay
                    expected_spike_time = t - delay
                    if self.spike_time[j] == expected_spike_time:
                        # Prune backward edge j->i
                        # Backward edge j->i corresponds to forward edge i->j in original graph
                        # When we prune the backward path, we keep the forward path!
                        # We only zero the backward edge, not the forward edge
                        self.w_backward[i][j] = 0.0
        
        # Store spike history AFTER processing
        self.spike_history.append(self.s.copy())
    
    def read_out_next_hop(self, current: int) -> Optional[int]:
        """
        Read out next hop from current node (READOUTNEXTHOP function, Lines 14-20).
        
        The next hop is the node j such that:
        1. There's a forward edge i to j in the original graph
        2. The backward edge from j to i was PRUNED
        
        Matrix convention: w_backward[i][j] represents backward edge from j to i
        When we prune w_backward[i][j], we're marking that j contributed to i's spike
        During readout from i, we look for which j had its backward edge to i pruned
        
        Args:
            current: Current node in path
            
        Returns:
            Next node in path, or None if no valid hop
        """
        i = self.node_to_idx[current]
        
        # For all j with forward edge i to j
        for j in range(self.n_nodes):
            if self.w_forward[i][j] == 1.0:
                # Check if backward edge j to i (stored as w_backward[i][j]) was pruned
                if self.w_backward_original[i][j] > 0 and self.w_backward[i][j] == 0:
                    return self.nodes[j]
        
        return None
    
    def run(self, max_steps: int = 1000) -> Tuple[List[int], int, int]:
        """
        Run the complete Loihi graph search algorithm.
        
        Returns:
            (path, hop_count, steps): Shortest path, number of hops, timesteps taken
        """
        # Line 21: Trigger search by spiking destination
        dst_idx = self.node_to_idx[self.destination]
        self.s[dst_idx] = 1
        self.spike_time[dst_idx] = 0  # Destination spikes at t=0
        
        # Initialize spike history with t=0 state (destination spiking)
        self.spike_history = [self.s.copy()]
        
        t = 1  # Start from t=1 since t=0 is already in spike_history
        

        src_idx = self.node_to_idx[self.source]
        while self.s[src_idx] == 0 and t < max_steps:
            self.advance_wavefront(t)
            t += 1
        
        if self.s[src_idx] == 0:
            # Source never spiked - no path
            return [], 0, t
        

        self.advance_wavefront(t)
        t += 1
        

        path = [self.source]
        n = 0
        
        current = self.source
        while current != self.destination and n < self.n_nodes:
            next_hop = self.read_out_next_hop(current)
            if next_hop is None:
                # No path found
                break
            path.append(next_hop)
            current = next_hop
            n += 1
        
        return path, n, t


def preprocess_fanout_constraint(adj_list: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Preprocess graph to satisfy Loihi fanout constraint while preserving relative path costs.
    
    Nodes with fanout > 1 must have all outgoing edges with cost=1.
    Split higher-cost edges using auxiliary nodes.
    
    To preserve relative path costs, we add +1 to ALL edges:
    - Split edges: become 1 + c (total cost c+1)
    - Non-split edges: become c+1
    This ensures preprocessing doesn't change the relative ordering of path costs.
    
    Args:
        adj_list: Input adjacency list
        
    Returns:
        Transformed adjacency list satisfying fanout constraint
    """
    # Find max node ID to generate unique auxiliary node IDs
    max_node = max(max(adj_list.keys()), max(dst for neighbors in adj_list.values() for dst, _ in neighbors))
    next_aux_id = max_node + 1
    
    new_adj = {}
    
    for node, neighbors in adj_list.items():
        if len(neighbors) <= 1:
            # No fanout constraint violation, but add +1 to preserve relative costs
            new_adj[node] = [(dest, cost + 1) for dest, cost in neighbors]
        else:
            # Fanout > 1: ensure all costs = 1 (constraint requirement)
            new_neighbors = []
            for dest, cost in neighbors:
                if cost == 1:
                    # Don't split, but add +1 to preserve relative costs
                    new_neighbors.append((dest, 2))
                else:
                    # Create auxiliary node
                    aux_id = next_aux_id
                    next_aux_id += 1
                    
                    # First edge: node -> aux (cost=1)
                    new_neighbors.append((aux_id, 1))
                    
                    # Second edge: aux -> dest (cost=original)
                    # Total delay: 1 + cost = cost+1 (matches non-split edges)
                    new_adj[aux_id] = [(dest, cost)]
            
            new_adj[node] = new_neighbors
    
    # Add any destination nodes that weren't in keys
    all_dests = {dest for neighbors in new_adj.values() for dest, _ in neighbors}
    for dest in all_dests:
        if dest not in new_adj:
            new_adj[dest] = []
    
    return new_adj


def dijkstra(adj_list: Dict[int, List[Tuple[int, int]]], source: int, destination: int) -> Tuple[List[int], int]:
    """
    Dijkstra's shortest path algorithm for comparison.
    
    Args:
        adj_list: Adjacency list {node: [(neighbor, cost), ...]}
        source: Source node
        destination: Destination node
        
    Returns:
        (path, total_cost): Shortest path and its total cost
    """
    import heapq
    
    # Initialize
    dist = {node: float('inf') for node in adj_list.keys()}
    dist[source] = 0
    parent = {node: None for node in adj_list.keys()}
    
    # Priority queue: (distance, node)
    pq = [(0, source)]
    visited = set()
    
    while pq:
        d, u = heapq.heappop(pq)
        
        if u in visited:
            continue
        
        visited.add(u)
        
        if u == destination:
            break
        
        for v, cost in adj_list.get(u, []):
            if v not in visited:
                new_dist = dist[u] + cost
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
                    heapq.heappush(pq, (new_dist, v))
    
    # Reconstruct path
    if dist[destination] == float('inf'):
        return [], float('inf')
    
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    
    return path, int(dist[destination])


if __name__ == "__main__":
    # Simple test
    adj = {
        0: [(1, 2), (2, 5)],
        1: [(3, 3)],
        2: [(3, 1)],
        3: []
    }
    
    # Preprocess for fanout
    adj_processed = preprocess_fanout_constraint(adj)
    
    print("Original adjacency list:", adj)
    print("Processed adjacency list:", adj_processed)
    
    # Run Loihi search
    loihi = LoihiGraphSearch(adj_processed, source=0, destination=3)
    path_loihi, hops, steps = loihi.run()
    
    print(f"\nLoihi result: path={path_loihi}, hops={hops}, steps={steps}")
    
    # Run Dijkstra on original
    path_dijk, cost_dijk = dijkstra(adj, source=0, destination=3)
    print(f"Dijkstra result: path={path_dijk}, cost={cost_dijk}")
