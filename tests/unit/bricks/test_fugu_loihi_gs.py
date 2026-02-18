#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Fugu LoihiGSBrick implementation.
"""

import unittest
import os
import time
from typing import Dict, List, Tuple

from fugu import Scaffold
from fugu.bricks import LoihiGSBrick
from fugu.backends import gsearch_Backend
from tests.unit.bricks.loihi_graph_search import LoihiGraphSearch, preprocess_fanout_constraint
import networkx as nx
import pytest


def dijkstra(adj: Dict[int, List[Tuple[int, int]]], source: int, destination: int) -> Tuple[List[int], int]:
    """
    Classic Dijkstra's algorithm for shortest path.
    
    Args:
        adj: Adjacency list {node: [(neighbor, cost), ...]}
        source: Starting node
        destination: Target node
    
    Returns:
        (path, cost): Shortest path and its total cost
    """
    import heapq
    
    nodes = set(adj.keys())
    for neighbors in adj.values():
        for neighbor, _ in neighbors:
            nodes.add(neighbor)
    
    dist = {node: float('inf') for node in nodes}
    dist[source] = 0
    parent = {node: None for node in nodes}
    pq = [(0, source)]
    visited = set()
    
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        
        if u == destination:
            break
        
        for v, cost in adj.get(u, []):
            if dist[u] + cost < dist[v]:
                dist[v] = dist[u] + cost
                parent[v] = u
                heapq.heappush(pq, (dist[v], v))
    
    # Reconstruct path
    if dist[destination] == float('inf'):
        return [], float('inf')
    
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    
    return path, dist[destination]


class TestFuguLoihiGraphSearch(unittest.TestCase):
    """Test Fugu LoihiGSBrick implementation against Dijkstra ground truth."""
    
    def run_fugu_brick(self, adj: Dict[int, List[Tuple[int, int]]], 
                       source: int, dest: int, 
                       test_name: str = "Test") -> Tuple[List[int], int, int]:
        """
        Run Fugu brick implementation and return path, cost, and steps.
        
        Args:
            adj: Original adjacency list with edge costs
            source: Source node
            dest: Destination node
            test_name: Name for brick identification
            
        Returns:
            (path, cost, steps): Path as node list, total cost, simulation steps
        """
        # Create Fugu scaffold and brick
        scaffold = Scaffold()
        brick = LoihiGSBrick(adj, name=f"LoihiGS_{test_name.replace(' ', '_')}", 
                            source=source, destination=dest)
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()
        
        # Compile and run
        backend = gsearch_Backend()
        backend.compile(scaffold, {})

        max_steps = 5000
        result = backend.run(n_steps=max_steps)
        
        # Extract neuron path produced by backend 
        path_neurons = result['path']
        backend_cost = result.get('cost', float('inf'))

        if not path_neurons:
            # Return immediately; caller will assert failure rather than cheat with Dijkstra
            return [], backend_cost, result['steps']

        # Build inverse map
        neuron_to_node = {v: k for k, v in brick.node_to_neuron.items()}
        mapped = []
        for n in path_neurons:
            if n in neuron_to_node:
                mapped.append(neuron_to_node[n])

        # Remove auxiliary nodes but preserve ordering of original nodes
        def is_aux(x):
            return isinstance(x, str) and '__aux__' in x
        cleaned = []
        for node in mapped:
            if is_aux(node):
                continue
            # Normalize numeric strings
            if isinstance(node, str) and node.isdigit():
                node = int(node)
            cleaned.append(node)

        # Collapse consecutive duplicates (could arise from parent pointer jumps over aux chains)
        collapsed = []
        for node in cleaned:
            if not collapsed or collapsed[-1] != node:
                collapsed.append(node)

        # Keep only nodes present in original adjacency keys (guards against stray labels)
        path_original = [n for n in collapsed if n in adj]

        # Optionally recompute cost for cross-check when path length >1
        if len(path_original) > 1:
            recomputed = 0
            for u, v in zip(path_original[:-1], path_original[1:]):
                ec = next((c for dst2, c in adj.get(u, []) if dst2 == v), None)
                if ec is None:
                    # Path contains an edge that doesn't exist in original graph
                    raise AssertionError(
                        f"Backend returned invalid path {path_original}: "
                        f"edge ({u}, {v}) does not exist in original adjacency list"
                    )
                recomputed += ec
            # Verify backend cost matches recomputed cost from path edges
            if backend_cost != recomputed:
                raise AssertionError(
                    f"Backend cost {backend_cost} != recomputed {recomputed} for path {path_original}"
                )
        elif len(path_original) == 1 and source == dest and backend_cost != 0:
            raise AssertionError(
                f"Degenerate source==dest path cost mismatch: backend={backend_cost}, expected=0"
            )

        return path_original, backend_cost, result['steps']
    
    def compare_with_dijkstra(self, adj: Dict[int, List[Tuple[int, int]]], 
                             source: int, dest: int, 
                             test_name: str):
        """
        Compare Fugu implementation against Dijkstra.
        
        Args:
            adj: Adjacency list
            source: Source node
            dest: Destination node
            test_name: Test identifier
        """
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        print(f"Graph: {len(set(adj.keys()))} nodes, source={source}, dest={dest}")
        
        # Run Dijkstra
        path_dijk, cost_dijk = dijkstra(adj, source, dest)
        print(f"Dijkstra: path={path_dijk}, cost={cost_dijk}")
        
        # Run Fugu brick
        path_fugu, cost_fugu, steps_fugu = self.run_fugu_brick(adj, source, dest, test_name)
        print(f"Fugu: path={path_fugu}, cost={cost_fugu}, steps={steps_fugu}")

        # Run direct LoihiGraphSearch (preprocess for fanout constraint)
        loihi_path_proc = []
        loihi_cost = float('inf')
        loihi_steps = -1
        do_loihi_compare = len(adj) <= 30  # skip very large stress tests for runtime
        if do_loihi_compare:
            adj_proc = preprocess_fanout_constraint(adj)
            lg = LoihiGraphSearch(adj_proc, source=source, destination=dest)
            path_loihi_raw, _, steps = lg.run(max_steps=5000)
            loihi_steps = steps
            if path_loihi_raw:
                # Collapse auxiliary nodes (IDs > max original id)
                max_orig = max(adj.keys())
                loihi_path_proc = [n for n in path_loihi_raw if n <= max_orig]
                # Compute cost over original adjacency
                if len(loihi_path_proc) > 1:
                    cost_tmp = 0
                    for u, v in zip(loihi_path_proc[:-1], loihi_path_proc[1:]):
                        if u == v:
                            continue
                        ec = next((c for dst, c in adj.get(u, []) if dst == v), None)
                        if ec is None:
                            cost_tmp = float('inf')
                            break
                        cost_tmp += ec
                    loihi_cost = cost_tmp
                elif len(loihi_path_proc) == 1 and source == dest:
                    loihi_cost = 0
            print(f"LoihiDirect: path={loihi_path_proc}, cost={loihi_cost}, steps={loihi_steps}")
        else:
            print("LoihiDirect: skipped (graph size > 30)")
        
        # Validate
        # Require backend to produce a non-empty path (no Dijkstra fallback substitution)
        self.assertTrue(path_fugu, "Backend produced empty path; previously masked by Dijkstra fallback")
        self.assertEqual(path_fugu[0], source, "Fugu path must start at source")
        self.assertEqual(path_fugu[-1], dest, "Fugu path must end at destination")
        self.assertEqual(cost_fugu, cost_dijk, 
                        f"Costs differ: Fugu={cost_fugu}, Dijkstra={cost_dijk}")
        if do_loihi_compare and loihi_path_proc:
            self.assertEqual(cost_fugu, loihi_cost,
                             f"Costs differ vs direct Loihi: Fugu={cost_fugu}, LoihiDirect={loihi_cost}")
        
        print(f"PASS: Fugu matches Dijkstra, cost={cost_dijk}")
    
    # ========== Test Cases ==========
    
    def test_simple_chain(self):
        """Simple linear chain: 0->1->2->3."""
        adj = {0: [(1, 3)], 1: [(2, 4)], 2: [(3, 3)], 3: []}
        self.compare_with_dijkstra(adj, 0, 3, "Simple Chain")
    
    def test_binary_choice(self):
        """Two paths: cheap long vs expensive short."""
        adj = {0: [(1, 1), (2, 10)], 1: [(2, 1)], 2: []}
        self.compare_with_dijkstra(adj, 0, 2, "Binary Choice")
    
    def test_diamond_equal_cost(self):
        """Diamond with equal cost paths."""
        adj = {0: [(1, 5), (2, 5)], 1: [(3, 5)], 2: [(3, 5)], 3: []}
        self.compare_with_dijkstra(adj, 0, 3, "Diamond Equal Cost")
    
    def test_high_fanout(self):
        """Single node with high fanout."""
        adj = {0: [(i, i) for i in range(1, 11)], **{i: [] for i in range(1, 11)}}
        self.compare_with_dijkstra(adj, 0, 5, "High Fanout")
    
    def test_unit_costs(self):
        """All edges have cost=1."""
        adj = {0: [(1, 1), (2, 1)], 1: [(3, 1)], 2: [(3, 1)], 3: []}
        self.compare_with_dijkstra(adj, 0, 3, "Unit Costs")
    
    def test_varied_costs(self):
        """Wide range of edge costs."""
        adj = {0: [(1, 64), (2, 1)], 1: [(3, 1)], 2: [(3, 63)], 3: []}
        self.compare_with_dijkstra(adj, 0, 3, "Varied Costs")
    
    def test_long_chain(self):
        """Long chain of 20 nodes."""
        n = 20
        adj = {i: [(i+1, i+2)] for i in range(n)}
        adj[n] = []
        self.compare_with_dijkstra(adj, 0, n, "Long Chain")
    
    def test_complete_small(self):
        """Complete graph K5."""
        adj = {i: [(j, abs(i-j)+1) for j in range(5) if j != i] for i in range(5)}
        self.compare_with_dijkstra(adj, 0, 4, "Complete K5")
    
    def test_multiple_fanout_nodes(self):
        """Multiple nodes with fanout > 1."""
        adj = {
            0: [(1, 3), (2, 5)],
            1: [(3, 2), (4, 4)],
            2: [(4, 2), (5, 3)],
            3: [(6, 1)], 4: [(6, 1)], 5: [(6, 1)], 6: []
        }
        self.compare_with_dijkstra(adj, 0, 6, "Multiple Fanout Nodes")
    
    def test_grid_like(self):
        """3x3 grid structure (move right or down)."""
        adj = {
            0: [(1, 2), (3, 3)], 1: [(2, 2), (4, 3)], 2: [(5, 3)],
            3: [(4, 2), (6, 3)], 4: [(5, 2), (7, 3)], 5: [(8, 3)],
            6: [(7, 2)], 7: [(8, 2)], 8: []
        }
        self.compare_with_dijkstra(adj, 0, 8, "Grid Like")
    
    def test_indirect_better_than_direct(self):
        """Indirect path cheaper than direct edge."""
        adj = {0: [(1, 1), (3, 100)], 1: [(2, 1)], 2: [(3, 1)], 3: []}
        self.compare_with_dijkstra(adj, 0, 3, "Indirect Better")
    
    def test_cascading_fanout(self):
        """Multiple levels of fanout."""
        adj = {
            0: [(1, 2), (2, 3)],
            1: [(3, 1), (4, 2)],
            2: [(4, 1), (5, 2)],
            3: [(6, 1)], 4: [(6, 1)], 5: [(6, 1)], 6: []
        }
        self.compare_with_dijkstra(adj, 0, 6, "Cascading Fanout")
    
    def test_maximum_cost_edges(self):
        """Test with maximum allowed edge costs."""
        adj = {0: [(1, 127), (2, 1)], 1: [(3, 1)], 2: [(3, 126)], 3: []}
        self.compare_with_dijkstra(adj, 0, 3, "Maximum Cost Edges")
    
    def test_dense_graph(self):
        """Dense connectivity with many redundant paths."""
        adj = {
            0: [(1, 5), (2, 3), (3, 7)],
            1: [(2, 2), (3, 4), (4, 6)],
            2: [(3, 1), (4, 5)],
            3: [(4, 2)],
            4: []
        }
        self.compare_with_dijkstra(adj, 0, 4, "Dense Graph")
    
    def test_sparse_graph(self):
        """Minimal connectivity, unique path."""
        adj = {0: [(1, 10)], 1: [(2, 20)], 2: [(3, 30)], 3: [(4, 40)], 4: []}
        self.compare_with_dijkstra(adj, 0, 4, "Sparse Graph")
    
    def test_deep_nested_fanout(self):
        """Deep nesting with choices at each level."""
        adj = {
            0: [(1, 1), (2, 2)],
            1: [(3, 1), (4, 2)],
            2: [(4, 1), (5, 2)],
            3: [(6, 1)], 4: [(6, 1)], 5: [(6, 1)],
            6: [(7, 1), (8, 2)],
            7: [(9, 1)], 8: [(9, 1)], 9: []
        }
        self.compare_with_dijkstra(adj, 0, 9, "Deep Nested Fanout")
    
    def test_many_equal_cost_paths(self):
        """Multiple paths with identical total cost."""
        adj = {
            0: [(1, 5), (2, 5)],
            1: [(3, 10), (4, 10)],
            2: [(3, 10), (4, 10)],
            3: [(5, 5)], 4: [(5, 5)], 5: []
        }
        self.compare_with_dijkstra(adj, 0, 5, "Many Equal Cost Paths")
    
    def test_alternating_high_low_costs(self):
        """Alternating expensive and cheap edges."""
        adj = {
            0: [(1, 1), (2, 100)],
            1: [(3, 100)],
            2: [(3, 1), (4, 100)],
            3: [(4, 1)],
            4: []
        }
        self.compare_with_dijkstra(adj, 0, 4, "Alternating Costs")
    
    def test_power_of_two_costs(self):
        """Edge costs as powers of 2."""
        adj = {
            0: [(1, 1), (2, 2)],
            1: [(3, 4)],
            2: [(3, 8), (4, 16)],
            3: [(4, 32)],
            4: []
        }
        self.compare_with_dijkstra(adj, 0, 4, "Power of Two Costs")
    
    def test_layered_complete_graph(self):
        """Complete bipartite layers."""
        adj = {
            0: [(1, 3), (2, 5), (3, 7)],
            1: [(4, 2), (5, 4)],
            2: [(4, 1), (5, 3)],
            3: [(4, 6), (5, 2)],
            4: [], 5: []
        }
        self.compare_with_dijkstra(adj, 0, 4, "Layered Complete")
    
    def test_fibonacci_costs(self):
        """Fibonacci sequence as edge costs."""
        fib = [1, 1, 2, 3, 5, 8, 13]
        adj = {i: [(i+1, fib[i])] for i in range(len(fib)-1)}
        adj[len(fib)-1] = []
        self.compare_with_dijkstra(adj, 0, len(fib)-1, "Fibonacci Costs")
    
    def test_bottleneck_paths(self):
        """Paths that converge through bottleneck nodes."""
        adj = {
            0: [(1, 2), (2, 3), (3, 4)],
            1: [(4, 5)], 2: [(4, 5)], 3: [(4, 5)],
            4: [(5, 1)],
            5: []
        }
        self.compare_with_dijkstra(adj, 0, 5, "Bottleneck Paths")
    
    def test_wide_shallow_graph(self):
        """Wide graph with shallow depth."""
        adj = {0: [(i, i*2) for i in range(1, 16)]}
        for i in range(1, 16):
            adj[i] = [(16, 1)]
        adj[16] = []
        self.compare_with_dijkstra(adj, 0, 16, "Wide Shallow")
    
    def test_prime_number_costs(self):
        """Prime numbers as edge costs."""
        primes = [2, 3, 5, 7, 11, 13]
        adj = {i: [(i+1, primes[i])] for i in range(len(primes)-1)}
        adj[len(primes)-1] = []
        self.compare_with_dijkstra(adj, 0, len(primes)-1, "Prime Costs")
    
    def test_exponential_growth_costs(self):
        """Exponentially increasing costs."""
        adj = {
            0: [(1, 1), (2, 10)],
            1: [(3, 10), (4, 100)],
            2: [(3, 1), (4, 10)],
            3: [(5, 1)], 4: [(5, 1)],
            5: []
        }
        self.compare_with_dijkstra(adj, 0, 5, "Exponential Growth")
    
    def test_very_large_random_graph(self):
        """Large random graph (30 nodes)."""
        import random
        random.seed(42)
        n = 30
        adj = {i: [] for i in range(n+1)}
        for i in range(n):
            # Each node connects to 2-4 subsequent nodes
            remaining = n - i
            if remaining <= 0:
                continue
            low = 2 if remaining >= 2 else 1
            high = min(4, remaining)
            if low > high:
                continue
            num_edges = random.randint(low, high)
            targets = random.sample(range(i+1, n+1), num_edges)
            for t in targets:
                cost = random.randint(1, 20)
                adj[i].append((t, cost))
        self.compare_with_dijkstra(adj, 0, n, "Large Random Graph")
    
    def test_all_edges_cost_one_except_optimal(self):
        """Nearly all edges cost 1, but optimal path uses higher costs."""
        adj = {
            0: [(1, 1), (2, 5)],
            1: [(3, 1), (4, 1)],
            2: [(5, 1)],
            3: [(6, 1)], 4: [(6, 1)], 5: [(6, 1)],
            6: [(7, 1)],
            7: []
        }
        # Direct 0->1->3->6->7 = 4, but 0->2->5->6->7 = 8
        self.compare_with_dijkstra(adj, 0, 7, "All Edges Cost One")

    def test_huge_fully_connected_graph(self):
        """Stress test: Fully connected directed graph with 100 nodes and varied costs.

        Cost function chosen to avoid trivial direct shortest path dominance while
        keeping maximum edge cost moderate so simulation completes quickly.
        """
        n = 100
        def cost_fn(i, j):
            # Deterministic pseudo-random but bounded cost in [1,19]
            return 1 + ((i * 37 + j * 53) % 19)
        adj = {i: [(j, cost_fn(i, j)) for j in range(n) if j != i] for i in range(n)}
        # Source=0, Destination=n-1
        self.compare_with_dijkstra(adj, 0, n - 1, "Huge Fully Connected K100")

   
    def test_construction_preprocessing_mapping_basic(self):
        """Merged: test_loihi_gs_preprocessing_and_mapping basic auxiliary creation."""
        adj = {
            'A': [('B', 3), ('C', 1)],
            'B': [('C', 2)],
            'C': []
        }
        brick = LoihiGSBrick(adj, name='MergedBasic')
        G = nx.DiGraph()
        brick.build(G, None, None, None, None)
        neuron_A = brick.node_to_neuron['A']
        neuron_B = brick.node_to_neuron['B']
        neuron_C = brick.node_to_neuron['C']
        aux_nodes = [k for k in brick.node_to_neuron.keys() if isinstance(k, str) and k.startswith('A__aux__')]
        self.assertEqual(len(aux_nodes), 1)
        aux = aux_nodes[0]
        neuron_aux = brick.node_to_neuron[aux]
        self.assertTrue(G.has_edge(neuron_A, neuron_aux))
        self.assertEqual(G[neuron_A][neuron_aux]['delay'], 1)
        self.assertTrue(G.has_edge(neuron_aux, neuron_B))
        self.assertEqual(G[neuron_aux][neuron_B]['delay'], 1)
        self.assertTrue(G.has_edge(neuron_B, neuron_aux))
        self.assertEqual(G[neuron_B][neuron_aux]['delay'], 2)
        self.assertTrue(G.has_edge(neuron_A, neuron_C))
        self.assertEqual(G[neuron_A][neuron_C]['delay'], 1)
        self.assertTrue(G.has_edge(neuron_C, neuron_A))
        self.assertEqual(G[neuron_C][neuron_A]['delay'], 1)

    def test_construction_adjacency_matrix(self):
        mat = [
            [0, 2, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
        brick = LoihiGSBrick(mat, name='MergedMatrix')
        G = nx.DiGraph()
        brick.build(G, None, None, None, None)
        n0 = brick.node_to_neuron[0]
        n1 = brick.node_to_neuron[1]
        n2 = brick.node_to_neuron[2]
        self.assertEqual(G[n0][n1]['delay'], 1)
        self.assertEqual(G[n1][n2]['delay'], 1)
        self.assertEqual(G[n1][n0]['delay'], 2)
        self.assertEqual(G[n2][n1]['delay'], 1)

    def test_construction_float_cost_quantize(self):
        adj = {'A': [('B', 2.7)], 'B': []}
        brick = LoihiGSBrick(adj, name='MergedFloat', require_integer_costs=False)
        G = nx.DiGraph()
        brick.build(G, None, None, None, None)
        nA = brick.node_to_neuron['A']
        nB = brick.node_to_neuron['B']
        self.assertEqual(G[nA][nB]['delay'], 1)
        self.assertEqual(G[nB][nA]['delay'], 3)

    def test_construction_disconnected_raises(self):
        adj = {'A': [('B', 1)], 'C': [('D', 1)], 'B': [], 'D': []}
        brick = LoihiGSBrick(adj, name='MergedDisconn')
        G = nx.DiGraph()
        with self.assertRaises(ValueError):
            brick.build(G, None, None, None, None)

    def test_construction_branching_aux_nodes(self):
        adj = {'P': [('A', 1), ('B', 4), ('C', 3)], 'A': [], 'B': [], 'C': []}
        brick = LoihiGSBrick(adj, name='MergedBranch')
        G = nx.DiGraph()
        brick.build(G, None, None, None, None)
        aux_keys = [k for k in brick.node_to_neuron.keys() if isinstance(k, str) and k.startswith('P__aux__')]
        self.assertEqual(len(aux_keys), 2)

    # Transformation / fanout tests (selected core assertions)
    def test_transformation_single_edge(self):
        adj = {0: [(1, 5)], 1: []}
        brick = LoihiGSBrick(adj, source=0, destination=1, name='MergedSingle')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks()
        graph = scaffold.graph
        n0 = brick.node_to_neuron[0]; n1 = brick.node_to_neuron[1]
        self.assertEqual(graph[n0][n1]['delay'], 1)
        self.assertEqual(graph[n1][n0]['delay'], 5)

    def test_transformation_cycle_weakly_connected(self):
        adj = {0: [(1, 2)], 1: [(2, 3)], 2: [(0, 1)]}
        brick = LoihiGSBrick(adj, source=0, destination=2, name='MergedCycle')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks()
        self.assertGreater(len(scaffold.graph.nodes), 0)

    def test_transformation_all_cost_one_no_aux(self):
        adj = {0: [(1,1),(2,1),(3,1)],1:[(4,1)],2:[(4,1)],3:[(4,1)],4:[]}
        brick = LoihiGSBrick(adj, source=0, destination=4, name='MergedAllOnes')
        nodes, edges = brick._parse_input(); proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
        self.assertEqual(len(aux_nodes), 0)
        self.assertEqual(len(proc_nodes), len(nodes))

    def test_transformation_high_fanout_aux_created(self):
        adj = {0: [(i, i+1) for i in range(1,8)], **{i: [] for i in range(1,8)}}
        brick = LoihiGSBrick(adj, source=0, destination=7, name='MergedHighFanout')
        nodes, edges = brick._parse_input(); proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        aux_nodes = [n for n in proc_nodes if '__aux__' in str(n)]
        self.assertEqual(len(aux_nodes), 7)
        fanout_map = {}
        for u,v,c in proc_edges:
            fanout_map.setdefault(u, []).append((v,c))
        costs = [c for v,c in fanout_map[0]]
        self.assertTrue(all(c==1 for c in costs))

    def test_transformation_chain_preservation(self):
        adj = {0:[(1,5)],1:[(2,3)],2:[(3,7)],3:[]}
        brick = LoihiGSBrick(adj, source=0, destination=3, name='MergedChain')
        nodes, edges = brick._parse_input(); proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        self.assertEqual(len([n for n in proc_nodes if '__aux__' in str(n)]), 0)
        self.assertEqual(len(proc_edges), len(edges))

    def test_transformation_backward_delay_encoding(self):
        adj = {0:[(1,7)],1:[(2,13)],2:[(3,23)],3:[]}
        brick = LoihiGSBrick(adj, source=0, destination=3, name='MergedDelays')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); graph = scaffold.graph
        n0 = brick.node_to_neuron[0]; n1 = brick.node_to_neuron[1]; n2 = brick.node_to_neuron[2]; n3 = brick.node_to_neuron[3]
        self.assertEqual(graph[n1][n0]['delay'], 7)
        self.assertEqual(graph[n2][n1]['delay'], 13)
        self.assertEqual(graph[n3][n2]['delay'], 23)

    def test_transformation_forward_delay_fixed(self):
        adj = {0:[(1,10),(2,20)],1:[(3,30)],2:[(3,40)],3:[]}
        brick = LoihiGSBrick(adj, source=0, destination=3, name='MergedForward')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); graph = scaffold.graph
        for u,v,d in graph.edges(data=True):
            if d.get('direction')=='forward':
                self.assertEqual(d.get('delay'),1)

    def test_transformation_source_destination_metadata(self):
        adj = {0:[(1,2)],1:[(2,3)],2:[]}
        brick = LoihiGSBrick(adj, source=0, destination=2, name='MergedMeta')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); graph = scaffold.graph
        bundle = graph.graph['loihi_gs']
        self.assertEqual(bundle['source'],0); self.assertEqual(bundle['destination'],2)
        src = bundle['source_neuron']; dst = bundle['destination_neuron']
        self.assertTrue(graph.nodes[src].get('is_source'))
        self.assertTrue(graph.nodes[dst].get('is_destination'))

    def test_transformation_neuron_properties(self):
        adj = {0:[(1,5)],1:[]}
        brick = LoihiGSBrick(adj, source=0, destination=1, name='MergedNeuronProps')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); graph = scaffold.graph
        dst_neuron = graph.graph['loihi_gs']['destination_neuron']
        for node,data in graph.nodes(data=True):
            for key in ['threshold','decay','p','potential','index']:
                self.assertIn(key,data)
            self.assertEqual(data['threshold'],0.9)
            self.assertEqual(data['decay'],0)
            self.assertEqual(data['p'],1.0)
            if node==dst_neuron:
                # Destination starts at 1.5 (above threshold) to prevent re-spike after natural first spike
                self.assertEqual(data['potential'],1.5)
            else:
                self.assertEqual(data['potential'],0.0)

    def test_transformation_large_costs(self):
        adj = {0:[(1,64),(2,32)],1:[(3,63)],2:[(3,1)],3:[]}
        brick = LoihiGSBrick(adj, source=0, destination=3, name='MergedLargeCosts')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); graph = scaffold.graph
        max_delay = 0
        for u,v,d in graph.edges(data=True):
            if d.get('direction')=='backward':
                max_delay = max(max_delay, d.get('delay'))
        # cost 64 split into 1 + 63 so max backward delay 63
        self.assertEqual(max_delay,63)

    def test_transformation_aux_naming_uniqueness(self):
        adj = {0:[(1,5),(2,3)],1:[(2,4)],2:[]}
        brick = LoihiGSBrick(adj, source=0, destination=2, name='MergedNaming')
        nodes, edges = brick._parse_input(); proc_nodes, _ = brick._preprocess_fanout(nodes, edges)
        self.assertEqual(len(proc_nodes), len(set(proc_nodes)))

    def test_transformation_complete_small(self):
        adj = {0:[(1,2),(2,3),(3,4)],1:[(0,2),(2,5),(3,6)],2:[(0,3),(1,5),(3,7)],3:[(0,4),(1,6),(2,7)]}
        brick = LoihiGSBrick(adj, source=0, destination=3, name='MergedComplete')
        nodes, edges = brick._parse_input(); proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        # verify fanout constraint
        fanout_map = {}
        for u,v,c in proc_edges:
            fanout_map.setdefault(u, []).append(c)
        for u,costs in fanout_map.items():
            if len(costs)>1:
                self.assertTrue(all(x==1 for x in costs))

    def test_transformation_integration_backend(self):
        adj = {0:[(1,2),(2,5)],1:[(3,3)],2:[(3,1)],3:[]}
        brick = LoihiGSBrick(adj, source=0, destination=3, name='MergedIntegration')
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); backend = gsearch_Backend(); backend.compile(scaffold,{})
        self.assertEqual(backend.fugu_graph.graph['loihi_gs']['source'],0)
        self.assertEqual(backend.fugu_graph.graph['loihi_gs']['destination'],3)

    @pytest.mark.skipif(not os.getenv('FUGU_STRESS'), reason='Set FUGU_STRESS=1 to enable dense 50-node test')
    def test_transformation_dense_graph_50_nodes(self):
        import random, time
        random.seed(42)
        n_nodes = 50; density = 0.4
        adj = {}
        for i in range(n_nodes):
            neighbors=[]
            for j in range(i+1,n_nodes):
                if random.random()<density:
                    neighbors.append((j, random.randint(1,64)))
            adj[i]=neighbors
        brick = LoihiGSBrick(adj, source=0, destination=n_nodes-1, name='MergedDense50')
        nodes, edges = brick._parse_input(); proc_nodes, proc_edges = brick._preprocess_fanout(nodes, edges)
        fanout_map={}
        for u,v,c in proc_edges:
            fanout_map.setdefault(u, []).append(c)
        for u,costs in fanout_map.items():
            if len(costs)>1:
                self.assertTrue(all(x==1 for x in costs))
        scaffold = Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); graph = scaffold.graph
        forward=0; backward=0
        for u,v,d in graph.edges(data=True):
            if d.get('direction')=='forward':
                self.assertEqual(d.get('delay'),1); forward+=1
            else:
                backward+=1; self.assertGreaterEqual(d.get('delay'),1)
        self.assertEqual(forward, backward)

    @pytest.mark.skipif(not os.getenv('FUGU_PERF'), reason='Set FUGU_PERF=1 to enable dense performance comparison test')
    def test_performance_dense_weird_graphs(self):
        """Performance and correctness across multiple dense / pathological graphs.

        Graphs:
          1. DenseRandom80: Erdos-Renyi style dense directed (p=0.5) on 80 nodes.
          2. TwoClusterBridge: Two 40-node cliques sparsely bridged.
          3. HeavyFanoutLayered: 5 layers of size 15, fully connected layer-to-layer.
          4. MixedCycleHybrid: 80 nodes partitioned into groups with internal cycles and random forward edges.

        Measures wall-clock times for:
          - Fugu (brick build + compile + run)
          - Direct LoihiGraphSearch (preprocess + run)
          - Dijkstra (heap-based)
        Ensures all three return identical shortest path cost.
        """
        def dense_random_80():
            import random
            random.seed(7)
            n=80; p=0.5
            adj={i:[] for i in range(n)}
            for i in range(n):
                for j in range(n):
                    if i==j: continue
                    if random.random()<p:
                        adj[i].append((j, random.randint(1,64)))
            return adj,0,n-1,'DenseRandom80'

        def two_cluster_bridge():
            import random
            random.seed(11)
            c=40
            adj={i:[] for i in range(2*c)}
            # clique edges inside each cluster
            for base in [0,c]:
                for i in range(base, base+c):
                    for j in range(base, base+c):
                        if i!=j:
                            adj[i].append((j, (1+((i*13+j*17)%64))))
            # sparse bridges
            bridges=50
            for _ in range(bridges):
                a=random.randint(0,c-1)
                b=random.randint(c,2*c-1)
                adj[a].append((b, random.randint(1,64)))
                adj[b].append((a, random.randint(1,64)))
            return adj,0,2*c-1,'TwoClusterBridge'

        def heavy_fanout_layered():
            import random
            random.seed(19)
            layers=5; size=15
            nodes=[(L,i) for L in range(layers) for i in range(size)]
            id_map={ (L,i): L*size+i for L,i in nodes }
            adj={ id_map[(L,i)]:[] for L,i in nodes }
            for L in range(layers-1):
                for i in range(size):
                    u=id_map[(L,i)]
                    for j in range(size):
                        v=id_map[(L+1,j)]
                        adj[u].append((v, random.randint(1,64)))
            # add destination sink
            sink=layers*size
            adj[sink]=[]
            for i in range(size):
                adj[id_map[(layers-1,i)]].append((sink, random.randint(1,64)))
            return adj, id_map[(0,0)], sink, 'HeavyFanoutLayered'

        def mixed_cycle_hybrid():
            import random
            random.seed(23)
            n=80
            group=8
            adj={i:[] for i in range(n)}
            # internal cycles per group
            for g in range(n//group):
                base=g*group
                for k in range(group):
                    u=base+k; v=base+((k+1)%group)
                    adj[u].append((v, random.randint(1,64)))
            # random forward edges between groups (DAG-ish)
            for u in range(n):
                for _ in range(4):
                    v=random.randint(0,n-1)
                    if v!=u:
                        adj[u].append((v, random.randint(1,64)))
            return adj,0,n-1,'MixedCycleHybrid'

        graphs=[dense_random_80(), two_cluster_bridge(), heavy_fanout_layered(), mixed_cycle_hybrid()]

        results=[]
        for adj, src, dst, tag in graphs:
            # Dijkstra timing
            t0=time.perf_counter(); path_d,cost_d=dijkstra(adj, src, dst); td=time.perf_counter()-t0
            # Direct Loihi timing
            t1=time.perf_counter(); adj_proc=preprocess_fanout_constraint(adj); lg=LoihiGraphSearch(adj_proc, source=src, destination=dst); path_l,_,_steps=lg.run(max_steps=10000); tl=time.perf_counter()-t1
            # Collapse aux for cost
            if path_l:
                max_orig=max(adj.keys())
                path_l_clean=[n for n in path_l if n<=max_orig]
                cost_l=0
                for u,v in zip(path_l_clean[:-1], path_l_clean[1:]):
                    ec=next((c for dst2,c in adj.get(u,[]) if dst2==v), None)
                    if ec is None: cost_l=float('inf'); break
                    cost_l+=ec
            else:
                cost_l=float('inf')
            # Fugu timing
            t2=time.perf_counter(); brick=LoihiGSBrick(adj, source=src, destination=dst, name=f'MergedPerf_{tag}'); scaffold=Scaffold(); scaffold.add_brick(brick, output=True); scaffold.lay_bricks(); backend=gsearch_Backend(); backend.compile(scaffold,{}); res=backend.run(n_steps=10000); tf=time.perf_counter()-t2
            fpath=res['path']; neuron_to_node={v:k for k,v in brick.node_to_neuron.items()}; raw_nodes=[neuron_to_node.get(n,n) for n in fpath if n in neuron_to_node]; # simple normalization
            collapsed=[]
            for n in raw_nodes:
                base = n.split('__aux__')[0] if isinstance(n,str) and '__aux__' in n else n
                try:
                    base_i=int(base)
                except (ValueError,TypeError):
                    base_i=base
                if not collapsed or collapsed[-1]!=base_i:
                    collapsed.append(base_i)
            cost_f=0 if len(collapsed)>1 else (0 if (len(collapsed)==1 and collapsed[0]==src==dst) else float('inf'))
            if len(collapsed)>1:
                for u,v in zip(collapsed[:-1], collapsed[1:]):
                    ec=next((c for dst2,c in adj.get(u,[]) if dst2==v), None)
                    if ec is None: cost_f=float('inf'); break
                    cost_f+=ec
            # Assertions
            self.assertEqual(cost_d, cost_l, f"Direct Loihi cost mismatch on {tag}: Dijkstra={cost_d} Loihi={cost_l}")
            self.assertEqual(cost_d, cost_f, f"Fugu cost mismatch on {tag}: Dijkstra={cost_d} Fugu={cost_f}")
            results.append((tag, cost_d, td, tl, tf))

        # Simple performance sanity: algorithms should produce same cost; print summary
        print("\nPerformance Summary (tag, cost, t_dijkstra, t_loihi_direct, t_fugu):")
        for r in results:
            print(f"  {r[0]}: cost={r[1]} dijk={r[2]:.4f}s loihi={r[3]:.4f}s fugu={r[4]:.4f}s")


if __name__ == '__main__':
    unittest.main(verbosity=3)
