#!/usr/bin/env python3
"""
End-to-end tests for gsearch_backend that actually execute the algorithm.
"""

import pytest
from fugu.bricks.loihi_gs_brick import LoihiGSBrick
from fugu.scaffold import Scaffold
from fugu.backends.gsearch_backend import gsearch_Backend


def test_simple_path():
    """Test simple linear path: 0 -> 1 -> 2 -> 3"""
    adj = {
        0: [(1, 2)],
        1: [(2, 3)],
        2: [(3, 1)],
        3: []
    }
    
    brick = LoihiGSBrick(adj, source=0, destination=3, name='SimplePath')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    
    backend = gsearch_Backend()
    backend.compile(scaffold, compile_args={})

    # Run the algorithm
    result = backend.run(n_steps=100)
    
    # Debug: check what happened
    print(f"\n{'='*70}")
    print(f"RESULT SUMMARY:")
    print(f"  Source spiked: {result['source_spiked']}")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Remaining backward edges: {result['remaining_backward']}")
    print(f"  Path found: {result['path']}")
    print(f"  Path length: {len(result['path'])}")
    
    # Check backward edges before and after
    total_backward = sum(1 for u, v, d in backend.fugu_graph.edges(data=True) if d.get('direction') == 'backward')
    zeroed_backward = sum(1 for u, v, d in backend.fugu_graph.edges(data=True) 
                          if d.get('direction') == 'backward' and d.get('weight', 0) == 0)
    print(f"\n  Total backward edges: {total_backward}")
    print(f"  Zeroed backward edges: {zeroed_backward}")
    print(f"  Non-zero backward edges: {total_backward - zeroed_backward}")
    
    # Show ALL backward edges (both zero and non-zero)
    print(f"\n  All backward edges:")
    for u, v, d in backend.fugu_graph.edges(data=True):
        if d.get('direction') == 'backward':
            print(f"    {u} -> {v} (delay={d.get('delay')}, weight={d.get('weight')})")
    
    # Show spike times
    print(f"\n  Spike times:")
    for name, time in backend.spike_time.items():
        if time >= 0:
            print(f"    {name}: t={time}")
    
    print(f"{'='*70}\n")
    
    # Verify results
    assert result['source_spiked'], "Source should have spiked"
    assert len(result['path']) > 0, "Should find a path"
    
    # Convert neuron names back to original node labels
    neuron_to_node = {v: k for k, v in brick.node_to_neuron.items()}
    path_labels = [neuron_to_node.get(n) for n in result['path'] if n in neuron_to_node]
    
    print(f"\nPath found: {path_labels}")
    print(f"Steps taken: {result['steps']}")
    print(f"Remaining backward edges: {result['remaining_backward']}")
    
    # Should find path 0 -> 1 -> 2 -> 3
    assert path_labels[0] == 0, "Path should start at source"
    assert path_labels[-1] == 3, "Path should end at destination"


def test_shortest_path_selection():
    """Test that algorithm finds shortest path when multiple paths exist."""
    adj = {
        0: [(1, 2), (2, 10)],  # Two paths: 0->1->3 (cost 5) or 0->2->3 (cost 11)
        1: [(3, 3)],
        2: [(3, 1)],
        3: []
    }
    
    brick = LoihiGSBrick(adj, source=0, destination=3, name='ShortestPath')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    
    backend = gsearch_Backend()
    backend.compile(scaffold, compile_args={})
    
    result = backend.run(n_steps=100)
    
    assert result['source_spiked'], "Source should have spiked"
    assert len(result['path']) > 0, "Should find a path"
    
    # Convert path
    neuron_to_node = {v: k for k, v in brick.node_to_neuron.items()}
    path_labels = [neuron_to_node.get(n) for n in result['path'] if n in neuron_to_node]
    
    print(f"\nPath found: {path_labels}")
    
    # Should take shorter path through node 1, not node 2
    assert 1 in path_labels, "Should use shorter path through node 1"
    assert 2 not in path_labels, "Should not use longer path through node 2"


def test_diamond_graph():
    """Test diamond topology with converging paths."""
    adj = {
        0: [(1, 3), (2, 5)],  # Split to 1 and 2
        1: [(3, 2)],          # Converge at 3
        2: [(3, 1)],          # 0->1->3 costs 5, 0->2->3 costs 6
        3: []
    }
    
    brick = LoihiGSBrick(adj, source=0, destination=3, name='Diamond')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    
    backend = gsearch_Backend()
    backend.compile(scaffold, compile_args={})
    
    result = backend.run(n_steps=100)
    
    assert result['source_spiked'], "Source should have spiked"
    assert len(result['path']) > 0, "Should find a path"
    
    neuron_to_node = {v: k for k, v in brick.node_to_neuron.items()}
    path_labels = [neuron_to_node.get(n) for n in result['path'] if n in neuron_to_node]
    
    print(f"\nPath found: {path_labels}")
    print(f"Steps: {result['steps']}")
    
    # Should find shorter path (0->1->3, cost 5)
    assert path_labels == [0, 1, 3] or all(n in [0, 1, 3] for n in path_labels if n is not None)


def test_no_path():
    """Test behavior when no path exists (disconnected graph raises during brick construction)."""
    # This should fail during brick creation due to connectivity check
    adj = {
        0: [(1, 2)],
        1: [],
        2: [(3, 1)],  # Disconnected from 0,1
        3: []
    }
    
    brick = LoihiGSBrick(adj, source=0, destination=3, name='NoPath')
    
    with pytest.raises(ValueError, match="weakly connected"):
        scaffold = Scaffold()
        scaffold.add_brick(brick, output=True)
        scaffold.lay_bricks()


def test_high_cost_edges():
    """Test with high-cost edges (near max cost of 64)."""
    adj = {
        0: [(1, 63), (2, 32)],  # High costs
        1: [(3, 1)],
        2: [(3, 32)],
        3: []
    }
    
    brick = LoihiGSBrick(adj, source=0, destination=3, name='HighCost')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    
    backend = gsearch_Backend()
    backend.compile(scaffold, compile_args={})
    
    # May need more steps for high-cost edges
    result = backend.run(n_steps=200)
    
    assert result['source_spiked'], "Source should spike even with high costs"
    assert len(result['path']) > 0, "Should find a path"
    
    neuron_to_node = {v: k for k, v in brick.node_to_neuron.items()}
    path_labels = [neuron_to_node.get(n) for n in result['path'] if n in neuron_to_node]
    
    print(f"\nPath found: {path_labels}")
    print(f"Steps: {result['steps']}")
    
    # Should use path through node 2 (cost 64) rather than node 1 (cost 64)
    assert path_labels[0] == 0 and path_labels[-1] == 3


def test_auxiliary_nodes_in_path():
    """Test that auxiliary nodes created by fanout preprocessing work correctly."""
    adj = {
        0: [(1, 10), (2, 5)],  # Node 0 has fanout=2, will create aux nodes
        1: [(3, 1)],
        2: [(3, 15)],
        3: []
    }
    
    brick = LoihiGSBrick(adj, source=0, destination=3, name='AuxNodes')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    
    backend = gsearch_Backend()
    backend.compile(scaffold, compile_args={})
    
    result = backend.run(n_steps=100)
    
    assert result['source_spiked'], "Source should spike"
    assert len(result['path']) > 0, "Should find path through auxiliary nodes"
    
    # Path may include auxiliary nodes, but should still connect source to destination
    bundle = backend.fugu_graph.graph['loihi_gs']
    assert result['path'][0] == bundle['source_neuron']
    assert result['path'][-1] == bundle['destination_neuron']
    
    print(f"\nFull path (with aux nodes): {result['path']}")
    print(f"Path length: {len(result['path'])}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
