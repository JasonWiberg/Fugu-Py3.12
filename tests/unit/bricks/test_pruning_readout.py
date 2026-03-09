"""Test harness for debugging gsearch_backend pruning and readout."""

import networkx as nx
from fugu import Scaffold
from fugu.bricks.loihi_gs_brick import LoihiGSBrick
from fugu.backends.gsearch_backend import gsearch_Backend

def test_simple_chain_pruning():
    """Test pruning logic on simple chain: 0->1->2->3"""
    # Graph: 0->1 (cost 3), 1->2 (cost 4), 2->3 (cost 3)
    adj = {0: [(1, 3)], 1: [(2, 4)], 2: [(3, 3)], 3: []}
    
    # Build scaffold
    scaffold = Scaffold()
    brick = LoihiGSBrick(input_graph=adj, source=0, destination=3, name='Test')
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    
    # Compile backend
    backend = gsearch_Backend()
    backend.compile(scaffold, {})
    
    # Get neuron names
    bundle = backend.fugu_graph.graph['loihi_gs']
    dst = bundle['destination_neuron']
    src = bundle['source_neuron']
    
    # Map node indices to neuron names
    node_to_neuron = {}
    for name, data in backend.fugu_graph.nodes(data=True):
        idx = data.get('index')
        if idx is not None:
            node_to_neuron[idx] = name
    
    print("=== GRAPH STRUCTURE ===")
    print(f"Source: {src} (node 0)")
    print(f"Destination: {dst} (node 3)")
    print(f"\nNeuron mapping:")
    for i in sorted(node_to_neuron.keys()):
        print(f"  Node {i}: {node_to_neuron[i]}")
    
    print(f"\n=== BACKWARD EDGES ===")
    for u, v, data in backend.fugu_graph.edges(data=True):
        if data.get('direction') == 'backward':
            u_idx = backend.fugu_graph.nodes[u]['index']
            v_idx = backend.fugu_graph.nodes[v]['index']
            delay = data.get('delay')
            weight = data.get('weight')
            print(f"  ({u}, {v}): node {u_idx}->node {v_idx}, delay={delay}, weight={weight}")
    
    print(f"\n=== FORWARD EDGES ===")
    for u, v, data in backend.fugu_graph.edges(data=True):
        if data.get('direction') == 'forward':
            u_idx = backend.fugu_graph.nodes[u]['index']
            v_idx = backend.fugu_graph.nodes[v]['index']
            delay = data.get('delay')
            weight = data.get('weight')
            print(f"  ({u}, {v}): node {u_idx}->node {v_idx}, delay={delay}, weight={weight}")
    
    # Check w_backward_original
    print(f"\n=== W_BACKWARD_ORIGINAL ===")
    w_back_orig = bundle.get('w_backward_original', {})
    for key, weight in w_back_orig.items():
        u, v = key
        u_idx = backend.fugu_graph.nodes[u]['index']
        v_idx = backend.fugu_graph.nodes[v]['index']
        print(f"  {key}: node {u_idx}->node {v_idx}, weight={weight}")
    
    # Run simulation
    print(f"\n=== SIMULATION ===")
    backend.current_timestep = 0
    backend.spike_time[dst] = 0
    # Prime destination to spike at t=0 (matches backend.run())
    backend.nn.nrns[dst].spike = True
    backend.nn.nrns[dst].spike_hist.append(True)
    print(f"t=0: Destination {dst} primed to spike")
    
    source_spiked = False
    while not source_spiked and backend.current_timestep < 20:
        backend.current_timestep += 1
        backend.nn.step()
        
        # Let prune_step detect newly spiked neurons
        diag = backend.prune_step()
        source_spiked = diag.get('source_spiked', False)
        
        # Check which neurons spiked (for debugging)
        for name, neuron in backend.nn.nrns.items():
            if getattr(neuron, 'spike', False):
                if backend.spike_time.get(name, -1) == backend.current_timestep:
                    idx = backend.fugu_graph.nodes[name]['index']
                    print(f"t={backend.current_timestep}: Neuron {name} (node {idx}) spiked")
    
    # Extra step
    if source_spiked:
        backend.current_timestep += 1
        backend.nn.step()
        backend.prune_step()
    
    print(f"\n=== AFTER PRUNING ===")
    print(f"Spike times: {backend.spike_time}")
    
    print(f"\n=== REMAINING BACKWARD EDGES ===")
    for u, v, data in backend.fugu_graph.edges(data=True):
        if data.get('direction') == 'backward' and data.get('weight', 0) > 0:
            u_idx = backend.fugu_graph.nodes[u]['index']
            v_idx = backend.fugu_graph.nodes[v]['index']
            print(f"  ({u}, {v}): node {u_idx}->node {v_idx}, weight={data.get('weight')}")
    
    print(f"\n=== PATH RECONSTRUCTION ===")
    backend.current_hop = src
    path = [src]
    visited = {src}
    
    for step in range(10):
        print(f"\nStep {step}: current_hop = {backend.current_hop} (node {backend.fugu_graph.nodes[backend.current_hop]['index']})")
        
        # Check forward edges
        for _, nxt, fwd_data in backend.fugu_graph.out_edges(backend.current_hop, data=True):
            if fwd_data.get('direction') == 'forward':
                nxt_idx = backend.fugu_graph.nodes[nxt]['index']
                print(f"  Forward edge to {nxt} (node {nxt_idx})")
                
                # Check backward edge
                back_key = (nxt, backend.current_hop)
                if back_key in w_back_orig:
                    orig_weight = w_back_orig[back_key]
                    print(f"    Backward key {back_key} in w_backward_original: {orig_weight}")
                    
                    if backend.fugu_graph.has_edge(nxt, backend.current_hop):
                        curr_weight = backend.fugu_graph[nxt][backend.current_hop].get('weight', 0)
                        print(f"    Current backward edge weight: {curr_weight}")
                        if curr_weight == 0:
                            print(f"    -> PRUNED! This is the next hop")
                    else:
                        print(f"    ERROR: Backward edge not in graph!")
        
        nxt = backend.readout_next_hop()
        if nxt is None or nxt in visited:
            print(f"  readout_next_hop returned: {nxt}")
            break
        
        print(f"  Next hop: {nxt} (node {backend.fugu_graph.nodes[nxt]['index']})")
        path.append(nxt)
        visited.add(nxt)
        backend.current_hop = nxt
        
        if nxt == dst:
            print(f"  Reached destination!")
            break
    
    print(f"\nFinal path: {path}")
    print(f"Expected: [src={src}, ..., dst={dst}]")
    
    # Convert to node indices
    path_indices = [backend.fugu_graph.nodes[n]['index'] for n in path]
    print(f"Path as node indices: {path_indices}")
    print(f"Expected: [0, 1, 2, 3]")

if __name__ == '__main__':
    test_simple_chain_pruning()
