import random
import networkx as nx
from fugu import Scaffold
from fugu.bricks import LoihiGSBrick
from fugu.backends import gsearch_Backend


def run_loihi_search(adj, source, destination, n_steps=500):
    """Helper: build brick + backend, run search, return path (original node labels) and cost."""
    brick = LoihiGSBrick(adj, source=source, destination=destination, name='GS')
    scaffold = Scaffold()
    scaffold.add_brick(brick, output=True)
    scaffold.lay_bricks()
    backend = gsearch_Backend()
    backend.compile(scaffold, compile_args={})
    result = backend.run(n_steps=n_steps)
    neuron_path = result['path']
    bundle = scaffold.graph.graph.get('loihi_gs', {})
    node_to_neuron = bundle.get('node_to_neuron', {})
    neuron_to_node = {v: k for k, v in node_to_neuron.items()}

    def normalize(label):
        if isinstance(label, int):
            return label
        if isinstance(label, str) and '__aux__' in label:
            base = label.split('__aux__')[0]
            # Fanout preprocessing may keep base as int or str; attempt int conversion
            try:
                return int(base)
            except ValueError:
                return base  # fallback
        return label if not isinstance(label, str) else (int(label) if label.isdigit() else label)

    raw_nodes = [neuron_to_node[n] for n in neuron_path if n in neuron_to_node]
    norm_nodes = [normalize(x) for x in raw_nodes]
    # Collapse consecutive duplicates (aux expansions)
    collapsed = []
    for node in norm_nodes:
        if not collapsed or collapsed[-1] != node:
            collapsed.append(node)

    # Compute path cost from original adjacency using collapsed list
    cost = 0
    for u, v in zip(collapsed[:-1], collapsed[1:]):
        if u == v:
            continue
        for w, c in adj.get(u, []):
            if w == v:
                cost += c
                break
    return collapsed, cost, result


def dijkstra_cost(adj, source, destination):
    G = nx.DiGraph()
    for u, edges in adj.items():
        for v, c in edges:
            G.add_edge(u, v, weight=c)
    try:
        path = nx.shortest_path(G, source, destination, weight='weight')
        cost = nx.shortest_path_length(G, source, destination, weight='weight')
        return path, cost
    except nx.NetworkXNoPath:
        return [], None


def gen_random_adj(n_nodes=8, edge_prob=0.3, min_cost=1, max_cost=6, seed=0):
    random.seed(seed)
    adj = {i: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            if random.random() < edge_prob:
                c = random.randint(min_cost, max_cost)
                adj[i].append((j, c))
    return adj


def assert_cost_match(loihi_path, loihi_cost, dijkstra_path, dijkstra_cost_val):
    assert dijkstra_path, "Reference Dijkstra found no path; graph likely disconnected for chosen pair"
    assert loihi_path, "Loihi search produced no path"
    assert loihi_cost == dijkstra_cost_val, (
        f"Cost mismatch: Loihi={loihi_cost} Dijkstra={dijkstra_cost_val}\n"
        f"Loihi path={loihi_path}\nDijkstra path={dijkstra_path}"
    )


def choose_connected_pair(adj):
    # Build graph and pick a pair with path
    G = nx.DiGraph()
    for u, edges in adj.items():
        for v, c in edges:
            G.add_edge(u, v, weight=c)
    nodes = list(G.nodes())
    for src in nodes:
        for dst in nodes:
            if src == dst:
                continue
            if nx.has_path(G, src, dst):
                return src, dst
    return None, None


def test_branching_graph():
    # A diamond shape with asymmetric costs to force choice of cheaper middle path
    adj = {
        0: [(1, 5), (2, 2)],
        1: [(3, 1)],
        2: [(3, 4)],
        3: []
    }
    src, dst = 0, 3
    loihi_path, loihi_cost_val, _ = run_loihi_search(adj, src, dst)
    d_path, d_cost_val = dijkstra_cost(adj, src, dst)
    assert_cost_match(loihi_path, loihi_cost_val, d_path, d_cost_val)


def test_equal_cost_alternatives():
    # Two parallel equal-cost routes; accept any minimal cost
    adj = {
        0: [(1, 2), (2, 2)],
        1: [(3, 2)],
        2: [(3, 2)],
        3: []
    }
    src, dst = 0, 3
    loihi_path, loihi_cost_val, _ = run_loihi_search(adj, src, dst)
    _, d_cost_val = dijkstra_cost(adj, src, dst)
    assert loihi_cost_val == d_cost_val, f"Expected cost {d_cost_val}, got {loihi_cost_val}"


def test_long_chain():
    # Linear chain with varying costs
    n = 10
    adj = {i: [(i+1, (i % 3) + 1)] for i in range(n-1)}
    adj[n-1] = []
    src, dst = 0, n-1
    loihi_path, loihi_cost_val, _ = run_loihi_search(adj, src, dst)
    d_path, d_cost_val = dijkstra_cost(adj, src, dst)
    assert_cost_match(loihi_path, loihi_cost_val, d_path, d_cost_val)


def test_grid_graph():
    # 3x3 grid, directed right and down with random costs
    size = 3
    adj = {}
    rng = random.Random(42)
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            edges = []
            if c + 1 < size:
                edges.append((idx + 1, rng.randint(1, 4)))
            if r + 1 < size:
                edges.append((idx + size, rng.randint(1, 4)))
            adj[idx] = edges
    src, dst = 0, size*size - 1
    loihi_path, loihi_cost_val, _ = run_loihi_search(adj, src, dst)
    d_path, d_cost_val = dijkstra_cost(adj, src, dst)
    assert_cost_match(loihi_path, loihi_cost_val, d_path, d_cost_val)


def test_random_graphs_multiple():
    # Run multiple seeds to stress variety
    for seed in range(5):
        adj = gen_random_adj(n_nodes=12, edge_prob=0.25, seed=seed)
        # Require whole graph to be weakly connected (LoihiGSBrick constraint)
        G = nx.DiGraph()
        G.add_nodes_from(adj.keys())
        for u, edges in adj.items():
            for v, c in edges:
                G.add_edge(u, v, weight=c)
        if not nx.is_weakly_connected(G):
            continue  # skip graphs violating brick requirement
        src, dst = choose_connected_pair(adj)
        if src is None:
            continue  # no source/destination with path
        loihi_path, loihi_cost_val, _ = run_loihi_search(adj, src, dst)
        d_path, d_cost_val = dijkstra_cost(adj, src, dst)
        assert_cost_match(loihi_path, loihi_cost_val, d_path, d_cost_val)
