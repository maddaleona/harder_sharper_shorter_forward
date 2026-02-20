##############################################################
# NETWORK METRICS
##############################################################

import numpy as np
import networkx as nx


def compute_network_outreach(G):
    """
    Spatially weighted dispersion of passes.
    Sum of (edge_weight * Euclidean_distance) over all edges,
    normalized by total network strength.
    Higher = longer-range ball circulation.
    """
    outreach_values = []
    for node in G.nodes():
        node_outreach = 0.0
        coord_node = G.nodes[node]['ref_coordinate']
        for neighbor in G.successors(node):
            w = G[node][neighbor]['weight']
            coord_neigh = G.nodes[neighbor]['ref_coordinate']
            dist = np.linalg.norm(np.array(coord_neigh) - np.array(coord_node))
            node_outreach += w * dist
        outreach_values.append(node_outreach)
    return float(np.mean(outreach_values))


def compute_max_eigenvalue(G):
    """
    Principal eigenvalue of the weighted adjacency matrix.
    Reflects overall connectivity and hierarchical structure.
    """
    try:
        A = nx.to_numpy_array(G, weight='weight')
        eigenvalues = np.linalg.eigvals(A)
        return float(np.max(np.abs(eigenvalues)))
    except Exception:
        return np.nan


def compute_avg_shortest_path(G):
    """
    Average shortest path length on the largest strongly connected component.
    Lower = more efficient ball progression.
    """
    try:
        if nx.is_strongly_connected(G):
            G_sub = G
        else:
            scc = max(nx.strongly_connected_components(G), key=len)
            G_sub = G.subgraph(scc).copy()
        return float(nx.average_shortest_path_length(G_sub, weight='weight', method='dijkstra'))
    except Exception:
        return np.nan


def compute_all_network_metrics(G):
    """
    Compute all network metrics reported in the paper.

    Parameters
    ----------
    G : nx.DiGraph
        Normalized pitch-passing network.

    Returns
    -------
    dict with keys: network_outreach, max_eigenvalue, avg_shortest_path
    """
    return {
        'network_outreach':   compute_network_outreach(G),
        'max_eigenvalue':     compute_max_eigenvalue(G),
        'avg_shortest_path':  compute_avg_shortest_path(G),
    }
