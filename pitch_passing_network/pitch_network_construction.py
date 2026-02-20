##############################################################
# NETWORK CONSTRUCTION
##############################################################

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

# === PARAMETERS ===
N_X_BINS = 10
N_Y_BINS = 5
MIN_X, MAX_X = 0.0, 120.0
MIN_Y, MAX_Y = 0.0, 80.0
X_BINS = np.linspace(MIN_X, MAX_X, N_X_BINS + 1)
Y_BINS = np.linspace(MIN_Y, MAX_Y, N_Y_BINS + 1)
COORDINATE_GRID = np.arange(N_X_BINS * N_Y_BINS).reshape((N_Y_BINS, N_X_BINS), order='F')


def get_passing_df(df):
    """Filter only successful passes."""
    return df[(df['type_name'] == 'Pass') & (df['pass_outcome_name'].isna())]


def get_cell(location):
    """Map (x, y) coordinates to grid cell index."""
    x, y = location
    H, _, _ = np.histogram2d([y], [x], bins=[Y_BINS, X_BINS])
    return COORDINATE_GRID[np.where(H)][0]


def add_cell_allocations(df):
    """Add source and target node columns based on pass locations."""
    df = df.copy()
    df['location'] = df['location'].apply(lambda loc: list(np.clip(loc, [0, 0], [120, 80])))
    df['pass_end_location'] = df['pass_end_location'].apply(lambda loc: list(np.clip(loc, [0, 0], [120, 80])))
    df['location_node'] = df['location'].apply(get_cell)
    df['pass_end_location_node'] = df['pass_end_location'].apply(get_cell)
    return df


def get_reference_coordinate(node_label):
    """Get the reference (x, y) coordinate for a grid node."""
    node_y_idx, node_x_idx = np.where(node_label == COORDINATE_GRID)
    return X_BINS[node_x_idx], Y_BINS[node_y_idx]


def build_network(df):
    """Build a directed weighted pitch-passing network from pass data."""
    G = nx.DiGraph()
    G.add_nodes_from(range(N_X_BINS * N_Y_BINS))
    for node in G.nodes():
        G.nodes[node]['ref_coordinate'] = get_reference_coordinate(node)
    edge_counts = Counter(zip(df['location_node'], df['pass_end_location_node']))
    for (src, tgt), count in edge_counts.items():
        G.add_edge(src, tgt, weight=count)
    return G


def normalize_network(G):
    """Normalize edge weights so they sum to 100."""
    G = G.copy()
    total = sum(d['weight'] for _, _, d in G.edges(data=True))
    if total > 0:
        for _, _, d in G.edges(data=True):
            d['weight'] = d['weight'] * 100 / total
    return G


def build_team_networks(df):
    """
    Given a match DataFrame, return normalized pitch-passing networks
    for home and away teams.

    Parameters
    ----------
    df : pd.DataFrame
        Raw match event data (JSON-loaded).

    Returns
    -------
    dict with keys 'home_team', 'away_team', 'G_home', 'G_away'
    """
    pass_df = get_passing_df(df)
    pass_df = add_cell_allocations(pass_df)

    home_team = df['home_team'].iloc[0]
    away_team = df['away_team'].iloc[0]

    G_home = normalize_network(build_network(pass_df[pass_df['possession_team_name'] == home_team]))
    G_away = normalize_network(build_network(pass_df[pass_df['possession_team_name'] == away_team]))

    return {'home_team': home_team, 'away_team': away_team, 'G_home': G_home, 'G_away': G_away}
