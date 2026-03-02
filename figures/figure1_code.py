import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from mplsoccer import Pitch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


def create_network_plot_for_subplot(ax, csv_file, title):
    """
    Crea un plot di rete su un asse specifico usando mplsoccer.
    """
    df = pd.read_csv(csv_file)

    length = 120
    width  = 80

    # FIX 4: set axis limits explicitly so patches/scatter render correctly
    ax.set_xlim(0, length)
    ax.set_ylim(0, width)

    # Grass stripes background
    light_green = '#4CAF50'
    dark_green  = '#2E7D32'
    stripe_width = 4
    n_stripes = int(width / stripe_width)

    for i in range(n_stripes + 1):
        y_start = i * stripe_width
        y_end   = min((i + 1) * stripe_width, width)
        color   = light_green if i % 2 == 0 else dark_green
        stripe  = patches.Rectangle(
            (0, y_start), length, y_end - y_start,
            linewidth=0, facecolor=color, alpha=0.15, zorder=1
        )
        ax.add_patch(stripe)

    rect_length   = 12
    rect_width    = 16
    n_divisions_x = length // rect_length
    n_divisions_y = width  // rect_width

    for i in range(1, n_divisions_x):
        ax.plot([i * rect_length] * 2, [0, width],
                color='white', linewidth=1, alpha=0.85, linestyle='--')
    for i in range(1, n_divisions_y):
        ax.plot([0, length], [i * rect_width] * 2,
                color='white', linewidth=1, alpha=0.85, linestyle='--')

    # Build graph
    G = nx.Graph()
    for _, row in df.iterrows():
        src = int(row['source'])
        tgt = int(row['target'])
        w   = float(row['weight'])
        G.add_edge(src, tgt, weight=w)

    invalid_nodes = [n for n in G.nodes() if n < 0 or n > 49]
    if invalid_nodes:
        G.remove_nodes_from(invalid_nodes)

    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception:
        deg = dict(G.degree())
        max_deg = max(deg.values()) if deg else 1
        eigenvector_centrality = {n: deg[n] / max_deg for n in G.nodes()}

    # Node positions and sizes
    nodes_positions = {}
    node_sizes      = {}
    ec_values       = list(eigenvector_centrality.values())
    ec_min          = min(ec_values) if ec_values else 0
    ec_max          = max(ec_values) if ec_values else 1

    for node_id in range(50):
        j        = node_id // n_divisions_y
        i        = node_id  % n_divisions_y
        center_x = j * rect_length + rect_length / 2
        center_y = i * rect_width  + rect_width  / 2
        nodes_positions[node_id] = (center_x, center_y)

        if node_id in G.nodes():
            ec_val = eigenvector_centrality.get(node_id, 0)
            norm   = (ec_val - ec_min) / (ec_max - ec_min) if ec_max > ec_min else 0
            node_sizes[node_id] = 50 + (norm ** 2) * 350
        else:
            node_sizes[node_id] = 80

    # Draw edges
    if G.edges():
        weights    = [d['weight'] for _, _, d in G.edges(data=True)]
        min_w, max_w = min(weights), max(weights)

        regular_edges   = [(u, v, d) for u, v, d in G.edges(data=True) if u != v]
        self_loop_edges = [(u, v, d) for u, v, d in G.edges(data=True) if u == v]

        for node1, node2, edge_data in regular_edges:
            if node1 not in nodes_positions or node2 not in nodes_positions:
                continue
            w    = edge_data['weight']
            norm = (w - min_w) / (max_w - min_w) if max_w > min_w else 0.5
            lw   = 0.2 + norm * 2.0
            gi   = 0.6 - norm * 0.6
            x1, y1 = nodes_positions[node1]
            x2, y2 = nodes_positions[node2]

            mid_x    = (x1 + x2) / 2
            mid_y    = (y1 + y2) / 2
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            curve_dir    = 1 if (node1 + node2) % 2 == 0 else -1
            curve_offset = curve_dir * (distance * 0.1 + 2)

            if x2 != x1:
                px = -(y2 - y1)
                py =   x2 - x1
                pl = np.sqrt(px**2 + py**2)
                px, py = px / pl * curve_offset, py / pl * curve_offset
            else:
                px, py = curve_offset, 0

            t      = np.linspace(0, 1, 50)
            curve_x = (1-t)**2 * x1 + 2*(1-t)*t * (mid_x+px) + t**2 * x2
            curve_y = (1-t)**2 * y1 + 2*(1-t)*t * (mid_y+py) + t**2 * y2
            ax.plot(curve_x, curve_y,
                    color=(gi, gi, gi), linewidth=lw, alpha=0.8,
                    zorder=5, solid_capstyle='round')

        for node_id, _, edge_data in self_loop_edges:
            if node_id not in nodes_positions:
                continue
            w    = edge_data['weight']
            norm = (w - min_w) / (max_w - min_w) if max_w > min_w else 0.5
            lw   = 0.2 + norm * 2.0
            gi   = 0.6 - norm * 0.6
            ls   = 3 + norm * 8
            x, y = nodes_positions[node_id]

            t      = np.linspace(0, 1, 100)
            sx1, sy1 = x + 2, y + 2
            ex1, ey1 = x - 2, y + 2
            c1x, c1y = x + ls, y + ls
            c2x, c2y = x - ls, y + ls
            lx = (1-t)**3*sx1 + 3*(1-t)**2*t*c1x + 3*(1-t)*t**2*c2x + t**3*ex1
            ly = (1-t)**3*sy1 + 3*(1-t)**2*t*c1y + 3*(1-t)*t**2*c2y + t**3*ey1
            ax.plot(lx, ly,
                    color=(gi, gi, gi), linewidth=lw, alpha=0.8,
                    zorder=6, solid_capstyle='round')

    # Draw nodes
    for node_id, (x, y) in nodes_positions.items():
        size      = node_sizes[node_id]
        isolated  = node_id not in G.nodes()
        nc        = 'limegreen' if isolated else 'honeydew'
        has_loop  = any(u == v == node_id for u, v, _ in G.edges(data=True))
        ec_color  = 'white' if isolated else ('red' if has_loop else 'gold')
        ax.scatter(x, y, s=size, c=nc, alpha=1,
                   edgecolors=ec_color, linewidth=1, zorder=10)

    ax.set_title(title, fontsize=12, y=0.95)


def visualize_complete_analysis(data, home_team, away_team):
    """
    Visualizzazione completa: All Events | All Passes | Pitch Networks.

    Parameters
    ----------
    data      : list of dicts — StatsBomb event data
    home_team : str — home team name  (e.g. 'Arsenal')
    away_team : str — away team name  (e.g. 'Liverpool')
    """
    aspect = 2.5 / 2
    w = aspect * 10.0
    h = aspect * 8.5

    fig, ax = plt.subplots(3, 2, figsize=(w, h), dpi=300)
    plt.subplots_adjust(
        wspace=0.30, hspace=0.25,
        left=0.08, right=0.92, top=0.93, bottom=0.07
    )

    # FIX 1: team names are passed explicitly — no more key lookup on event dicts
    teams      = {0: home_team, 1: away_team}
    leftright  = ['home', 'away']
    cols       = plt.cm.tab20b([0, 0.5, 1])
    pass_cols  = {
        'Ground Pass': cols[0],
        'High Pass':   cols[1],
        'Low Pass':    cols[2],
    }

    # Row labels
    for row, label in enumerate(['All Events', 'All Passes', 'Pitch Networks']):
        ax[(row, 0)].text(0.0, 0.5, label,
                          transform=ax[(row, 0)].transAxes,
                          rotation=90, va='center', ha='center',
                          color='.6', fontsize=11)

    # Direction arrows for all rows
    for row in range(3):
        ax[(row, 0)].annotate(
            'direction of play',
            xy=(0.85, -0.01), xytext=(0.15, -0.01),
            xycoords=ax[(row, 0)].transAxes,
            textcoords=ax[(row, 0)].transAxes,
            ha='left', va='center', fontsize=9, color='.5',
            arrowprops=dict(fc='.7', ec='.7', arrowstyle='-|>', lw=1.25)
        )
        ax[(row, 1)].annotate(
            'direction of play',
            xy=(0.15, -0.01), xytext=(0.85, -0.01),
            xycoords=ax[(row, 1)].transAxes,
            textcoords=ax[(row, 1)].transAxes,
            ha='right', va='center', fontsize=9, color='.5',
            arrowprops=dict(fc='.65', ec='.65', arrowstyle='-|>', lw=1.25)
        )

    df = pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def setup_pitch(ax_t):
        pitch = Pitch(
            pitch_type='statsbomb', pitch_color='none',
            line_color='white', linewidth=1,
            stripe=False, goal_type='box', line_zorder=10
        )
        pitch.draw(ax=ax_t)
        ax_t.set_aspect('auto')

    def grass_stripes(ax_t):
        for i in range(int(80 / 4) + 1):
            y0 = i * 4
            y1 = min(y0 + 4, 80)
            c  = '#4CAF50' if i % 2 == 0 else '#2E7D32'
            ax_t.add_patch(patches.Rectangle(
                (0, y0), 120, y1 - y0,
                linewidth=0, facecolor=c, alpha=0.15, zorder=1
            ))

    # ------------------------------------------------------------------
    # ROW 0 — All Events
    # ------------------------------------------------------------------
    EVENT_COLORS = {
        'Carry':          'mediumorchid',
        'Dribble':        'mediumorchid',
        'Foul Committed': '#D7664C',
        'Pass':           '#D6B66C',
        'Shot':           '#4C72B0',
    }

    for col, team in teams.items():
        setup_pitch(ax[(0, col)])
        grass_stripes(ax[(0, col)])

        team_df = df[
            df['team_name'].eq(team) &
            df['location'].notna() &
            df['location'].apply(lambda l: isinstance(l, list) and len(l) >= 2)
        ]

        buckets = {k: {'x': [], 'y': []} for k in list(EVENT_COLORS) + ['Other']}
        for _, ev in team_df.iterrows():
            x, y   = ev['location'][0], ev['location'][1]
            bucket = ev['type_name'] if ev['type_name'] in EVENT_COLORS else 'Other'
            buckets[bucket]['x'].append(x)
            buckets[bucket]['y'].append(y)

        # plot order: Other first (background), then the rest
        plot_order = ['Other', 'Pass', 'Carry', 'Dribble', 'Foul Committed', 'Shot']
        styles = {
            'Other':          dict(c='grey',          s=5,  alpha=0.35, linewidth=0.5, zorder=2, label='Other Events'),
            'Carry':          dict(c='mediumorchid',  s=10, alpha=1,    linewidth=0.9, zorder=5, label='Carry/Dribble'),
            'Dribble':        dict(c='mediumorchid',  s=10, alpha=1,    linewidth=0.9, zorder=5, label='_nolegend_'),
            'Foul Committed': dict(c='#D7664C',       s=10, alpha=1,    linewidth=0.9, zorder=5, label='Foul Committed'),
            'Pass':           dict(c='#D6B66C',       s=10, alpha=1,    linewidth=0.9, zorder=5, label='Pass'),
            'Shot':           dict(c='#4C72B0',       s=10, alpha=1,    linewidth=0.9, zorder=5, label='Shot'),
        }
        for key in plot_order:
            xs, ys = buckets[key]['x'], buckets[key]['y']
            if xs:
                ax[(0, col)].scatter(xs, ys, **styles[key])

        ax[(0, col)].set_title(team, fontsize=12, y=0.95)

    ax[(0, 1)].invert_xaxis()  # away team: flip X only

    # ------------------------------------------------------------------
    # ROW 1 — All Passes
    # ------------------------------------------------------------------
    pass_data = [i for i in data if i['type_name'] == 'Pass']

    for col, (ha, team) in enumerate(zip(leftright, [home_team, away_team])):
        setup_pitch(ax[(1, col)])
        grass_stripes(ax[(1, col)])

        # FIX 1 applied: use the explicit team name variable
        team_passes = [i for i in pass_data if i['possession_team_name'] == team]

        x_start = [i['location'][0]          for i in team_passes]
        y_start = [i['location'][1]          for i in team_passes]
        x_end   = [i['pass_end_location'][0] for i in team_passes]
        y_end   = [i['pass_end_location'][1] for i in team_passes]
        p_types = [i['pass_height_name']     for i in team_passes]
        body_parts = [
            i['pass_body_part_name'] if i['pass_body_part_name'] is not None else 'None'
            for i in team_passes
        ]

        conn_style_map = {
            'Left Foot':  'arc3,rad=-0.15',
            'Right Foot': 'arc3,rad=0.15',
        }

        # FIX 2: build conn_style and filter short passes together
        #         so indices never get misaligned
        filtered = [
            (xs, ys, xe, ye, pt, bp)
            for xs, ys, xe, ye, pt, bp
            in zip(x_start, y_start, x_end, y_end, p_types, body_parts)
            if (xe - xs)**2 + (ye - ys)**2 >= 10
        ]

        for xs, ys, xe, ye, pt, bp in filtered:
            cs    = conn_style_map.get(bp, 'arc3,rad=0')
            color = pass_cols.get(pt, pass_cols['Ground Pass'])
            arrow = FancyArrowPatch(
                posA=(xs, ys), posB=(xe, ye),
                arrowstyle='-|>', mutation_scale=6,
                color=color, lw=0.8, alpha=0.8,
                connectionstyle=cs
            )
            ax[(1, col)].add_patch(arrow)

        ax[(1, col)].set_title(team, fontsize=12, y=0.95)

        if ha == 'away':
            ax[(1, col)].invert_xaxis()
            # FIX 3: removed invert_yaxis() — StatsBomb doesn't need it

    # ------------------------------------------------------------------
    # ROW 2 — Network Plots
    # ------------------------------------------------------------------
    setup_pitch(ax[(2, 0)])
    setup_pitch(ax[(2, 1)])

    try:
        create_network_plot_for_subplot(ax[(2, 0)], 'result_home_network.csv', home_team)
        create_network_plot_for_subplot(ax[(2, 1)], 'result_away_network.csv', away_team)
        ax[(2, 1)].invert_xaxis()
        # FIX 3: removed invert_yaxis() here too
    except FileNotFoundError as e:
        for col, msg in enumerate(['Home', 'Away']):
            ax[(2, col)].text(0.5, 0.5, f'CSV not found\nfor {msg} Network',
                              ha='center', va='center',
                              transform=ax[(2, col)].transAxes)

    # ------------------------------------------------------------------
    # Final setup
    # ------------------------------------------------------------------
    for ai, a in enumerate(fig.axes):
        a.set_xticks([])
        a.set_yticks([])
        a.text(0.03, 0.97, f'({chr(65+ai)})',
               ha='left', va='bottom', transform=a.transAxes,
               fontweight='bold', fontsize=12)

    # Legend 1 — Events
    handles, labels = ax[(0, 0)].get_legend_handles_labels()
    if handles:
        order = ['Pass', 'Carry/Dribble', 'Shot', 'Foul Committed', 'Other Events']
        pairs = {l: h for h, l in zip(handles, labels)}
        oh = [pairs[l] for l in order if l in pairs]
        ol = [l        for l in order if l in pairs]
        fig.legend(oh, ol, loc='upper center', bbox_to_anchor=(0.5, 0.86),
                   ncol=1, fontsize=9, markerscale=1.5, columnspacing=0.8)

    # Legend 2 — Pass types
    pass_legend_elements = [
        Line2D([0], [0], color=c, linewidth=3, label=l)
        for l, c in pass_cols.items()
    ]
    fig.legend(handles=pass_legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.53), ncol=1, fontsize=9)

    # Legend 3 — Network nodes
    network_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='honeydew',
               markeredgecolor='red',  markeredgewidth=1, markersize=8,
               linestyle='None', label='Nodes with\nself-loops'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='honeydew',
               markeredgecolor='gold', markeredgewidth=1, markersize=8,
               linestyle='None', label='Nodes without\nself-loops'),
    ]
    fig.legend(handles=network_legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.23), ncol=1, fontsize=9)

    plt.show()
    return fig, ax


# ======================================================================
# Usage
# ======================================================================
# fig, ax = visualize_complete_analysis(data, home_team='Arsenal', away_team='Liverpool')
