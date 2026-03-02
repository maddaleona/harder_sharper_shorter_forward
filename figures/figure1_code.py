"""
Figure 1 – complete pipeline
  1. Load StatsBomb JSON  (or generate fake data for testing)
  2. Build pitch-passing networks → result_home_network.csv / result_away_network.csv
  3. Produce the 3×2 visualisation (All Events | All Passes | Pitch Networks)

Usage
-----
  python figure1_complete.py                        # uses built-in fake data
  python figure1_complete.py --json 3955720.json    # uses a real StatsBomb file
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mplsoccer import Pitch

# ──────────────────────────────────────────────────────────────
# PITCH GRID CONSTANTS
# ──────────────────────────────────────────────────────────────
PITCH_LENGTH  = 120
PITCH_WIDTH   = 80
RECT_LENGTH   = 12   # 10 columns
RECT_WIDTH    = 16   # 5 rows
N_COLS        = PITCH_LENGTH // RECT_LENGTH   # 10
N_ROWS        = PITCH_WIDTH  // RECT_WIDTH    # 5
N_NODES       = N_COLS * N_ROWS               # 50


def xy_to_node(x: float, y: float) -> int:
    """Map StatsBomb (x, y) coordinates to a grid node ID (0–49)."""
    col = int(np.clip(x, 0, PITCH_LENGTH - 1e-9) // RECT_LENGTH)
    row = int(np.clip(y, 0, PITCH_WIDTH  - 1e-9) // RECT_WIDTH)
    return col * N_ROWS + row


# ──────────────────────────────────────────────────────────────
# 1.  FAKE DATA GENERATOR
# ──────────────────────────────────────────────────────────────
def generate_fake_match(home: str = "Arsenal",
                        away: str = "Liverpool",
                        n_events: int = 2200,
                        seed: int = 42) -> list[dict]:
    """
    Return a list of dicts that mimics StatsBomb event structure,
    realistic enough for the visualisation pipeline.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    teams   = [home, away]
    heights = ["Ground Pass", "High Pass", "Low Pass"]
    bodies  = ["Right Foot", "Left Foot", "Head", None]
    types   = ["Pass", "Carry", "Dribble", "Shot", "Foul Committed",
               "Pressure", "Ball Receipt*", "Block", "Clearance"]
    type_w  = [0.35, 0.20, 0.05, 0.03, 0.04, 0.12, 0.10, 0.05, 0.06]

    events = []
    possession_team = home
    for idx in range(n_events):
        # flip possession occasionally
        if rng.random() < 0.08:
            possession_team = away if possession_team == home else home

        team = possession_team if rng.random() < 0.85 else (
            away if possession_team == home else home)

        etype = rng.choices(types, weights=type_w)[0]

        # realistic start location biased toward own half for the team
        x = float(np.clip(np.random.normal(60, 25), 1, 119))
        y = float(np.clip(np.random.normal(40, 18), 1, 79))

        ev = {
            "id":                   f"ev_{idx:05d}",
            "index":                idx,
            "type_name":            etype,
            "team_name":            team,
            "possession_team_name": possession_team,
            "home_team":            home,
            "away_team":            away,
            "location":             [x, y],
            "pass_end_location":    None,
            "pass_height_name":     None,
            "pass_body_part_name":  None,
            "under_pressure":       rng.random() < 0.15,
        }

        if etype == "Pass":
            dx = float(np.random.normal(15, 10))
            dy = float(np.random.normal(0,  8))
            ex = float(np.clip(x + dx, 0, 120))
            ey = float(np.clip(y + dy, 0, 80))
            ev["pass_end_location"]   = [ex, ey]
            ev["pass_height_name"]    = rng.choices(heights, [0.75, 0.12, 0.13])[0]
            ev["pass_body_part_name"] = rng.choice(bodies)

        events.append(ev)

    return events


# ──────────────────────────────────────────────────────────────
# 2.  NETWORK BUILDER
# ──────────────────────────────────────────────────────────────
def build_network(passes: list[dict]) -> pd.DataFrame:
    """
    Given a list of Pass events for ONE team, return a DataFrame
    with columns [source, target, weight] suitable for the network plot.
    Weights are normalised so all edge weights sum to 100.
    """
    counts: dict[tuple[int, int], float] = defaultdict(float)

    for ev in passes:
        loc = ev.get("location")
        end = ev.get("pass_end_location")
        if not (loc and end and len(loc) >= 2 and len(end) >= 2):
            continue
        src = xy_to_node(loc[0], loc[1])
        tgt = xy_to_node(end[0], end[1])
        counts[(src, tgt)] += 1.0

    if not counts:
        return pd.DataFrame(columns=["source", "target", "weight"])

    total = sum(counts.values())
    rows  = [{"source": s, "target": t, "weight": round(w / total * 100, 6)}
             for (s, t), w in counts.items()]
    return pd.DataFrame(rows)


def build_and_save_networks(data: list[dict],
                            home: str,
                            away: str,
                            home_csv: str = "result_home_network.csv",
                            away_csv: str = "result_away_network.csv") -> None:
    passes = [e for e in data if e.get("type_name") == "Pass"]

    home_passes = [e for e in passes if e.get("possession_team_name") == home]
    away_passes = [e for e in passes if e.get("possession_team_name") == away]

    build_network(home_passes).to_csv(home_csv, index=False)
    build_network(away_passes).to_csv(away_csv, index=False)
    print(f"Saved {home_csv}  ({len(home_passes)} passes)")
    print(f"Saved {away_csv}  ({len(away_passes)} passes)")


# ──────────────────────────────────────────────────────────────
# 3.  VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────
def _setup_pitch(ax):
    pitch = Pitch(pitch_type="statsbomb", pitch_color="none",
                  line_color="white", linewidth=1,
                  stripe=False, goal_type="box", line_zorder=10)
    pitch.draw(ax=ax)
    ax.set_aspect("auto")


def _grass_stripes(ax):
    for i in range(N_ROWS * 5 + 1):
        y0 = i * 4
        y1 = min(y0 + 4, PITCH_WIDTH)
        c  = "#4CAF50" if i % 2 == 0 else "#2E7D32"
        ax.add_patch(patches.Rectangle(
            (0, y0), PITCH_LENGTH, y1 - y0,
            linewidth=0, facecolor=c, alpha=0.15, zorder=1))


def _grid_lines(ax):
    for i in range(1, N_COLS):
        ax.plot([i * RECT_LENGTH] * 2, [0, PITCH_WIDTH],
                color="white", lw=1, alpha=0.85, ls="--")
    for i in range(1, N_ROWS):
        ax.plot([0, PITCH_LENGTH], [i * RECT_WIDTH] * 2,
                color="white", lw=1, alpha=0.85, ls="--")


def _node_positions() -> dict[int, tuple[float, float]]:
    pos = {}
    for nid in range(N_NODES):
        col = nid // N_ROWS
        row = nid  % N_ROWS
        pos[nid] = (col * RECT_LENGTH + RECT_LENGTH / 2,
                    row * RECT_WIDTH  + RECT_WIDTH  / 2)
    return pos


def _direction_arrow(ax, home: bool):
    if home:
        ax.annotate("direction of play",
                    xy=(0.85, -0.01), xytext=(0.15, -0.01),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    ha="left", va="center", fontsize=9, color=".5",
                    arrowprops=dict(fc=".7", ec=".7", arrowstyle="-|>", lw=1.25))
    else:
        ax.annotate("direction of play",
                    xy=(0.15, -0.01), xytext=(0.85, -0.01),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    ha="right", va="center", fontsize=9, color=".5",
                    arrowprops=dict(fc=".65", ec=".65", arrowstyle="-|>", lw=1.25))


# ──────────────────────────────────────────────────────────────
# 4.  ROW PLOTTERS
# ──────────────────────────────────────────────────────────────
_EVENT_STYLES = {
    "Pass":           dict(c="#D6B66C", s=10, alpha=1,    lw=0.9, zorder=5, label="Pass"),
    "Carry":          dict(c="mediumorchid", s=10, alpha=1, lw=0.9, zorder=5, label="Carry/Dribble"),
    "Dribble":        dict(c="mediumorchid", s=10, alpha=1, lw=0.9, zorder=5, label="_nolegend_"),
    "Foul Committed": dict(c="#D7664C", s=10, alpha=1,    lw=0.9, zorder=5, label="Foul Committed"),
    "Shot":           dict(c="#4C72B0", s=10, alpha=1,    lw=0.9, zorder=5, label="Shot"),
    "_other_":        dict(c="grey",    s=5,  alpha=0.35, lw=0.5, zorder=2, label="Other Events"),
}

def _plot_events(ax, df_team):
    buckets = {k: {"x": [], "y": []} for k in list(_EVENT_STYLES)}
    for _, ev in df_team.iterrows():
        loc = ev["location"]
        if loc is None or len(loc) < 2:
            continue
        key = ev["type_name"] if ev["type_name"] in _EVENT_STYLES else "_other_"
        buckets[key]["x"].append(loc[0])
        buckets[key]["y"].append(loc[1])

    order = ["_other_", "Pass", "Carry", "Dribble", "Foul Committed", "Shot"]
    for key in order:
        xs, ys = buckets[key]["x"], buckets[key]["y"]
        if not xs:
            continue
        st = _EVENT_STYLES[key]
        ax.scatter(xs, ys, c=st["c"], s=st["s"], alpha=st["alpha"],
                   linewidth=st["lw"], zorder=st["zorder"], label=st["label"])


def _plot_passes(ax, team_passes, pass_cols):
    heights  = [e["pass_height_name"]    for e in team_passes]
    bodies   = [e["pass_body_part_name"] for e in team_passes]
    starts   = [e["location"]            for e in team_passes]
    ends     = [e["pass_end_location"]   for e in team_passes]

    conn_map = {"Left Foot": "arc3,rad=-0.15", "Right Foot": "arc3,rad=0.15"}
    for (xs, ys), (xe, ye), ht, bp in zip(starts, ends, heights, bodies):
        if (xe - xs) ** 2 + (ye - ys) ** 2 < 10:
            continue
        cs    = conn_map.get(bp, "arc3,rad=0")
        color = pass_cols.get(ht, pass_cols["Ground Pass"])
        ax.add_patch(FancyArrowPatch(
            posA=(xs, ys), posB=(xe, ye),
            arrowstyle="-|>", mutation_scale=6,
            color=color, lw=0.8, alpha=0.8, connectionstyle=cs))


def _plot_network(ax, csv_path: str, title: str):
    df = pd.read_csv(csv_path)
    pos = _node_positions()

    G = nx.Graph()
    for _, row in df.iterrows():
        s, t, w = int(row["source"]), int(row["target"]), float(row["weight"])
        if 0 <= s < N_NODES and 0 <= t < N_NODES:
            G.add_edge(s, t, weight=w)

    # eigenvector centrality → node size
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception:
        deg    = dict(G.degree())
        maxd   = max(deg.values()) if deg else 1
        ec     = {n: deg[n] / maxd for n in G.nodes()}

    ec_vals = list(ec.values()) or [0, 1]
    ec_min, ec_max = min(ec_vals), max(ec_vals)

    sizes = {}
    for nid in range(N_NODES):
        if nid in G.nodes():
            norm = (ec.get(nid, 0) - ec_min) / (ec_max - ec_min + 1e-12)
            sizes[nid] = 50 + norm ** 2 * 350
        else:
            sizes[nid] = 80

    # edges
    if G.edges():
        wts = [d["weight"] for _, _, d in G.edges(data=True)]
        wmin, wmax = min(wts), max(wts)

        regular   = [(u, v, d) for u, v, d in G.edges(data=True) if u != v]
        selfloops = [(u, v, d) for u, v, d in G.edges(data=True) if u == v]

        for u, v, d in regular:
            x1, y1 = pos[u];  x2, y2 = pos[v]
            norm = (d["weight"] - wmin) / (wmax - wmin + 1e-12)
            lw   = 0.2 + norm * 2.0
            gi   = 0.6 - norm * 0.6
            mx, my   = (x1+x2)/2, (y1+y2)/2
            dist     = np.hypot(x2-x1, y2-y1)
            cdir     = 1 if (u+v) % 2 == 0 else -1
            offset   = cdir * (dist * 0.1 + 2)
            if x2 != x1:
                px = -(y2-y1); py = x2-x1
                pl = np.hypot(px, py)
                px, py = px/pl*offset, py/pl*offset
            else:
                px, py = offset, 0
            t  = np.linspace(0, 1, 50)
            cx = (1-t)**2*x1 + 2*(1-t)*t*(mx+px) + t**2*x2
            cy = (1-t)**2*y1 + 2*(1-t)*t*(my+py) + t**2*y2
            ax.plot(cx, cy, color=(gi,gi,gi), lw=lw, alpha=0.8, zorder=5,
                    solid_capstyle="round")

        for nid, _, d in selfloops:
            x, y  = pos[nid]
            norm  = (d["weight"] - wmin) / (wmax - wmin + 1e-12)
            ls    = 3 + norm * 8
            lw    = 0.2 + norm * 2.0
            gi    = 0.6 - norm * 0.6
            t     = np.linspace(0, 1, 100)
            lx = (1-t)**3*(x+2) + 3*(1-t)**2*t*(x+ls) + 3*(1-t)*t**2*(x-ls) + t**3*(x-2)
            ly = (1-t)**3*(y+2) + 3*(1-t)**2*t*(y+ls) + 3*(1-t)*t**2*(y+ls) + t**3*(y+2)
            ax.plot(lx, ly, color=(gi,gi,gi), lw=lw, alpha=0.8, zorder=6,
                    solid_capstyle="round")

    # nodes
    for nid, (x, y) in pos.items():
        isolated = nid not in G.nodes()
        nc       = "limegreen" if isolated else "honeydew"
        has_loop = any(u == v == nid for u, v, _ in G.edges(data=True))
        ec_col   = "white" if isolated else ("red" if has_loop else "gold")
        ax.scatter(x, y, s=sizes[nid], c=nc, alpha=1,
                   edgecolors=ec_col, linewidth=1, zorder=10)

    ax.set_title(title, fontsize=12, y=0.95)


# ──────────────────────────────────────────────────────────────
# 5.  MAIN FIGURE
# ──────────────────────────────────────────────────────────────
def visualize_complete(data: list[dict],
                       home: str,
                       away: str,
                       home_csv: str = "result_home_network.csv",
                       away_csv:  str = "result_away_network.csv",
                       save_path: str | None = None):

    cols      = plt.cm.tab20b([0, 0.5, 1])
    pass_cols = {"Ground Pass": cols[0], "High Pass": cols[1], "Low Pass": cols[2]}

    aspect = 2.5 / 2
    fig, ax = plt.subplots(3, 2, figsize=(aspect * 10, aspect * 8.5), dpi=300)
    plt.subplots_adjust(wspace=0.30, hspace=0.25,
                        left=0.08, right=0.92, top=0.93, bottom=0.07)

    # row labels
    for row, lbl in enumerate(["All Events", "All Passes", "Pitch Networks"]):
        ax[row, 0].text(0.0, 0.5, lbl, transform=ax[row, 0].transAxes,
                        rotation=90, va="center", ha="center",
                        color=".6", fontsize=11)

    # direction arrows
    for row in range(3):
        _direction_arrow(ax[row, 0], home=True)
        _direction_arrow(ax[row, 1], home=False)

    df = pd.DataFrame(data)

    # ---- row 0: all events ----
    teams = {0: home, 1: away}
    for col, team in teams.items():
        _setup_pitch(ax[0, col])
        _grass_stripes(ax[0, col])
        tdf = df[df["team_name"] == team]
        _plot_events(ax[0, col], tdf)
        ax[0, col].set_title(team, fontsize=12, y=0.95)
    ax[0, 1].invert_xaxis()

    # ---- row 1: passes ----
    passes = [e for e in data if e["type_name"] == "Pass"]
    for col, (team, invert) in enumerate([(home, False), (away, True)]):
        _setup_pitch(ax[1, col])
        _grass_stripes(ax[1, col])
        team_passes = [e for e in passes
                       if e["possession_team_name"] == team
                       and e["pass_end_location"] is not None]
        _plot_passes(ax[1, col], team_passes, pass_cols)
        ax[1, col].set_title(team, fontsize=12, y=0.95)
        if invert:
            ax[1, col].invert_xaxis()

    # ---- row 2: networks ----
    for col, (csv_path, team) in enumerate([(home_csv, home), (away_csv, away)]):
        _setup_pitch(ax[2, col])
        _grass_stripes(ax[2, col])
        _grid_lines(ax[2, col])
        try:
            _plot_network(ax[2, col], csv_path, team)
        except FileNotFoundError:
            ax[2, col].text(0.5, 0.5, f"CSV not found:\n{csv_path}",
                            ha="center", va="center",
                            transform=ax[2, col].transAxes)
    ax[2, 1].invert_xaxis()

    # labels (A)-(F)
    for i, a in enumerate(fig.axes):
        a.set_xticks([]); a.set_yticks([])
        a.text(0.03, 0.97, f"({chr(65+i)})", ha="left", va="bottom",
               transform=a.transAxes, fontweight="bold", fontsize=12)

    # legends
    handles, labels = ax[0, 0].get_legend_handles_labels()
    order   = ["Pass", "Carry/Dribble", "Shot", "Foul Committed", "Other Events"]
    pairs   = {l: h for h, l in zip(handles, labels)}
    fig.legend([pairs[l] for l in order if l in pairs],
               [l for l in order if l in pairs],
               loc="upper center", bbox_to_anchor=(0.5, 0.86),
               ncol=1, fontsize=9, markerscale=1.5, columnspacing=0.8)

    fig.legend(handles=[Line2D([0],[0], color=c, lw=3, label=l)
                        for l, c in pass_cols.items()],
               loc="upper center", bbox_to_anchor=(0.5, 0.53),
               ncol=1, fontsize=9)

    fig.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="honeydew",
               markeredgecolor="red",  markersize=8, linestyle="None",
               label="Nodes with\nself-loops"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="honeydew",
               markeredgecolor="gold", markersize=8, linestyle="None",
               label="Nodes without\nself-loops"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.23), ncol=1, fontsize=9)

    plt.show()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight",
                    facecolor="white", pad_inches=0.1)
        print(f"Saved figure → {save_path}")
    return fig, ax


# ──────────────────────────────────────────────────────────────
# 6.  ENTRY POINT
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",  default=None, help="Path to StatsBomb JSON file")
    parser.add_argument("--home",  default="Arsenal")
    parser.add_argument("--away",  default="Liverpool")
    parser.add_argument("--save",  default=None, help="Output image path (optional)")
    args = parser.parse_args()

    if args.json:
        path = Path(args.json)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} events from {path}")
        # infer team names from data if not overridden
        home = args.home
        away = args.away
    else:
        print("No JSON provided – generating fake match data …")
        home = args.home
        away = args.away
        data = generate_fake_match(home=home, away=away)
        print(f"Generated {len(data)} fake events ({home} vs {away})")

    # build and save networks
    build_and_save_networks(data, home, away)

    # plot
    visualize_complete(data, home, away, save_path=args.save)


if __name__ == "__main__":
    main()
