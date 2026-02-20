##############################################################
# KPIs
##############################################################

import numpy as np
import pandas as pd
import ast


# === HELPERS ===

def _extract_xy(val):
    """Parse (x, y) from list, tuple, or string representation."""
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    if isinstance(val, str):
        try:
            arr = ast.literal_eval(val)
            if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                return float(arr[0]), float(arr[1])
        except (ValueError, SyntaxError):
            pass
    return np.nan, np.nan


def _team_passes(df, team, successful_only=False):
    """Return pass rows for a given team, optionally only successful ones."""
    mask = (df['type_name'] == 'Pass') & (df['possession_team_name'] == team)
    if successful_only:
        mask &= df['pass_outcome_name'].isna()
    return df[mask]


# === KPI FUNCTIONS ===

def compute_pass_accuracy(df, team):
    """Percentage of attempted passes that were successfully completed."""
    passes = df[(df['type_name'] == 'Pass') & (df['possession_team_name'] == team)]
    if passes.empty:
        return np.nan
    return 100 * passes['pass_outcome_name'].isna().sum() / len(passes)


def compute_ground_pass_accuracy(df, team):
    """Accuracy of ground-level passes only."""
    passes = df[
        (df['type_name'] == 'Pass') &
        (df['possession_team_name'] == team) &
        (df['pass_height_name'] == 'Ground Pass')
    ]
    if passes.empty:
        return np.nan
    return 100 * passes['pass_outcome_name'].isna().sum() / len(passes)


def compute_passes_under_pressure(df, team):
    """Number of completed passes made while passer was under defensive pressure."""
    passes = _team_passes(df, team, successful_only=True)
    if 'under_pressure' not in passes.columns:
        return np.nan
    return int(passes['under_pressure'].sum())


def compute_passes_per_possession(df, team):
    """Average number of completed passes per possession phase."""
    all_poss = df.groupby('possession')['possession_team_name'].first()
    pass_counts = df[df['type_name'] == 'Pass'].groupby('possession').size()
    poss_df = pd.DataFrame({'team': all_poss, 'pass_count': pass_counts}).fillna(0)
    team_poss = poss_df[poss_df['team'] == team]['pass_count']
    return float(team_poss.mean()) if not team_poss.empty else np.nan


def compute_passes_before_shot(df, team):
    """Average passes in sequences leading to a shot attempt."""
    timestamps = pd.to_datetime(df['timestamp'])
    periods = df['period']
    adj_ts = timestamps + pd.to_timedelta((periods - 1) * 45, unit='m')
    tmp = df[['possession', 'type_name', 'possession_team_name']].copy()
    tmp['adj_ts'] = adj_ts
    shot_poss = tmp[tmp['type_name'] == 'Shot']['possession'].unique()
    shot_df = tmp[tmp['possession'].isin(shot_poss)]
    results = []
    for pid, grp in shot_df.groupby('possession'):
        grp = grp.sort_values('adj_ts')
        first_shot = grp[grp['type_name'] == 'Shot']['adj_ts'].iloc[0]
        n = grp[(grp['type_name'] == 'Pass') & (grp['adj_ts'] < first_shot)].shape[0]
        results.append({'team': grp['possession_team_name'].iloc[0], 'pbs': n})
    if not results:
        return np.nan
    pbs_df = pd.DataFrame(results)
    team_pbs = pbs_df[pbs_df['team'] == team]['pbs']
    return float(team_pbs.mean()) if not team_pbs.empty else np.nan


def compute_shot_distance(df, team):
    """Average Euclidean distance from shot location to goal center."""
    shots = df[(df['type_name'] == 'Shot') & (df['team_name'] == team)].copy()
    if shots.empty or 'shot_end_location' not in shots.columns:
        return np.nan
    shots['sx'] = shots['location'].apply(lambda l: l[0] if isinstance(l, list) else np.nan)
    shots['sy'] = shots['location'].apply(lambda l: l[1] if isinstance(l, list) else np.nan)
    shots['ex'] = shots['shot_end_location'].apply(lambda l: l[0] if isinstance(l, list) else np.nan)
    shots['ey'] = shots['shot_end_location'].apply(lambda l: l[1] if isinstance(l, list) else np.nan)
    dist = np.sqrt((shots['ex'] - shots['sx'])**2 + (shots['ey'] - shots['sy'])**2)
    return float(dist.mean())


def compute_vertical_play(df, team):
    """Ratio of forward (vertical) to lateral (horizontal) passes."""
    passes = _team_passes(df, team)
    if passes.empty:
        return np.nan
    start_xy = passes['location'].apply(_extract_xy)
    end_xy = passes['pass_end_location'].apply(_extract_xy)
    p = passes.copy()
    p[['sx', 'sy']] = pd.DataFrame(start_xy.tolist(), index=p.index)
    p[['ex', 'ey']] = pd.DataFrame(end_xy.tolist(), index=p.index)
    p = p.dropna(subset=['sx', 'sy', 'ex', 'ey'])
    dx = (p['ex'] - p['sx']).abs()
    dy = (p['ey'] - p['sy']).abs()
    n_vertical   = (dy >= dx).sum()
    n_horizontal = (dx > dy).sum()
    return float(n_vertical / n_horizontal) if n_horizontal > 0 else np.nan


def compute_pass_center_of_mass(df, team):
    """Mean x-coordinate of successful pass origins (attacking depth proxy)."""
    passes = _team_passes(df, team, successful_only=True)
    if passes.empty:
        return np.nan
    x_vals = passes['location'].apply(lambda l: l[0] if isinstance(l, list) else np.nan)
    return float(x_vals.mean())


def compute_offsides(df, team):
    """Total offside infractions committed by the team."""
    offside_passes = df[(df['type_name'] == 'Pass') &
                        (df['pass_outcome_name'] == 'Pass Offside') &
                        (df['possession_team_name'] == team)]
    offside_events = df[(df['type_name'] == 'Offside') &
                        (df['possession_team_name'] == team)]
    return len(offside_passes) + len(offside_events)


def compute_throwin_length(df, team):
    """Average throw-in length in meters."""
    YARD_TO_M = 0.9144
    throwins = df[(df['type_name'] == 'Pass') &
                  (df['pass_type_name'] == 'Throw-in') &
                  (df['possession_team_name'] == team)].copy()
    if throwins.empty:
        return np.nan
    return float((throwins['pass_length'] * YARD_TO_M).mean())


def compute_all_kpis(df, team):
    """
    Compute all KPIs reported in the paper for a given team in a match.

    Parameters
    ----------
    df : pd.DataFrame
        Match event data.
    team : str
        Team name.

    Returns
    -------
    dict of KPI values.
    """
    return {
        'pass_accuracy':          compute_pass_accuracy(df, team),
        'ground_pass_accuracy':   compute_ground_pass_accuracy(df, team),
        'passes_under_pressure':  compute_passes_under_pressure(df, team),
        'passes_per_possession':  compute_passes_per_possession(df, team),
        'passes_before_shot':     compute_passes_before_shot(df, team),
        'shot_distance':          compute_shot_distance(df, team),
        'vertical_play':          compute_vertical_play(df, team),
        'pass_center_of_mass_x':  compute_pass_center_of_mass(df, team),
        'offsides':               compute_offsides(df, team),
        'throwin_length_m':       compute_throwin_length(df, team),
    }
