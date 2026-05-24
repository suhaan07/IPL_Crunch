"""
append_new_matches.py — Download new Cricsheet IPL JSON files and append to iplexcel.csv.

Usage:
    python append_new_matches.py

The script pulls the latest ipl_male_json.zip from Cricsheet, finds any match IDs
not already in iplexcel.csv, parses them, and appends new rows.
"""

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

CSV_PATH = Path("iplexcel.csv")
CRICSHEET_ZIP_URL = "https://cricsheet.org/downloads/ipl_male_json.zip"

EXTRAS_MAP = {
    "wides": "extras_wides",
    "noballs": "extras_noballs",
    "byes": "extras_byes",
    "legbyes": "extras_legbyes",
}

WICKET_KINDS = {
    "caught", "bowled", "lbw", "run out", "stumped",
    "caught and bowled", "hit wicket", "obstructing the field",
    "handled the ball", "timed out",
}


def _parse_match(info: dict, innings_data: list, match_id: int) -> list[dict]:
    rows = []
    date_str = str(info["dates"][0])
    season = str(info.get("season", date_str[:4]))
    venue = info.get("venue", "")
    city = info.get("city", "")
    teams = info["teams"]
    team1, team2 = teams[0], teams[1]
    toss = info.get("toss", {})
    toss_winner = toss.get("winner", "")
    toss_decision = toss.get("decision", "")
    outcome = info.get("outcome", {})
    if "winner" in outcome:
        winner = outcome["winner"]
        win_by_runs = outcome.get("by", {}).get("runs", 0)
        win_by_wickets = outcome.get("by", {}).get("wickets", 0)
    else:
        winner = ""
        win_by_runs = 0
        win_by_wickets = 0
    pom_list = info.get("player_of_match", [])
    player_of_match = pom_list[0] if pom_list else ""

    for inn_idx, inn in enumerate(innings_data):
        batting_team = inn.get("team", "")
        for over_obj in inn.get("overs", []):
            over_num = over_obj["over"]
            for ball_idx, delivery in enumerate(over_obj.get("deliveries", [])):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                is_wide = "wides" in extras
                is_noball = "noballs" in extras
                legal_ball = int(not is_wide and not is_noball)

                wicket_kind = ""
                wicket_player_out = ""
                if wickets:
                    w = wickets[0]
                    wicket_kind = w.get("kind", "")
                    wicket_player_out = w.get("player_out", "")

                row = {
                    "match_id": match_id,
                    "date": date_str,
                    "season": season,
                    "venue": venue,
                    "city": city,
                    "team1": team1,
                    "team2": team2,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "winner": winner,
                    "win_by_runs": win_by_runs,
                    "win_by_wickets": win_by_wickets,
                    "player_of_match": player_of_match,
                    "innings": inn_idx + 1,
                    "batting_team": batting_team,
                    "over": over_num,
                    "ball": ball_idx + 1,
                    "batter": delivery.get("batter", ""),
                    "bowler": delivery.get("bowler", ""),
                    "non_striker": delivery.get("non_striker", ""),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total": runs.get("total", 0),
                    "extras_wides": extras.get("wides", 0),
                    "extras_noballs": extras.get("noballs", 0),
                    "extras_byes": extras.get("byes", 0),
                    "extras_legbyes": extras.get("legbyes", 0),
                    "wicket_kind": wicket_kind,
                    "wicket_player_out": wicket_player_out,
                    "legal_ball": legal_ball,
                }
                rows.append(row)
    return rows


def main():
    print(f"Loading {CSV_PATH} ...")
    df_existing = pd.read_csv(CSV_PATH, low_memory=False, dtype={"season": str, "match_id": str})
    existing_ids = set(df_existing["match_id"].astype(str))
    print(f"  {len(df_existing):,} rows, {len(existing_ids)} unique match IDs")

    print(f"\nDownloading {CRICSHEET_ZIP_URL} ...")
    resp = requests.get(CRICSHEET_ZIP_URL, timeout=60)
    resp.raise_for_status()
    print(f"  Downloaded {len(resp.content):,} bytes")

    new_rows = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        json_files = [n for n in zf.namelist() if n.endswith(".json")]
        print(f"  {len(json_files)} JSON files in zip")

        for fname in sorted(json_files):
            match_id_str = Path(fname).stem
            if match_id_str in existing_ids:
                continue  # already in CSV
            try:
                with zf.open(fname) as f:
                    data = json.load(f)
                info = data.get("info", {})
                # Only IPL
                if info.get("event", {}).get("name", "").upper() != "INDIAN PREMIER LEAGUE":
                    continue
                innings_data = data.get("innings", [])
                rows = _parse_match(info, innings_data, int(match_id_str))
                new_rows.extend(rows)
                print(f"  + {match_id_str}: {info['teams'][0]} vs {info['teams'][1]} "
                      f"({info['dates'][0]}) — {len(rows)} deliveries")
            except Exception as e:
                print(f"  ! {fname}: {e}")

    if not new_rows:
        print("\nNo new matches found — CSV is already up to date.")
        return

    df_new = pd.DataFrame(new_rows)

    # Normalize RCB name
    for col in ["team1", "team2", "winner", "batting_team", "toss_winner"]:
        if col in df_new.columns:
            df_new[col] = df_new[col].str.replace(
                "Royal Challengers Bengaluru", "Royal Challengers Bangalore", regex=False
            )

    # Align columns to existing CSV
    df_new = df_new.reindex(columns=df_existing.columns)

    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(CSV_PATH, index=False)
    print(f"\nAppended {len(new_rows):,} rows ({len(df_new['match_id'].unique())} matches).")
    print(f"CSV now has {len(df_combined):,} rows.")


if __name__ == "__main__":
    main()
