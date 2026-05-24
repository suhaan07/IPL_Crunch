"""
fetch_standings.py — Scrape current IPL 2026 standings from Wikipedia.

Usage:
    python fetch_standings.py

Writes standings_2026.json in the same directory.
Run this any time you want to refresh standings before running ipl_crunch_full.py.
"""

import json
import re
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WIKI_URL = "https://en.wikipedia.org/wiki/2026_Indian_Premier_League"
OUT_PATH = Path("standings_2026.json")
HEADERS  = {"User-Agent": "Mozilla/5.0 (IPL-Crunch/1.0; research project)"}

# Normalize Wikipedia team names → names used in iplexcel.csv
NAME_MAP = {
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Royal Challengers Bangalore": "Royal Challengers Bangalore",
    "Gujarat Titans":              "Gujarat Titans",
    "Sunrisers Hyderabad":         "Sunrisers Hyderabad",
    "Punjab Kings":                "Punjab Kings",
    "Rajasthan Royals":            "Rajasthan Royals",
    "Kolkata Knight Riders":       "Kolkata Knight Riders",
    "Chennai Super Kings":         "Chennai Super Kings",
    "Delhi Capitals":              "Delhi Capitals",
    "Mumbai Indians":              "Mumbai Indians",
    "Lucknow Super Giants":        "Lucknow Super Giants",
}


def _clean(text: str) -> str:
    # Remove citation refs, normalize Unicode minus (U+2212) to ASCII hyphen
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("−", "-")
    return text.strip()


def fetch_standings() -> dict:
    print(f"Fetching {WIKI_URL} ...")
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    standings, nrr = {}, {}

    for table in soup.find_all("table", class_=re.compile("wikitable")):
        header_cells = table.find("tr").find_all(["th", "td"])
        col_names = [_clean(c.get_text()) for c in header_cells]

        if "Pts" not in col_names or "NRR" not in col_names:
            continue

        # Locate column indices (table may vary year to year)
        def idx(*names):
            for name in names:
                try:
                    return col_names.index(name)
                except ValueError:
                    pass
            return None

        i_team = idx("Team")
        i_m    = idx("Pld", "M", "Mat", "Matches")
        i_w    = idx("W", "Won")
        i_l    = idx("L", "Lost")
        i_nr   = idx("NR", "N/R", "No result")
        i_pts  = idx("Pts", "Points")
        i_nrr  = idx("NRR", "Net RR")

        if None in (i_team, i_m, i_w, i_pts):
            continue

        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            needed = max(filter(lambda x: x is not None, [i_team, i_m, i_w, i_pts]))
            if len(cells) < needed + 1:
                continue
            # Strip qualifiers like (Q), (E), (DR) from team name
            raw_name = re.sub(r"\s*\([^)]+\)", "", _clean(cells[i_team].get_text())).strip()
            team = NAME_MAP.get(raw_name, raw_name)
            if not team or team not in NAME_MAP.values():
                continue
            try:
                m   = int(_clean(cells[i_m].get_text()))
                w   = int(_clean(cells[i_w].get_text()))
                l   = int(_clean(cells[i_l].get_text())) if i_l is not None else 0
                nr  = int(_clean(cells[i_nr].get_text())) if i_nr is not None else 0
                pts = int(_clean(cells[i_pts].get_text()))
                nrr_val = float(_clean(cells[i_nrr].get_text())) if i_nrr is not None else 0.0
            except (ValueError, IndexError):
                continue
            standings[team] = {"M": m, "W": w, "L": l, "NR": nr, "Pts": pts}
            nrr[team]       = nrr_val

        if standings:
            break  # found the first valid points table

    if not standings:
        raise RuntimeError("Could not find a valid points table on the Wikipedia page.")

    return {"as_of": str(date.today()), "standings": standings, "nrr": nrr}


def main():
    data = fetch_standings()
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"\nSaved to {OUT_PATH}  (as of {data['as_of']})\n")
    print(f"{'Team':<35} {'M':>3} {'W':>3} {'L':>3} {'NR':>3} {'Pts':>4} {'NRR':>7}")
    print("-" * 62)
    for team, s in sorted(data["standings"].items(), key=lambda x: -x[1]["Pts"]):
        print(f"{team:<35} {s['M']:>3} {s['W']:>3} {s['L']:>3} {s['NR']:>3} "
              f"{s['Pts']:>4} {data['nrr'].get(team, 0):>+7.3f}")


if __name__ == "__main__":
    main()
