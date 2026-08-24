"""diagnose_combos.py — why don't winner (KXWCGAME) legs show up in +EV combos?"""
import collections
import statistics as st

from kalshi_client import KalshiClient
from ratings import load_database
from price_combos import fetch_all_combos, score_live_combos, _fp

c = KalshiClient(); db = load_database()
combos = fetch_all_combos(c)
liquid = [x for x in combos if _fp(x.get("yes_ask_dollars")) > 0 and x.get("mve_selected_legs")]
WCMAP = {"KXWCGAME", "KXWCTOTAL", "KXWCSPREAD", "KXWCBTTS"}


def series_of(x):
    return [l["market_ticker"].split("-")[0] for l in x["mve_selected_legs"]]


game_liquid = game_mappable = 0
block = collections.Counter()
for x in liquid:
    s = series_of(x)
    if "KXWCGAME" in s:
        game_liquid += 1
        if all(p in WCMAP for p in s):
            game_mappable += 1
        else:
            for p in s:
                if p not in WCMAP:
                    block[p] += 1

print(f"liquid combos: {len(liquid)}")
print(f"  with a winner (KXWCGAME) leg: {game_liquid}")
print(f"  of those FULLY MAPPABLE (priceable): {game_mappable}")
print(f"  series blocking the rest: {dict(block.most_common(8))}\n")

results, _, _ = score_live_combos(c, db, combos)

def has_game(cc):
    return any(l["market_ticker"].startswith("KXWCGAME") for l in cc["mve_selected_legs"])

wg = [(e, j, p, cc) for (e, j, p, f, cc) in results if has_game(cc)]
ng = [e for (e, j, p, f, cc) in results if not has_game(cc)]
print(f"priced combos: {len(results)} | WITH winner leg: {len(wg)} | WITHOUT: {len(ng)}")
if wg:
    print(f"  mean edge WITH winner leg:    {100*st.mean(e for e, *_ in wg):+.2f}%")
if ng:
    print(f"  mean edge WITHOUT winner leg: {100*st.mean(ng):+.2f}%")
print("\n  top winner-leg combos by edge (model joint vs Kalshi price):")
for e, j, p, cc in sorted(wg, key=lambda t: -t[0])[:6]:
    print(f"   edge {e*100:+5.1f}%  joint {j*100:4.1f}%  price {p*100:4.1f}c  | "
          f"{cc.get('title','')[:52]}")
print("\n  WORST winner-leg combos by edge (what drags the mean down):")
for e, j, p, cc in sorted(wg, key=lambda t: t[0])[:6]:
    nlegs = len(cc["mve_selected_legs"])
    print(f"   edge {e*100:+7.1f}%  joint {j*100:5.2f}%  price {p*100:5.1f}c  "
          f"legs={nlegs}  | {cc.get('title','')[:44]}")
