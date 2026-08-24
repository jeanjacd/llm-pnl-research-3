"""
optimizer.py
============
Finds the best parlay/combo given:
  - a minimum joint hit probability  (confidence floor, e.g. 0.55)
  - a minimum payout multiplier       (e.g. 3.0x)

Advanced bits:
  1. CORRELATION-AWARE JOINT PROBABILITY. Legs from the same match are evaluated by
     AND-ing their per-iteration boolean outcome arrays from the Monte Carlo (exact
     joint), NOT by multiplying marginals. Legs across different matches are treated
     as independent and multiplied. This is the single biggest accuracy win vs a
     standard sportsbook parlay calc.
  2. POSITIVE-EDGE FILTER. A leg only enters the candidate pool if our model
     probability exceeds Kalshi's price-implied probability (edge > threshold).
  3. HALF-KELLY STAKING on the resulting parlay, treated as one compound bet.
  4. EXHAUSTIVE SEARCH over leg combinations (workload is not a constraint here).
"""

from __future__ import annotations
import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from simulate import MarketBook, wilson_interval


def kalshi_fee_cents(price_cents: float) -> float:
    """
    Kalshi trading fee per contract, in cents: 0.07 * P * (1 - P) where P is the
    price in dollars. ~1.75c on a 50c contract (3.5% of stake), more on longshots.
    Charged at execution, so it raises your effective cost.
    """
    p = price_cents / 100.0
    return 0.07 * p * (1.0 - p) * 100.0


@dataclass
class Leg:
    match_key: str            # identifies which simulated match this came from
    book: MarketBook          # the simulation it belongs to (for correlation)
    market_name: str          # market within that book
    model_prob: float         # our simulated probability
    kalshi_ticker: str
    price_cents: int          # quoted cost to enter (YES or NO ask), BEFORE fees
    decimal_odds: float       # net payout multiplier for this leg (after fees)
    side: str = "yes"
    fair_prob: float | None = None   # blended (model<->market) estimate; None => model
    fee_cents: float = 0.0    # Kalshi fee added to effective cost

    @property
    def effective_cost(self) -> float:
        """Price + fee, in cents -- what the bet really costs you."""
        return self.price_cents + self.fee_cents

    @property
    def implied_prob(self) -> float:
        """Break-even probability INCLUDING fees (effective cost / payout)."""
        return self.effective_cost / 100.0

    @property
    def estimate(self) -> float:
        """Our best probability estimate for this leg (blended if set)."""
        return self.fair_prob if self.fair_prob is not None else self.model_prob

    @property
    def edge(self) -> float:
        return self.estimate - self.implied_prob


@dataclass
class Parlay:
    legs: list[Leg]
    joint_prob: float
    ci_low: float
    ci_high: float
    multiplier: float         # product of decimal odds
    ev: float                 # expected value per unit staked (e.g. 0.18 = +18%)
    kelly_fraction: float
    half_kelly: float

    def describe(self) -> str:
        label = "single bet" if len(self.legs) == 1 else f"{len(self.legs)}-leg parlay"
        head = (f"{label} | hit {self.joint_prob*100:.1f}% "
                f"[{self.ci_low*100:.1f},{self.ci_high*100:.1f}] | "
                f"{self.multiplier:.2f}x | EV {self.ev*100:+.1f}% | "
                f"half-Kelly {self.half_kelly*100:.1f}% of bankroll")
        lines = [head]
        for lg in self.legs:
            lines.append(f"    - [{lg.match_key}] {lg.market_name} "
                         f"({lg.side.upper()} @ {lg.price_cents}c, odds {lg.decimal_odds:.2f}, "
                         f"model {lg.model_prob*100:.0f}% vs market {lg.implied_prob*100:.0f}%, "
                         f"edge {lg.edge*100:+.1f}%)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Joint probability (correlation-aware)
# ---------------------------------------------------------------------------
def parlay_joint_prob(legs: list[Leg]) -> tuple[float, float, float]:
    """
    Returns (joint_prob, ci_low, ci_high).
    Legs grouped by their simulation `book` are combined exactly via boolean AND
    (captures in-match correlation). Independent groups are multiplied.
    """
    groups: dict[int, list[Leg]] = {}
    for lg in legs:
        groups.setdefault(id(lg.book), []).append(lg)

    prob = 1.0
    # For the CI we approximate by combining the binomial variance of each group.
    var_terms = []
    n_ref = None
    for group in groups.values():
        book = group[0].book
        n_ref = book.n
        mask = np.ones(book.n, dtype=bool)
        for lg in group:
            mask &= book.outcomes[lg.market_name]
        p_group = float(mask.mean())
        prob *= p_group
        var_terms.append(p_group)

    if n_ref is None:
        return 0.0, 0.0, 0.0
    # Wilson CI on the final product treated as a proportion at the reference N.
    lo, hi = wilson_interval(prob * n_ref, n_ref)
    return prob, lo, hi


def kelly_fraction(p: float, multiplier: float) -> float:
    """Kelly fraction for a single compound bet with win prob p and decimal odds `multiplier`."""
    b = multiplier - 1.0
    if b <= 0:
        return 0.0
    f = (p * b - (1 - p)) / b
    return max(0.0, f)


def evaluate_parlay(legs: list[Leg]) -> Parlay:
    p_model, lo, hi = parlay_joint_prob(legs)
    # Shrink the joint toward the market in proportion to each leg's blend, preserving
    # the simulation's correlation structure. shrink = prod(fair_i / model_i).
    shrink = 1.0
    for lg in legs:
        if lg.model_prob > 0:
            shrink *= lg.estimate / lg.model_prob
    p = min(max(p_model * shrink, 0.0), 1.0)
    lo, hi = min(lo * shrink, 1.0), min(hi * shrink, 1.0)
    mult = math.prod(lg.decimal_odds for lg in legs)
    ev = p * mult - 1.0
    kf = kelly_fraction(p, mult)
    return Parlay(legs=list(legs), joint_prob=p, ci_low=lo, ci_high=hi,
                  multiplier=mult, ev=ev, kelly_fraction=kf, half_kelly=kf / 2.0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def build_candidate_legs(
    books_and_prices: list[tuple[str, MarketBook, dict[str, dict]]],
    min_edge: float = 0.02,
    model_trust: float = 1.0,
    min_leg_prob: float = 0.0,
) -> list[Leg]:
    """
    books_and_prices: list of (match_key, MarketBook, prices) where `prices` maps
        market_name -> {"ticker", "price_cents", "decimal_odds", "side"}.

    model_trust (0..1): blends our model with the market's implied probability:
        fair = model_trust * model + (1 - model_trust) * implied
    1.0 = pure model (original behavior); lower = trust the market more (conservative).
    Edge and EV are computed from `fair`, so we stop betting on model overconfidence.

    min_leg_prob: sanity floor — drop legs whose model probability is below this, since
        edge estimates on very rare outcomes are the least trustworthy (and highest
        variance). ~0.05 is a reasonable default.

    Only markets with a Kalshi price, model_prob >= min_leg_prob, AND edge >= min_edge.
    """
    legs: list[Leg] = []
    for match_key, book, prices in books_and_prices:
        for name, m in [(nm, book.market(nm)) for nm in book.names()]:
            quote = prices.get(name)
            if not quote:
                continue
            if m.prob < min_leg_prob:
                continue  # too rare to price reliably
            price = int(quote["price_cents"])
            fee = kalshi_fee_cents(price)
            effective = price + fee
            implied = effective / 100.0           # break-even prob incl. fees
            fair = model_trust * m.prob + (1.0 - model_trust) * (price / 100.0)
            leg = Leg(
                match_key=match_key,
                book=book,
                market_name=name,
                model_prob=m.prob,
                kalshi_ticker=quote["ticker"],
                price_cents=price,
                decimal_odds=100.0 / effective,   # net of fees
                side=quote.get("side", "yes"),
                fair_prob=fair,
                fee_cents=fee,
            )
            if leg.edge >= min_edge:
                legs.append(leg)
    return legs


def _category(name: str) -> str:
    """Classify a market so we can balance categories across the recommendations."""
    n = name.lower()
    if n.endswith("to score"):
        return "scorer"
    if name.startswith("Score "):
        return "correct_score"
    if "wins by" in n:
        return "spread"
    if n.startswith("both teams"):
        return "btts"
    if n.startswith(("over", "under")):
        return "total"
    if " over " in n or "clean sheet" in n:
        return "team_total"
    if n.endswith("win") or n == "draw" or "win or draw" in n or "either team" in n:
        return "result"
    return "other"


def _select_pool(legs: list[Leg], cap: int) -> list[Leg]:
    """
    Build the candidate pool round-robin across categories (each ordered by edge),
    so niche markets (scorers, BTTS, correct score) are represented and not crowded
    out by whichever category happens to have the most legs.
    """
    buckets: dict[str, list[Leg]] = {}
    for lg in sorted(legs, key=lambda l: l.edge, reverse=True):
        buckets.setdefault(_category(lg.market_name), []).append(lg)
    pool: list[Leg] = []
    while len(pool) < cap:
        progressed = False
        for cat in list(buckets):
            if buckets[cat]:
                pool.append(buckets[cat].pop(0))
                progressed = True
                if len(pool) >= cap:
                    break
        if not progressed:
            break
    return pool


def _same_game_conflict(a: Leg, b: Leg) -> bool:
    """
    True if two legs from the SAME game cannot sensibly be combined:
      * mutually exclusive  -> they never both happen (e.g. Over 2.5 & Under 2.5,
        a win & a draw, two different correct scores). Joint count == 0.
      * nested / redundant  -> one logically implies the other, so the combo is a
        'given' (e.g. Under 2.5 implies Under 3.5; a correct score implies the
        result). Joint count == one leg's own count.
    Detected exactly from the simulation outcome arrays.
    """
    A = a.book.outcomes[a.market_name]
    B = b.book.outcomes[b.market_name]
    ab = int((A & B).sum())
    if ab == 0:
        return True
    if ab == int(A.sum()) or ab == int(B.sum()):
        return True
    return False


def _enumerate_parlays(candidate_legs: list[Leg], min_legs: int, max_legs: int,
                       candidate_pool_cap: int, min_distinct_games: int = 1,
                       max_legs_per_game: int = 99, max_legs_per_category: int = 99):
    """
    Yield every distinct, valid evaluated Parlay over the candidate pool.

    Only combo-eligible leg sets are emitted: within a single game, no two legs may
    be mutually exclusive or nested/redundant (see _same_game_conflict). Legs from
    different games are always combinable (independent).

    min_legs=1 allows single-bet suggestions (a one-leg "parlay" that meets the
    confidence/multiplier floors). Single bets are exempt from min_distinct_games.

    min_distinct_games:    multi-leg parlays must span >= this many games.
    max_legs_per_game:     cap legs from any one game.
    max_legs_per_category: cap legs from any one market category (scorer, total,
                           btts, ...), so niche markets stay viable but don't dominate.
    """
    pool = _select_pool(candidate_legs, candidate_pool_cap)

    # Precompute conflicting same-game pairs once (by pool index).
    conflict: set[tuple[int, int]] = set()
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if pool[i].match_key == pool[j].match_key and \
               _same_game_conflict(pool[i], pool[j]):
                conflict.add((i, j))

    seen: set[frozenset] = set()
    for r in range(max(1, min_legs), max_legs + 1):
        for idx in itertools.combinations(range(len(pool)), r):
            if any((idx[x], idx[y]) in conflict
                   for x in range(r) for y in range(x + 1, r)):
                continue
            combo = [pool[k] for k in idx]
            # Team totals ("{team} over X.5 goals") are single-bet only -- they overlap
            # too heavily with match totals / result / BTTS to be a clean parlay leg.
            if r > 1 and any(_category(lg.market_name) == "team_total" for lg in combo):
                continue
            keys = {(lg.match_key, lg.market_name) for lg in combo}
            if len(keys) != len(combo):
                continue
            game_counts = Counter(lg.match_key for lg in combo)
            if r > 1 and len(game_counts) < min_distinct_games:
                continue  # single bets are exempt from the cross-game requirement
            if max(game_counts.values()) > max_legs_per_game:
                continue
            cat_counts = Counter(_category(lg.market_name) for lg in combo)
            if max(cat_counts.values()) > max_legs_per_category:
                continue
            sig = frozenset((lg.kalshi_ticker, lg.market_name) for lg in combo)
            if sig in seen:
                continue
            seen.add(sig)
            yield evaluate_parlay(combo)


def find_best_parlays(
    candidate_legs: list[Leg],
    min_confidence: float,
    min_multiplier: float,
    max_legs: int = 4,
    min_legs: int = 2,
    top_n: int = 15,
    candidate_pool_cap: int = 40,
    rank_by: str = "ev",            # "ev" | "confidence" | "kelly"
    min_distinct_games: int = 1,
    max_legs_per_game: int = 99,
    max_legs_per_category: int = 99,
) -> list[Parlay]:
    """
    Exhaustively search leg combinations meeting BOTH constraints:
        joint_prob >= min_confidence  AND  multiplier >= min_multiplier
    Returns the top_n parlays ranked by `rank_by`. min_legs=1 includes single bets.

    Diversification: min_distinct_games / max_legs_per_game (cross-game) and
    max_legs_per_category (keeps niche markets viable but not overrepresented).

    Workload note: per the requirement, we prioritize finding hitting parlays over
    efficiency. We still cap the candidate pool so the search stays finite; raise
    candidate_pool_cap to widen it.
    """
    results = [par for par in _enumerate_parlays(
                   candidate_legs, min_legs, max_legs, candidate_pool_cap,
                   min_distinct_games, max_legs_per_game, max_legs_per_category)
               if par.joint_prob >= min_confidence and par.multiplier >= min_multiplier]
    keyfn = {
        "ev": lambda p: p.ev,
        "confidence": lambda p: p.joint_prob,
        "kelly": lambda p: p.half_kelly,
    }[rank_by]
    results.sort(key=keyfn, reverse=True)
    return results[:top_n]


@dataclass
class NearMiss:
    parlay: Parlay
    failed: str          # "confidence" | "multiplier"
    shortfall: float     # relative gap to the failed constraint (0..1+)

    def describe(self) -> str:
        if self.failed == "confidence":
            why = (f"FAILED confidence: {self.parlay.joint_prob*100:.1f}% "
                   f"(short by {self.shortfall*100:.1f} pts of target)")
        else:
            why = (f"FAILED multiplier: {self.parlay.multiplier:.2f}x "
                   f"(short of target)")
        return f"~{why}\n  {self.parlay.describe()}"


def find_near_miss_parlays(
    candidate_legs: list[Leg],
    min_confidence: float,
    min_multiplier: float,
    max_legs: int = 4,
    min_legs: int = 2,
    top_n: int = 8,
    candidate_pool_cap: int = 40,
    require_positive_ev: bool = True,
    min_distinct_games: int = 1,
    max_legs_per_game: int = 99,
    max_legs_per_category: int = 99,
) -> list[NearMiss]:
    """
    When nothing clears BOTH floors, surface the parlays that satisfied ONE floor and
    came closest on the other -- so you can see how far off you were and adjust knobs.

    A near miss must satisfy exactly one constraint (so it's genuinely 'close', not a
    blowout on both). By default we still require positive expected value, so we never
    suggest a money-losing combo even as a near miss. Ranked by smallest shortfall.
    """
    misses: list[NearMiss] = []
    for par in _enumerate_parlays(candidate_legs, min_legs, max_legs, candidate_pool_cap,
                                  min_distinct_games, max_legs_per_game,
                                  max_legs_per_category):
        if require_positive_ev and par.ev <= 0:
            continue
        meets_conf = par.joint_prob >= min_confidence
        meets_mult = par.multiplier >= min_multiplier
        if meets_conf == meets_mult:
            continue  # both pass (handled elsewhere) or both fail (not 'close')
        if meets_mult and not meets_conf:
            shortfall = (min_confidence - par.joint_prob) / min_confidence
            misses.append(NearMiss(par, "confidence", shortfall))
        else:  # meets_conf and not meets_mult
            shortfall = (min_multiplier - par.multiplier) / min_multiplier
            misses.append(NearMiss(par, "multiplier", shortfall))

    misses.sort(key=lambda m: m.shortfall)
    return misses[:top_n]
