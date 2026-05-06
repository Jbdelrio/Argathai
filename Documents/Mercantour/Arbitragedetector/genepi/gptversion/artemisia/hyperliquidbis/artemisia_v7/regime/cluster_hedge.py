"""
cluster_hedge.py — Correlation-based clustering for S1 hedge selection.
Inspired by Jing 2025. Uses networkx Louvain communities instead of
the unavailable python-louvain package.

Purpose: replace the naive "pick most-correlated coin" with a community-
stable hedge that is less prone to flash correlations breaking at entry.
"""
import logging
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities

log = logging.getLogger(__name__)


def build_correlation_graph(returns_df: pd.DataFrame,
                             min_correlation: float = 0.50) -> nx.Graph:
    """
    Build an undirected graph where an edge exists between two coins
    if their return correlation exceeds min_correlation.

    Args:
        returns_df: DataFrame of close-to-close returns, columns = coin names
        min_correlation: Minimum |correlation| to add an edge

    Returns:
        networkx Graph with 'weight' attribute = correlation value
    """
    corr = returns_df.corr(numeric_only=True)
    G = nx.Graph()
    G.add_nodes_from(corr.columns)

    for i in corr.columns:
        for j in corr.columns:
            if i >= j:
                continue
            c = corr.loc[i, j]
            if not np.isnan(c) and abs(c) >= min_correlation:
                G.add_edge(i, j, weight=float(c))

    log.debug(
        "Correlation graph: %d nodes, %d edges (min_corr=%.2f)",
        G.number_of_nodes(), G.number_of_edges(), min_correlation
    )
    return G


def detect_communities(G: nx.Graph, seed: int = 42) -> dict[str, int]:
    """
    Detect communities via Louvain algorithm (networkx built-in).

    Returns dict: {coin -> community_id (int)}
    """
    if G.number_of_edges() == 0:
        # No edges: each node is its own community
        return {node: i for i, node in enumerate(G.nodes())}

    communities = louvain_communities(G, weight="weight", seed=seed)
    mapping = {}
    for comm_id, community in enumerate(communities):
        for node in community:
            mapping[node] = comm_id

    log.debug(
        "Louvain communities: %d detected from %d nodes",
        len(communities), G.number_of_nodes()
    )
    return mapping


def find_hedge_via_clustering(
    target_coin: str,
    candles_map: dict[str, pd.DataFrame],
    funding_rates: dict[str, float],
    min_correlation: float = 0.50,
    lookback_bars: int = 4032,   # 14 days * 288 bars/day at 5-min, or 14*96 at 15-min
    min_funding_ratio: float = 0.30,
) -> str | None:
    """
    Find the best hedge coin for target_coin using Louvain clustering.

    Instead of picking the single most-correlated coin (naive), this:
    1. Builds a correlation graph across all available coins
    2. Detects stable correlation communities via Louvain
    3. Finds coins in the SAME community as target_coin
    4. Among those, picks the one with the lowest funding rate
       (we're short target_coin with high funding; hedge coin should have
        low/neutral funding to minimise the cost of hedging)

    Args:
        target_coin: The high-funding coin we want to hedge
        candles_map: {coin: DataFrame with 'close' column}
        funding_rates: {coin: annualised funding % (signed)}
        min_correlation: Edge threshold for graph construction
        lookback_bars: How many recent bars to use for correlation
        min_funding_ratio: Hedge coin's abs(funding) must be < this * target's

    Returns:
        Best hedge coin symbol, or None if no valid hedge found
    """
    # Build returns matrix
    returns_dict = {}
    for coin, df in candles_map.items():
        if df.empty or "close" not in df.columns:
            continue
        closes = df["close"].tail(lookback_bars)
        if len(closes) < 50:
            continue
        returns_dict[coin] = closes.pct_change().dropna()

    if target_coin not in returns_dict:
        log.warning("Target coin %s not in returns data", target_coin)
        return None

    # Align all series to same length
    min_len = min(len(s) for s in returns_dict.values())
    aligned = pd.DataFrame({
        coin: s.values[-min_len:]
        for coin, s in returns_dict.items()
    })

    if aligned.shape[1] < 3:
        log.warning("Not enough coins (%d) for clustering", aligned.shape[1])
        return None

    # Build graph and detect communities
    G = build_correlation_graph(aligned, min_correlation=min_correlation)
    communities = detect_communities(G)

    target_community = communities.get(target_coin)
    if target_community is None:
        log.warning("%s not in any community (isolated node)", target_coin)
        # Fall back to simple correlation
        return _fallback_correlation(target_coin, aligned, funding_rates, min_funding_ratio)

    # Coins in same community (excluding target)
    same_community = [
        coin for coin, comm in communities.items()
        if comm == target_community and coin != target_coin
    ]

    if not same_community:
        log.warning("No other coins in %s's community", target_coin)
        return _fallback_correlation(target_coin, aligned, funding_rates, min_funding_ratio)

    # Filter by funding: hedge should have low absolute funding
    target_funding_abs = abs(funding_rates.get(target_coin, 100))
    candidates = [
        coin for coin in same_community
        if abs(funding_rates.get(coin, 100)) < target_funding_abs * min_funding_ratio
    ]

    if not candidates:
        # Relax: just take lowest-funding coin in community
        candidates = sorted(
            same_community,
            key=lambda c: abs(funding_rates.get(c, 100))
        )
        log.info(
            "%s: no candidates meeting funding ratio %.2f, using lowest-funding: %s",
            target_coin, min_funding_ratio, candidates[0]
        )

    if not candidates:
        return None

    # Among candidates, pick highest correlation to target
    corr_with_target = aligned[candidates].corrwith(aligned[target_coin])
    best_hedge = corr_with_target.idxmax()

    log.info(
        "Hedge for %s: %s (community %d, corr=%.3f, funding=%.1f%%)",
        target_coin, best_hedge, target_community,
        float(corr_with_target[best_hedge]),
        funding_rates.get(best_hedge, 0)
    )
    return best_hedge


def _fallback_correlation(target_coin: str,
                           aligned: pd.DataFrame,
                           funding_rates: dict[str, float],
                           min_funding_ratio: float) -> str | None:
    """Simple correlation fallback when clustering fails."""
    if target_coin not in aligned.columns:
        return None

    target_funding_abs = abs(funding_rates.get(target_coin, 100))
    other_coins = [c for c in aligned.columns if c != target_coin]

    candidates = [
        c for c in other_coins
        if abs(funding_rates.get(c, 100)) < target_funding_abs * min_funding_ratio
    ]
    if not candidates:
        candidates = sorted(other_coins, key=lambda c: abs(funding_rates.get(c, 100)))

    if not candidates:
        return None

    corr = aligned[candidates].corrwith(aligned[target_coin])
    return corr.idxmax()


def rank_hedge_candidates(target_coin: str,
                           candles_map: dict[str, pd.DataFrame],
                           funding_rates: dict[str, float],
                           top_n: int = 3,
                           lookback_bars: int = 4032) -> list[tuple[str, float, float]]:
    """
    Return top_n hedge candidates ranked by (correlation in same community, low funding).
    Returns list of (coin, correlation, annual_funding_pct).
    """
    returns_dict = {}
    for coin, df in candles_map.items():
        if df.empty or "close" not in df.columns:
            continue
        closes = df["close"].tail(lookback_bars)
        if len(closes) < 50:
            continue
        returns_dict[coin] = closes.pct_change().dropna()

    if target_coin not in returns_dict or len(returns_dict) < 3:
        return []

    min_len = min(len(s) for s in returns_dict.values())
    aligned = pd.DataFrame({
        coin: s.values[-min_len:] for coin, s in returns_dict.items()
    })

    G = build_correlation_graph(aligned, min_correlation=0.40)
    communities = detect_communities(G)
    target_comm = communities.get(target_coin)

    # Score = correlation (higher is better) - funding_cost_ratio
    scores = []
    for coin in aligned.columns:
        if coin == target_coin:
            continue
        corr = float(aligned[coin].corr(aligned[target_coin]))
        same_comm = (communities.get(coin) == target_comm) if target_comm is not None else False
        # Community bonus: same community gets a correlation boost
        adj_corr = corr * (1.2 if same_comm else 1.0)
        funding = funding_rates.get(coin, 0)
        scores.append((coin, adj_corr, funding))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Synthetic test: 10 coins, BTC/ETH/SOL correlated
    rng = np.random.default_rng(42)
    n = 500
    btc = 100 + np.cumsum(rng.normal(0, 1, n))
    coins_data = {
        "BTC":  btc,
        "ETH":  btc * 0.08 + rng.normal(0, 0.5, n),   # high corr with BTC
        "SOL":  btc * 0.01 + rng.normal(0, 0.8, n),   # moderate corr
        "DOGE": rng.normal(0, 1, n),                    # uncorrelated
        "AVAX": btc * 0.005 + rng.normal(0, 0.6, n),  # moderate corr
    }

    candles_map = {}
    for coin, prices in coins_data.items():
        candles_map[coin] = pd.DataFrame({"close": prices})

    funding_rates = {
        "BTC": 45.0,   # high funding -> target to short
        "ETH": 5.0,    # low funding -> good hedge
        "SOL": 15.0,
        "DOGE": 2.0,
        "AVAX": 8.0,
    }

    hedge = find_hedge_via_clustering("BTC", candles_map, funding_rates)
    print(f"\nBest hedge for BTC (high funding): {hedge}")
    print("Expected: ETH (high corr + low funding)")

    candidates = rank_hedge_candidates("BTC", candles_map, funding_rates, top_n=3)
    print("\nTop 3 candidates:")
    for coin, corr, fund in candidates:
        print(f"  {coin}: adj_corr={corr:.3f}, funding={fund:.1f}%")
