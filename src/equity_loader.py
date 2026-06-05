"""
equity_loader.py
--------------
Downloads daily OHLCV data for major stock indices, grouped by region.
Designed as the first stage of a clustering-based trading pipeline.

Two data sources are supported with mirrored APIs:
    * Yahoo Finance       -> load_index_data / get_close_prices
    * Refinitiv Workspace -> load_index_data_refinitiv / get_close_prices_refinitiv
      (LSEG Data Library, desktop session via a running Workspace)

Regions & instruments
    America : S&P 500, Dow Jones, Nasdaq Composite
    Europe  : FTSE 100, CAC 40, DAX
    Asia    : KOSPI, Nikkei 225, Hang Seng

Usage
-----
    from equity_loader import get_close_prices, get_close_prices_refinitiv

    # Yahoo Finance
    df = get_close_prices("2018-01-01", "2024-01-01", regions=["america"])

    # Refinitiv Workspace (same signature)
    df_america = get_close_prices_refinitiv(
        start_date="2018-01-01", end_date="2024-01-01",
        regions=["america"], interval="1d",
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Where get_close_prices_refinitiv() writes its output. Override via the
# EQUITY_OUTPUT_DIR environment variable, or just reassign this in your script:
#   import equity_loader
#   equity_loader.OUTPUT_DIR = "/path/to/data"
OUTPUT_DIR = os.environ.get("EQUITY_OUTPUT_DIR", "./data")

# ── Index registry (Yahoo tickers) ──────────────────────────────────────────

INDICES: Dict[str, Dict[str, str]] = {
    "america": {
        "SP500": "^GSPC",
        "DowJones": "^DJI",
        "Nasdaq": "^IXIC",
    },
    "europe": {
        "FTSE100": "^FTSE",
        "CAC40": "^FCHI",
        "DAX": "^GDAXI",
    },
    "asia": {
        "KOSPI": "^KS11",
        "Nikkei225": "^N225",
        "HangSeng": "^HSI",
    },
}

# ── Index registry (Refinitiv RICs) ─────────────────────────────────────────
# Same friendly names as INDICES, mapped to Reuters Instrument Codes.
# NOTE: index RIC entitlements vary by account — verify each one is
# permissioned in your Workspace if a download returns nothing.

INDICES_RIC: Dict[str, Dict[str, str]] = {
    "america": {
        "SP500": ".SPX",
        "DowJones": ".DJI",
        "Nasdaq": ".IXIC",
    },
    "europe": {
        "FTSE100": ".FTSE",
        "CAC40": ".FCHI",
        "DAX": ".GDAXI",
    },
    "asia": {
        "KOSPI": ".KS11",
        "Nikkei225": ".N225",
        "HangSeng": ".HSI",
    },
}

ALL_REGIONS = list(INDICES.keys())


# ── Helpers ─────────────────────────────────────────────────────────────────

def _resolve_regions(regions: Optional[List[str]] = None) -> List[str]:
    """Validate and normalise region names."""
    if regions is None:
        return ALL_REGIONS
    out = []
    for r in regions:
        r_low = r.strip().lower()
        if r_low not in INDICES:
            raise ValueError(
                f"Unknown region '{r}'. Choose from {ALL_REGIONS}"
            )
        out.append(r_low)
    return out


def _build_ticker_map(regions: List[str]) -> Dict[str, str]:
    """Return {friendly_name: yahoo_ticker} for the requested regions."""
    ticker_map: Dict[str, str] = {}
    for region in regions:
        ticker_map.update(INDICES[region])
    return ticker_map


def _build_ric_map(regions: List[str]) -> Dict[str, str]:
    """Return {friendly_name: refinitiv_ric} for the requested regions."""
    ric_map: Dict[str, str] = {}
    for region in regions:
        ric_map.update(INDICES_RIC[region])
    return ric_map


def _download_single(
    name: str,
    ticker: str,
    start: str,
    end: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Download one index from Yahoo Finance; return None on failure."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning("No data returned for %s (%s)", name, ticker)
            return None

        # Flatten MultiIndex columns that yfinance sometimes returns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df.columns = pd.MultiIndex.from_product(
            [[name], ["Open", "High", "Low", "Close", "Volume"]]
        )
        return df

    except Exception as exc:
        logger.error("Failed to download %s (%s): %s", name, ticker, exc)
        return None


# ── Refinitiv / LSEG Workspace helpers ──────────────────────────────────────

# Map this module's interval strings to the LSEG Data Library's intervals.
_RIC_INTERVAL_MAP: Dict[str, str] = {
    "1d": "daily",
    "1wk": "weekly",
    "1mo": "monthly",
    "daily": "daily",
    "weekly": "weekly",
    "monthly": "monthly",
}

# Standard Refinitiv OHLCV field codes (historical-pricing backend, used by
# get_history) -> our column names.
_RIC_FIELDS = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"]
_RIC_FIELD_RENAME = {
    "OPEN_PRC": "Open",
    "HIGH_1": "High",
    "LOW_1": "Low",
    "TRDPRC_1": "Close",
    "ACVOL_UNS": "Volume",
}

# Datagrid backend (used by get_data). TR.* fields tend to carry deeper
# history than get_history. The ".date" suffix on one field gives a Date
# column; the others come back under their display names.
_RIC_DATAGRID_FIELDS = [
    "TR.PriceOpen.date",
    "TR.PriceOpen",
    "TR.PriceHigh",
    "TR.PriceLow",
    "TR.PriceClose",
    "TR.Volume",
]
_RIC_DATAGRID_RENAME = {
    "Price Open": "Open",
    "Price High": "High",
    "Price Low": "Low",
    "Price Close": "Close",
    "Volume": "Volume",
}
_RIC_DATAGRID_FRQ = {
    "1d": "D", "1wk": "W", "1mo": "M",
    "daily": "D", "weekly": "W", "monthly": "M",
}

# get_history is capped at ~3,000 rows per request (an infrastructure limit,
# not an entitlement one). ~3,000 calendar days ≈ 8 years ≈ ~2,000 trading
# rows, comfortably under the cap, so we slice long ranges into chunks.
_RIC_CHUNK_DAYS = 3000


def _ensure_refinitiv_session() -> None:
    """
    Open a Refinitiv/LSEG session once and reuse it for subsequent calls.

    With no arguments, ``open_session()`` reads refinitiv-data.config.json
    (or falls back to a desktop session that connects to the Workspace app
    running on this machine). The session is intentionally left open so
    repeated loader calls don't re-authenticate every time.
    """
    import refinitiv.data as rd  # lazy import: only needed for Refinitiv path

    try:
        default = rd.session.get_default()
    except Exception:
        default = None

    if default is None:
        logger.info("Opening Refinitiv/LSEG session...")
        rd.open_session()


def _iter_date_chunks(start: str, end: str, chunk_days: int = _RIC_CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) ISO-date pairs covering [start, end]."""
    cur = pd.Timestamp(start)
    final = pd.Timestamp(end)
    step = pd.Timedelta(days=chunk_days)
    while cur <= final:
        chunk_end = min(cur + step, final)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + pd.Timedelta(days=1)


def _normalise_ohlcv(
    df: pd.DataFrame,
    name: str,
    rename_map: Dict[str, str],
) -> pd.DataFrame:
    """Coerce a raw Refinitiv frame into the (name, OHLCV) MultiIndex layout."""
    df = df.rename(columns=rename_map)

    # Backfill any missing field (e.g. indices with no volume) so the column
    # layout always matches the Yahoo path.
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.columns = pd.MultiIndex.from_product(
        [[name], ["Open", "High", "Low", "Close", "Volume"]]
    )
    return df


def _download_refinitiv_history(
    name: str,
    ric: str,
    start: str,
    end: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Chunked get_history pull (historical-pricing backend)."""
    import refinitiv.data as rd  # lazy import

    rd_interval = _RIC_INTERVAL_MAP.get(interval, interval)
    parts: List[pd.DataFrame] = []

    for c_start, c_end in _iter_date_chunks(start, end):
        try:
            chunk = rd.get_history(
                universe=ric,
                fields=_RIC_FIELDS,
                interval=rd_interval,
                start=c_start,
                end=c_end,
            )
        except Exception as exc:
            logger.error(
                "get_history failed for %s (%s) [%s → %s]: %s",
                name, ric, c_start, c_end, exc,
            )
            continue
        if chunk is not None and not chunk.empty:
            parts.append(chunk)

    if not parts:
        logger.warning("get_history returned no data for %s (%s)", name, ric)
        return None

    return _normalise_ohlcv(pd.concat(parts), name, _RIC_FIELD_RENAME)


def _download_refinitiv_datagrid(
    name: str,
    ric: str,
    start: str,
    end: str,
    interval: str,
) -> Optional[pd.DataFrame]:
    """Chunked get_data pull (datagrid backend, usually deeper history)."""
    import refinitiv.data as rd  # lazy import

    frq = _RIC_DATAGRID_FRQ.get(interval, "D")
    parts: List[pd.DataFrame] = []

    for c_start, c_end in _iter_date_chunks(start, end):
        try:
            chunk = rd.get_data(
                universe=[ric],
                fields=_RIC_DATAGRID_FIELDS,
                parameters={"SDate": c_start, "EDate": c_end, "Frq": frq},
            )
        except Exception as exc:
            logger.error(
                "get_data failed for %s (%s) [%s → %s]: %s",
                name, ric, c_start, c_end, exc,
            )
            continue
        if chunk is None or chunk.empty:
            continue

        # Locate the date column (display name is usually "Date") and use it
        # as the index; drop the instrument column.
        date_col = next(
            (c for c in chunk.columns if "date" in str(c).lower()), None
        )
        if date_col is None:
            logger.warning("get_data gave no date column for %s (%s)", name, ric)
            continue
        chunk = chunk.set_index(date_col)
        chunk = chunk.drop(columns=["Instrument"], errors="ignore")
        parts.append(chunk)

    if not parts:
        logger.warning("get_data returned no data for %s (%s)", name, ric)
        return None

    return _normalise_ohlcv(pd.concat(parts), name, _RIC_DATAGRID_RENAME)


def _download_single_refinitiv(
    name: str,
    ric: str,
    start: str,
    end: str,
    interval: str,
    method: str = "auto",
) -> Optional[pd.DataFrame]:
    """
    Download one index from Refinitiv Workspace; return None on failure.

    method
        "history"  – chunked get_history only.
        "datagrid" – chunked get_data (TR.*) only; favours deep history.
        "auto"     – try get_history; fall back to get_data if it returns
                     nothing, or if its earliest date falls well short of
                     the requested start (a sign of depth truncation).
    """
    if method == "history":
        return _download_refinitiv_history(name, ric, start, end, interval)
    if method == "datagrid":
        return _download_refinitiv_datagrid(name, ric, start, end, interval)

    # auto
    df = _download_refinitiv_history(name, ric, start, end, interval)
    if df is None or df.empty:
        logger.info("Falling back to get_data for %s (no get_history data)", name)
        return _download_refinitiv_datagrid(name, ric, start, end, interval)

    # Depth check: did get_history reach near the requested start?
    requested = pd.Timestamp(start)
    earliest = df.index.min()
    if (earliest - requested).days > 400:
        logger.info(
            "get_history for %s only reaches %s (requested %s); "
            "trying get_data for deeper history",
            name, earliest.date(), requested.date(),
        )
        deep = _download_refinitiv_datagrid(name, ric, start, end, interval)
        if deep is not None and not deep.empty and deep.index.min() < earliest:
            return deep

    return df


# ── Feature engineering helpers (useful for clustering) ─────────────────────

def compute_returns(
    close_df: pd.DataFrame,
    periods: List[int] = None,
) -> pd.DataFrame:
    """
    From a DataFrame of closing prices (columns = index names),
    compute log-returns over multiple horizons.

    Parameters
    ----------
    close_df : DataFrame with DatetimeIndex; one column per index.
    periods  : Return horizons in trading days. Default [1, 5, 21]
               (daily, weekly, monthly).

    Returns
    -------
    DataFrame with MultiIndex columns: (index_name, f"ret_{p}d").
    """
    import numpy as np

    if periods is None:
        periods = [1, 5, 21]

    frames = []
    for col in close_df.columns:
        series = close_df[col]
        for p in periods:
            ret = np.log(series / series.shift(p))
            ret.name = (col, f"ret_{p}d")
            frames.append(ret)

    return pd.concat(frames, axis=1)


def compute_volatility(
    close_df: pd.DataFrame,
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    Rolling annualised volatility of log-returns.

    Parameters
    ----------
    close_df : DataFrame of closing prices.
    windows  : Rolling window sizes in days. Default [21, 63]
               (1-month, 3-month).

    Returns
    -------
    DataFrame with MultiIndex columns: (index_name, f"vol_{w}d").
    """
    import numpy as np

    if windows is None:
        windows = [21, 63]

    log_ret = np.log(close_df / close_df.shift(1))
    frames = []
    for col in log_ret.columns:
        for w in windows:
            vol = log_ret[col].rolling(w).std() * np.sqrt(252)
            vol.name = (col, f"vol_{w}d")
            frames.append(vol)

    return pd.concat(frames, axis=1)


def build_feature_matrix(
    close_df: pd.DataFrame,
    return_periods: List[int] = None,
    vol_windows: List[int] = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    One-stop feature builder for clustering.

    Combines multi-horizon returns and rolling volatilities into a single
    matrix where each row is a date and each column is
    (index_name, feature_name).

    Parameters
    ----------
    close_df       : Closing prices, one column per index.
    return_periods : Passed to compute_returns.
    vol_windows    : Passed to compute_volatility.
    dropna         : Drop rows with any NaN (warm-up period).

    Returns
    -------
    DataFrame ready for sklearn clustering / PCA.
    """
    rets = compute_returns(close_df, return_periods)
    vols = compute_volatility(close_df, vol_windows)
    features = pd.concat([rets, vols], axis=1)
    if dropna:
        features.dropna(inplace=True)
    return features


# ── Main loader (Yahoo Finance) ─────────────────────────────────────────────

def load_index_data(
    start_date: str,
    end_date: str,
    regions: Optional[List[str]] = None,
    interval: str = "1d",
    merge: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Download index OHLCV data from Yahoo Finance.

    Parameters
    ----------
    start_date : str  – ISO date, e.g. "2020-01-01".
    end_date   : str  – ISO date, e.g. "2024-01-01".
    regions    : list  – Any subset of ["america", "europe", "asia"].
                         None (default) downloads all three.
    interval   : str  – Bar size: "1d", "1wk", "1mo", etc.
    merge      : bool – If True, return one wide DataFrame with a
                         MultiIndex on columns (index_name, OHLCV).
                         If False, return a dict {name: DataFrame}.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
    """
    regions = _resolve_regions(regions)
    ticker_map = _build_ticker_map(regions)

    logger.info(
        "Downloading %d indices for regions %s  [%s → %s]",
        len(ticker_map), regions, start_date, end_date,
    )

    results: Dict[str, pd.DataFrame] = {}
    frames: List[pd.DataFrame] = []

    for name, ticker in ticker_map.items():
        df = _download_single(name, ticker, start_date, end_date, interval)
        if df is not None:
            results[name] = df
            frames.append(df)

    if not frames:
        raise RuntimeError("All downloads failed – check network / dates.")

    if not merge:
        # Strip the MultiIndex so each frame has plain OHLCV columns
        plain = {}
        for name, df in results.items():
            flat = df.copy()
            flat.columns = flat.columns.get_level_values(1)
            plain[name] = flat
        return plain

    merged = pd.concat(frames, axis=1).sort_index()
    logger.info("Merged shape: %s", merged.shape)
    return merged


def get_close_prices(
    start_date: str,
    end_date: str,
    regions: Optional[List[str]] = None,
    interval: str = "1d",
    fill_method: Optional[str] = "ffill",
) -> pd.DataFrame:
    """
    Convenience wrapper – returns *only* closing prices as a simple
    DataFrame (columns = index names, rows = dates).

    Handy as direct input to ``build_feature_matrix``.

    Parameters
    ----------
    fill_method : How to handle NaN from different trading calendars.
                  "ffill" (default) forward-fills; None leaves gaps.
    """
    raw = load_index_data(
        start_date, end_date, regions=regions, interval=interval, merge=False
    )
    close = pd.DataFrame({name: df["Close"] for name, df in raw.items()})
    close.sort_index(inplace=True)

    if fill_method == "ffill":
        close.ffill(inplace=True)
    elif fill_method is not None:
        close.fillna(method=fill_method, inplace=True)

    return close


# ── Main loader (Refinitiv / LSEG Workspace) ────────────────────────────────

def load_index_data_refinitiv(
    start_date: str,
    end_date: str,
    regions: Optional[List[str]] = None,
    interval: str = "1d",
    merge: bool = True,
    method: str = "auto",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Download index OHLCV data from Refinitiv Workspace (LSEG Data Library).

    Mirrors :func:`load_index_data` exactly, but pulls from a desktop
    session connected to a running Refinitiv/LSEG Workspace instead of
    Yahoo Finance. Uses the RICs in ``INDICES_RIC``.

    Long date ranges are sliced into chunks automatically to work around
    the ~3,000-row-per-request cap on get_history.

    Parameters
    ----------
    start_date : str  – ISO date, e.g. "2020-01-01".
    end_date   : str  – ISO date, e.g. "2024-01-01".
    regions    : list  – Any subset of ["america", "europe", "asia"].
                         None (default) downloads all three.
    interval   : str  – "1d", "1wk", "1mo" (mapped to LSEG intervals).
    merge      : bool – If True, return one wide DataFrame with a
                         MultiIndex on columns (index_name, OHLCV).
                         If False, return a dict {name: DataFrame}.
    method     : str  – "auto" (default), "history", or "datagrid".
                         "datagrid" uses get_data (TR.* fields), which tends
                         to carry deeper history for old date ranges.

    Returns
    -------
    pd.DataFrame or dict[str, pd.DataFrame]
    """
    regions = _resolve_regions(regions)
    ric_map = _build_ric_map(regions)

    _ensure_refinitiv_session()

    logger.info(
        "Downloading %d indices from Refinitiv for regions %s  [%s → %s]",
        len(ric_map), regions, start_date, end_date,
    )

    results: Dict[str, pd.DataFrame] = {}
    frames: List[pd.DataFrame] = []

    for name, ric in ric_map.items():
        df = _download_single_refinitiv(
            name, ric, start_date, end_date, interval, method=method
        )
        if df is not None:
            results[name] = df
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "All Refinitiv downloads failed – check that Workspace is "
            "running, the session opened, and the RICs are entitled."
        )

    if not merge:
        plain = {}
        for name, df in results.items():
            flat = df.copy()
            flat.columns = flat.columns.get_level_values(1)
            plain[name] = flat
        return plain

    merged = pd.concat(frames, axis=1).sort_index()
    logger.info("Merged shape: %s", merged.shape)
    return merged


def get_close_prices_refinitiv(
    start_date: str,
    end_date: str,
    regions: Optional[List[str]] = None,
    interval: str = "1d",
    fill_method: Optional[str] = "ffill",
    save: bool = True,
    output_dir: Optional[str] = None,
    method: str = "auto",
) -> pd.DataFrame:
    """
    Refinitiv counterpart to :func:`get_close_prices`.

    Returns *only* closing prices as a simple DataFrame
    (columns = index names, rows = dates), ready for
    ``build_feature_matrix``.

    Example
    -------
        df_america = get_close_prices_refinitiv(
            start_date=start_date, end_date=end_date,
            regions=["america"], interval=interval,
        )

    Parameters
    ----------
    fill_method : How to handle NaN from different trading calendars.
                  "ffill" (default) forward-fills; None leaves gaps.
    save        : If True (default), write the result to a CSV under
                  ``output_dir``.
    output_dir  : Destination folder. Defaults to the module-level
                  ``OUTPUT_DIR`` when None. Created if it doesn't exist.
    method      : "auto" (default), "history", or "datagrid". Passed to
                  ``load_index_data_refinitiv``; use "datagrid" to force the
                  deep-history get_data path.
    """
    raw = load_index_data_refinitiv(
        start_date, end_date, regions=regions, interval=interval,
        merge=False, method=method,
    )
    close = pd.DataFrame({name: df["Close"] for name, df in raw.items()})
    close.sort_index(inplace=True)

    if fill_method == "ffill":
        close.ffill(inplace=True)
    elif fill_method is not None:
        close.fillna(method=fill_method, inplace=True)

    if save:
        target_dir = Path(output_dir if output_dir is not None else OUTPUT_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        region_tag = "-".join(_resolve_regions(regions))
        fname = f"close_{region_tag}_{interval}_{start_date}_{end_date}.csv"
        out_path = target_dir / fname

        close.to_csv(out_path)
        logger.info("Saved close prices to %s", out_path)

    return close


# ── CLI quick-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Downloading sample data  (2022-01-01 → 2024-01-01)")
    print("=" * 60)

    close = get_close_prices("2022-01-01", "2024-01-01")
    print(f"\nClose prices shape : {close.shape}")
    print(close.tail())

    print("\n── Feature matrix for clustering ──")
    features = build_feature_matrix(close)
    print(f"Feature matrix shape : {features.shape}")
    print(features.tail())

    # Refinitiv equivalent (requires Workspace running + refinitiv-data):
    # df_america = get_close_prices_refinitiv(
    #     start_date="2022-01-01", end_date="2024-01-01",
    #     regions=["america"], interval="1d",
    # )
    # print(df_america.tail())