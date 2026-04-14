"""
Macro Data Pipeline — FRED + Refinitiv
=======================================
Downloads weekly macro data from FRED (VIX, yields, fed funds, PMI)
and merges with Refinitiv data (DXY, Gold, Brent) into a single
aligned weekly DataFrame.

Output: data/macro/combined_macro_weekly.csv
"""

import os
import pandas as pd
import numpy as np
from fredapi import Fred
import importlib

# Optional: import your Refinitiv macro module
# import src.macro as macro
import refinitiv.data as rd
from datetime import datetime
import time
# =============================================================================
# Configuration
# =============================================================================

START_DATE = "2006-01-01"
END_DATE   = "2020-12-31"
FRED_API_KEY = "0d472975ba8d0e5ee549648673b1e3de"

OUTPUT_DIR = "data/macro"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FRED series: code -> (friendly_name, frequency)
# "D" = daily (needs resampling), "M" = monthly (needs resampling)
FRED_SERIES = {
    "VIXCLS":   ("VIX",            "D"),
    "DGS5":     ("US_Treasury_5Y", "D"),
    "DGS10":    ("US_Treasury_10Y","D"),
    "DGS30":    ("US_Treasury_30Y","D"),
    "FEDFUNDS": ("Fed_Funds_Rate", "M"),
    "NAPM":   ("ISM PMI Composite Index", "M"),  # proxy for ISM PMI
    # 3-month Eurodollar deposit rate (≈ LIBOR)
    #"USD3MTD156N": ("USD_3M_LIBOR", "D"),
}

# Refinitiv instruments (uncomment the ones you have access to)
REFINITIV_INSTRUMENTS = {
    ".DXY": (
        "DXY_Dollar_Index",
        ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
        {"OPEN_PRC": "OPEN", "HIGH_1": "HIGH", "LOW_1": "LOW", "TRDPRC_1": "CLOSE"},
    ),
    "XAU=": (
        "Gold_Spot",
        ["MID_OPEN", "MID_HIGH", "MID_LOW", "MID_PRICE"],
        {"MID_OPEN": "OPEN", "MID_HIGH": "HIGH", "MID_LOW": "LOW", "MID_PRICE": "CLOSE"},
    ),
    "LCOc1": (
        "Brent_Crude_Oil",
        ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
        {"OPEN_PRC": "OPEN", "HIGH_1": "HIGH", "LOW_1": "LOW", "TRDPRC_1": "CLOSE"},
    ),
}


# =============================================================================
# 1. Download & resample FRED data
# =============================================================================

def download_fred_data(api_key, series_dict, start, end):
    """
    Download all FRED series and resample to weekly (Friday).
    Returns a dict of {friendly_name: pd.Series (weekly)}.
    """
    fred = Fred(api_key=api_key)
    weekly_series = {}

    for code, (name, freq) in series_dict.items():
        print(f"  Downloading FRED: {code} ({name})...")
        raw = fred.get_series(code, observation_start=start, observation_end=end)

        # Convert to numeric, coerce '.' to NaN
        raw = pd.to_numeric(raw, errors="coerce")

        # Resample to weekly Friday — use last available observation
        weekly = raw.resample("W-FRI").last()

        # For monthly series, forward-fill within the month so weekly
        # rows between monthly releases carry the last known value
        if freq == "M":
            weekly = weekly.ffill()

        weekly.name = name
        weekly_series[name] = weekly

        # Save individual CSV
        weekly.to_csv(os.path.join(OUTPUT_DIR, f"{name}_weekly.csv"), header=True)
        print(f"    -> {len(weekly)} weekly observations saved.")

    return weekly_series


# =============================================================================
# 2. Download Refinitiv data  (skip if you don't have access)
# =============================================================================

def download_refinitiv_data(instruments, start, end, output_dir):
    """
    Download Refinitiv instruments via your macro module.
    Returns a dict of {friendly_name: pd.DataFrame} with OHLC columns.
    """
    try:
        import refinitiv.data as rd
        #import src.macro as macro
    except ImportError:
        print("  Refinitiv/macro module not available — skipping.")
        return {}

    rd.open_session()
    ensure_dir(output_dir)
    all_data = {}

    for ric, (name, fields, rename_map) in instruments.items():
        print(f"  Downloading Refinitiv: {ric} ({name})...")
        df = download_instrument(ric, name, fields, rename_map, start, end, output_dir)
        all_data[name] = df

    rd.close_session()
    return all_data


def resample_refinitiv_to_weekly(refinitiv_data):
    """
    Resample Refinitiv OHLC DataFrames to weekly.
    Returns dict of {name: pd.Series} using the CLOSE column.
    """
    weekly = {}
    for name, df in refinitiv_data.items():
        if df is None or df.empty:
            continue
        # Use CLOSE column, resample to weekly Friday
        close = df["CLOSE"] if "CLOSE" in df.columns else df.iloc[:, -1]
        close = pd.to_numeric(close, errors="coerce")
        w = close.resample("W-FRI").last()
        w.name = name
        weekly[name] = w
    return weekly


# =============================================================================
# 3. Merge everything into a single clean weekly DataFrame
# =============================================================================

def build_combined_weekly(fred_weekly, refinitiv_weekly=None):
    """
    Merge all weekly series into one DataFrame.
    Forward-fills small gaps (up to 2 weeks), then drops remaining NaN rows.
    """
    all_series = list(fred_weekly.values())
    if refinitiv_weekly:
        all_series.extend(refinitiv_weekly.values())

    df = pd.concat(all_series, axis=1)
    df.index.name = "Date"

    # Forward-fill gaps of up to 2 weeks (holidays, missing data)
    df = df.ffill(limit=2)

    # Report missing data before dropping
    missing = df.isna().sum()
    if missing.any():
        print("\n  Remaining NaNs after forward-fill (limit=2):")
        print(missing[missing > 0].to_string())

    # Drop rows where any column is still NaN
    n_before = len(df)
    df = df.dropna()
    n_after = len(df)
    print(f"\n  Rows: {n_before} -> {n_after} after dropna ({n_before - n_after} removed)")

    return df


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def ensure_dir(path: str):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def download_instrument(ric: str, name: str, fields: list, rename_map: dict,
                        start: str, end: str, output_dir: str) -> pd.DataFrame | None:
    """
    Download weekly data for a single instrument, looping year-by-year.
    Returns the combined DataFrame (or None if no data).
    """
    print(f"\n[Weekly] Downloading {name} ({ric}) ...")
    frames = []
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    current = start_dt
    while current < end_dt:
        year_end = min(datetime(current.year, 12, 31), end_dt)
        try:
            df = rd.get_history(
                universe=[ric],
                fields=fields,
                interval="weekly",
                start=current.strftime("%Y-%m-%d"),
                end=year_end.strftime("%Y-%m-%d"),
            )
            if df is not None and not df.empty:
                frames.append(df)
                print(f"  {current.year}: {len(df)} rows")
            else:
                print(f"  {current.year}: no data returned")
        except Exception as e:
            print(f"  {current.year}: ERROR - {e}")

        current = datetime(current.year + 1, 1, 1)
        time.sleep(0.5)

    if frames:
        result = pd.concat(frames)
        result = result.sort_index().loc[~result.index.duplicated(keep="first")]

        # Rename columns to standardised OPEN, HIGH, LOW, CLOSE
        result = result.rename(columns=rename_map)

        # Save individual file
        filepath = os.path.join(output_dir, f"{name}_weekly_2006_2020.csv")
        result.to_csv(filepath)
        print(f"  -> Saved {filepath} ({len(result)} rows)")
        return result
    else:
        print(f"  -> No data for {name}")
        return None


def create_combined_file(all_data: dict, output_dir: str):
    """
    Combine all instrument DataFrames into a single CSV.
    Uses only the CLOSE column from each instrument.
    """
    close_frames = {}
    for name, df in all_data.items():
        if df is not None and "CLOSE" in df.columns:
            close_frames[name] = df["CLOSE"]

    if close_frames:
        combined = pd.DataFrame(close_frames)
        combined = combined.sort_index()
        filepath = os.path.join(output_dir, "macro_weekly_combined_2006_2020.csv")
        combined.to_csv(filepath)
        print(f"\n[Combined] Saved {filepath} ({len(combined)} rows, {len(combined.columns)} instruments)")
    else:
        print("\n[Combined] No data to combine.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Refinitiv Macro & Commodity Data Downloader")
    print("=" * 60)

    # Open Refinitiv session
    print("\nOpening Refinitiv session...")
    rd.open_session()
    print("Session opened successfully.\n")

    ensure_dir(OUTPUT_DIR)
    all_data = {}

    try:
        # 1. Download each instrument individually
        print("-" * 40)
        print("STEP 1: Downloading WEEKLY data (individual files)")
        print("-" * 40)

        for ric, (name, fields, rename_map) in INSTRUMENTS.items():
            df = download_instrument(ric, name, fields, rename_map,
                                     START_DATE, END_DATE, OUTPUT_DIR)
            all_data[name] = df

        # 2. Create combined file (CLOSE prices only)
        print("\n" + "-" * 40)
        print("STEP 2: Creating combined weekly file (CLOSE only)")
        print("-" * 40)
        create_combined_file(all_data, OUTPUT_DIR)

    finally:
        rd.close_session()
        print("\nRefinitiv session closed.")

    print("\n" + "=" * 60)
    print(f"Done! Check the '{OUTPUT_DIR}' folder for your data.")
    print("=" * 60)

    print("""
──────────────────────────────────────────────────────
NOTE: Fed Funds Rate, Eurodollar, and US ISM PMI are
not available on your Refinitiv licence.
Use FRED as a free fallback:

    pip install fredapi

    from fredapi import Fred
    fred = Fred(api_key='YOUR_FREE_KEY')
    # Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html

    # Fed Funds Effective Rate (monthly)
    fed_funds = fred.get_series('FEDFUNDS', start_date='2006-01-01')
    fed_funds.to_csv('data/macro/Fed_Funds_Rate_monthly.csv')

    # ISM Manufacturing PMI (monthly)
    pmi = fred.get_series('MANEMP', start_date='2006-01-01')
    pmi.to_csv('data/macro/US_ISM_PMI_monthly.csv')

    # SOFR rate (replaced Eurodollar, available from Apr 2018)
    sofr = fred.get_series('SOFR', start_date='2018-04-03')
    sofr.to_csv('data/macro/SOFR_daily.csv')
──────────────────────────────────────────────────────
""")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Step 1: Downloading FRED data")
    print("=" * 60)
    fred_weekly = download_fred_data(FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE)

    print("\n" + "=" * 60)
    print("  Step 2: Downloading Refinitiv data")
    print("=" * 60)
    refinitiv_raw = download_refinitiv_data(REFINITIV_INSTRUMENTS, START_DATE, END_DATE, OUTPUT_DIR)
    refinitiv_weekly = resample_refinitiv_to_weekly(refinitiv_raw)

    print("\n" + "=" * 60)
    print("  Step 3: Building combined weekly dataset")
    print("=" * 60)
    combined = build_combined_weekly(fred_weekly, refinitiv_weekly)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "combined_macro_weekly.csv")
    combined.to_csv(out_path)
    print(f"\n  Saved to: {out_path}")
    print(f"  Shape: {combined.shape}")
    print(f"  Date range: {combined.index[0].date()} to {combined.index[-1].date()}")
    print(f"\n  Columns: {list(combined.columns)}")
    print("\n  Head:")
    print(combined.head().to_string())
    print("\n  Tail:")
    print(combined.tail().to_string())