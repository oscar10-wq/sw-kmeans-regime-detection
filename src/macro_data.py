"""
macro_data.py
=============
Downloads weekly macro and commodity data from Refinitiv (LSEG) Workspace.

Instruments covered:
    - VIX (futures front-month)
    - DXY (US Dollar Index)
    - Gold (spot)
    - Brent Crude Oil (futures front-month)
    - US Treasuries (2Y, 5Y, 10Y, 30Y yields)

Fed Funds Rate and PMI are NOT available on this Refinitiv licence.
Use FRED (fredapi) as a fallback — see bottom of this script.

Requirements:
    pip install refinitiv-data pandas
    Refinitiv Workspace must be running in the background.

Usage:
    python macro_data.py
"""

import refinitiv.data as rd
import pandas as pd
import os
import time
from datetime import datetime

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OUTPUT_DIR = "data/macro"

# Each instrument has its own fields because Refinitiv uses
# different field names depending on instrument type.
#
# Format: RIC -> (friendly_name, [fields], rename_map)
# rename_map standardises column names to OPEN, HIGH, LOW, CLOSE

INSTRUMENTS = {
    # ── Volatility ──
    "VIXc1": (
        "VIX",
        ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
        {"OPEN_PRC": "OPEN", "HIGH_1": "HIGH", "LOW_1": "LOW", "TRDPRC_1": "CLOSE"},
    ),

    # ── Currency ──
    ".DXY": (
        "DXY_Dollar_Index",
        ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1"],
        {"OPEN_PRC": "OPEN", "HIGH_1": "HIGH", "LOW_1": "LOW", "TRDPRC_1": "CLOSE"},
    ),

    # ── Commodities ──
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

    # ── US Treasuries (yields) ──
    "US2YT=RR": (
        "US_Treasury_2Y",
        ["OPEN_YLD", "HIGH_YLD", "LOW_YLD", "MID_YLD_1"],
        {"OPEN_YLD": "OPEN", "HIGH_YLD": "HIGH", "LOW_YLD": "LOW", "MID_YLD_1": "CLOSE"},
    ),
    "US5YT=RR": (
        "US_Treasury_5Y",
        ["OPEN_YLD", "HIGH_YLD", "LOW_YLD", "MID_YLD_1"],
        {"OPEN_YLD": "OPEN", "HIGH_YLD": "HIGH", "LOW_YLD": "LOW", "MID_YLD_1": "CLOSE"},
    ),
    "US10YT=RR": (
        "US_Treasury_10Y",
        ["OPEN_YLD", "HIGH_YLD", "LOW_YLD", "MID_YLD_1"],
        {"OPEN_YLD": "OPEN", "HIGH_YLD": "HIGH", "LOW_YLD": "LOW", "MID_YLD_1": "CLOSE"},
    ),
    "US30YT=RR": (
        "US_Treasury_30Y",
        ["OPEN_YLD", "HIGH_YLD", "LOW_YLD", "MID_YLD_1"],
        {"OPEN_YLD": "OPEN", "HIGH_YLD": "HIGH", "LOW_YLD": "LOW", "MID_YLD_1": "CLOSE"},
    ),
}

# Date range
START_DATE = "2006-01-01"
END_DATE = "2020-12-31"


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


if __name__ == "__main__":
    main()