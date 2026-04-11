"""
equities_refinitiv.py
=====================
Downloads equity index data from Refinitiv (LSEG) Workspace.

- 15-min intraday data (limited to ~1 year lookback via standard API)
- Daily data for the full 2006-2025 range (fallback / complement)

Indices covered:
    S&P 500, Dow Jones, Nasdaq, FTSE 100, CAC 40, DAX,
    Hang Seng, Nikkei 225, Kospi 200

Requirements:
    pip install refinitiv-data pandas
    Refinitiv Workspace must be running in the background.

Usage:
    python equities_refinitiv.py
"""

import refinitiv.data as rd
import pandas as pd
import os
import time
from datetime import datetime, timedelta

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

OUTPUT_DIR = "data/equities"

EQUITY_RICS = {
    ".SPX": "SP500",
    ".DJI": "DowJones",
    ".IXIC": "Nasdaq",
    ".FTSE": "FTSE100",
    ".FCHI": "CAC40",
    ".GDAXI": "DAX",
    ".HSI": "HangSeng",
    ".N225": "Nikkei225",
    ".KS200": "Kospi200",
}

FIELDS = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]

# Date range
START_DATE = "2006-01-01"
END_DATE = "2025-12-31"

# For intraday 15-min: Refinitiv standard API typically allows ~1 year.
# Adjust this if you have Tick History access (see below).
INTRADAY_LOOKBACK_DAYS = 365


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def ensure_dir(path: str):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def download_daily(rics: dict, start: str, end: str, output_dir: str):
    """
    Download daily OHLCV data for each RIC.
    Loops year-by-year to avoid hitting request size limits.
    """
    ensure_dir(output_dir)

    for ric, name in rics.items():
        print(f"\n[Daily] Downloading {name} ({ric}) ...")
        frames = []
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        # Loop year by year to stay within API limits
        current = start_dt
        while current < end_dt:
            year_end = min(
                datetime(current.year, 12, 31),
                end_dt,
            )
            try:
                df = rd.get_history(
                    universe=[ric],
                    fields=FIELDS,
                    interval="daily",
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
            time.sleep(0.5)  # be gentle with the API

        if frames:
            result = pd.concat(frames)
            result = result.sort_index().loc[~result.index.duplicated(keep="first")]
            filepath = os.path.join(output_dir, f"{name}_daily_2006_2025.csv")
            result.to_csv(filepath)
            print(f"  -> Saved {filepath} ({len(result)} rows)")
        else:
            print(f"  -> No data for {name}")


def download_intraday_15min(rics: dict, output_dir: str):
    """
    Download 15-min intraday data.

    NOTE: The standard Refinitiv Data API typically limits intraday
    history to ~1 year. For data going back to 2006 at 15-min
    frequency, you need Refinitiv Tick History (RTH) via
    DataScope Select — see download_intraday_tick_history() below.
    """
    ensure_dir(output_dir)

    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=INTRADAY_LOOKBACK_DAYS)

    for ric, name in rics.items():
        print(f"\n[15-min] Downloading {name} ({ric}) ...")
        try:
            df = rd.get_history(
                universe=[ric],
                fields=FIELDS,
                interval="15min",
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
            )
            if df is not None and not df.empty:
                filepath = os.path.join(output_dir, f"{name}_15min_recent.csv")
                df.to_csv(filepath)
                print(f"  -> Saved {filepath} ({len(df)} rows)")
            else:
                print(f"  -> No 15-min data returned for {name}")
        except Exception as e:
            print(f"  -> ERROR: {e}")

        time.sleep(0.5)


def download_intraday_tick_history(rics: dict, start: str, end: str, output_dir: str):
    """
    Download 15-min data for the full date range via Tick History (RTH).

    This requires a Tick History / DataScope Select entitlement.
    Uncomment and adapt if you have access.
    """
    # from refinitiv.data.delivery import endpoint_request
    #
    # ensure_dir(output_dir)
    #
    # for ric, name in rics.items():
    #     print(f"\n[Tick History 15-min] Downloading {name} ({ric}) ...")
    #
    #     request_body = {
    #         "ExtractionRequest": {
    #             "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.IntradayBarExtractionRequest",
    #             "ContentFieldNames": ["Open", "High", "Low", "Close", "Volume"],
    #             "IdentifierList": {
    #                 "@odata.type": "#DataScope.Select.Api.Extractions.ExtractionRequests.InstrumentIdentifierList",
    #                 "InstrumentIdentifiers": [{"Identifier": ric, "IdentifierType": "Ric"}],
    #             },
    #             "Condition": {
    #                 "MessageTimeStampIn": "GmtUtc",
    #                 "ReportDateRangeType": "Range",
    #                 "QueryStartDate": start,
    #                 "QueryEndDate": end,
    #                 "Interval": "FifteenMinutes",
    #             },
    #         }
    #     }
    #
    #     # Submit extraction and download result
    #     # (implementation depends on your RTH setup)
    #     pass

    print(
        "\n[Tick History] This function is commented out by default.\n"
        "Uncomment and configure if you have Tick History / DataScope Select access.\n"
        "Contact your Refinitiv account manager for entitlement details."
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Refinitiv Equity Index Data Downloader")
    print("=" * 60)

    # Open Refinitiv session
    print("\nOpening Refinitiv session...")
    rd.open_session()
    print("Session opened successfully.\n")

    try:
        # 1. Daily data (full 2006-2025 range)
        print("-" * 40)
        print("STEP 1: Downloading DAILY data (2006-2025)")
        print("-" * 40)
        download_daily(EQUITY_RICS, START_DATE, END_DATE, OUTPUT_DIR)

        # 2. 15-min intraday (recent ~1 year via standard API)
        print("\n" + "-" * 40)
        print("STEP 2: Downloading 15-MIN intraday data (recent)")
        print("-" * 40)
        download_intraday_15min(EQUITY_RICS, OUTPUT_DIR)

        # 3. (Optional) Full 15-min history via Tick History
        # Uncomment the next line if you have RTH access:
        # download_intraday_tick_history(EQUITY_RICS, START_DATE, END_DATE, OUTPUT_DIR)

    finally:
        rd.close_session()
        print("\nRefinitiv session closed.")

    print("\n" + "=" * 60)
    print("Done! Check the '{}' folder for your data.".format(OUTPUT_DIR))
    print("=" * 60)


if __name__ == "__main__":
    main()
