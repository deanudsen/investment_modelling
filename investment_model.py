import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# -----------------------------
# CONFIG
# -----------------------------
HOLDINGS_CSV = "income_baseline_holdings.csv"  # <- your holdings snapshot file

BASE_DATE = pd.Timestamp("2025-12-10")
YEARS = 10
MONTHS = YEARS * 12
ANNUAL_INFLATION = 0.02

# Use a fixed seed for reproducible "random" dividend variation
RNG = np.random.default_rng(42)

# Contribution scenarios (TOTAL per month, across all portfolios)
CONTRIBUTION_LEVELS = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]

OUTPUT_DIR = Path("projection_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# TICKER-LEVEL ASSUMPTIONS
# -----------------------------
# These are the realistic assumptions we agreed on.
# You can adjust any of these in one place.

TICKER_ASSUMPTIONS = {
    "BEP.UN": {"div": 2.07,   "price_g": 0.05,  "div_g": 0.03, "nav_g": 0.00},
    "CNQ":    {"div": 2.35,   "price_g": 0.10,  "div_g": 0.10, "nav_g": 0.00},
    "DGS":    {"div": 1.20,   "price_g": -0.02, "div_g": -0.02,"nav_g": -0.02},
    "ENB":    {"div": 3.74,   "price_g": 0.05,  "div_g": 0.03, "nav_g": 0.00},
    "FSZ":    {"div": 0.42,   "price_g": 0.02,  "div_g": 0.00, "nav_g": 0.00},
    "HMAX":   {"div": 2.16,   "price_g": 0.00,  "div_g": 0.00, "nav_g": 0.00},
    "SMAX":   {"div": 2.20,   "price_g": 0.00,  "div_g": 0.00, "nav_g": 0.00},
    "XEQT":   {"div": 0.73,   "price_g": 0.10,  "div_g": 0.02, "nav_g": 0.00},
    "MG":     {"div": 1.80,   "price_g": 0.10,  "div_g": 0.02, "nav_g": 0.00},
    "MTL":    {"div": 0.92,   "price_g": 0.02,  "div_g": 0.01, "nav_g": 0.00},
    "NWH.UN": {"div": 0.36*0.8,"price_g": -0.02,"div_g": 0.00, "nav_g": -0.02},  # 20% cut applied
    "QSR":    {"div": 2.20,   "price_g": 0.10,  "div_g": 0.06, "nav_g": 0.00},
    "T":      {"div": 1.64,   "price_g": 0.02,  "div_g": 0.01, "nav_g": 0.00},
    "TPZ":    {"div": 1.36,   "price_g": 0.05,  "div_g": 0.06, "nav_g": 0.00},
    "SPB":    {"div": 0.49,   "price_g": 0.02,  "div_g": 0.01, "nav_g": 0.00},
    "FRU":    {"div": 1.08,   "price_g": 0.05,  "div_g": 0.04, "nav_g": 0.00},
    "SRU.UN": {"div": 1.50,   "price_g": 0.00,  "div_g": 0.00, "nav_g": 0.00},
}

# Categories for hybrid allocation
GROWTH = {"CNQ", "XEQT", "QSR", "MG"}
STABLE = {"ENB", "TPZ", "FRU", "T", "MTL", "FSZ", "SPB", "BEP.UN"}
HIGH_YIELD = {"HMAX", "SMAX", "DGS", "NWH.UN", "SRU.UN"}

TARGET_CAT_WEIGHTS = {"Growth": 0.4, "Stable": 0.4, "HighYield": 0.2}


def category_of(ticker: str) -> str:
    if ticker in GROWTH:
        return "Growth"
    if ticker in STABLE:
        return "Stable"
    if ticker in HIGH_YIELD:
        return "HighYield"
    return "Stable"


# -----------------------------
# BUILD SNAPSHOT FROM FILE
# -----------------------------
def load_snapshot_from_file(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    # Make sure the columns we need exist
    required = {"Portfolio", "Ticker", "Quantity", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in holdings file: {missing}")

    # Aggregate per (Portfolio, Ticker)
    grouped = (
        df.groupby(["Portfolio", "Ticker"], as_index=False)
          .agg({"Quantity": "sum", "Price": "first"})
    )

    snapshot: dict[str, dict[str, dict]] = {}

    for _, row in grouped.iterrows():
        portfolio = str(row["Portfolio"])
        ticker = str(row["Ticker"])
        shares = float(row["Quantity"])
        price = float(row["Price"])

        # Validate quantity and price are positive
        if shares <= 0:
            raise ValueError(
                f"Invalid Quantity for ticker '{ticker}' in portfolio '{portfolio}': "
                f"{shares}. Quantity must be positive."
            )
        if price <= 0:
            raise ValueError(
                f"Invalid Price for ticker '{ticker}' in portfolio '{portfolio}': "
                f"{price}. Price must be positive."
            )

        if ticker not in TICKER_ASSUMPTIONS:
            raise ValueError(
                f"No assumptions entry in TICKER_ASSUMPTIONS for ticker '{ticker}'. "
                f"Please add it there before running the model."
            )

        assump = TICKER_ASSUMPTIONS[ticker]

        # Validate ticker assumptions contain required keys
        required_keys = {"div", "price_g", "div_g", "nav_g"}
        missing_keys = required_keys - set(assump.keys())
        if missing_keys:
            raise ValueError(
                f"TICKER_ASSUMPTIONS for '{ticker}' is missing required keys: {missing_keys}"
            )

        snapshot.setdefault(portfolio, {})
        snapshot[portfolio][ticker] = {
            "shares": shares,
            "price": price,
            "div": assump["div"],
            "price_g": assump["price_g"],
            "div_g": assump["div_g"],
            "nav_g": assump["nav_g"],
        }

    return snapshot


# -----------------------------
# PORTFOLIO WEIGHTS (for splitting contributions)
# -----------------------------
def compute_initial_portfolio_weights(snapshot: dict) -> dict:
    values = {}
    for p, holdings in snapshot.items():
        total = 0.0
        for t, info in holdings.items():
            total += info["shares"] * info["price"]
        values[p] = total
    grand = sum(values.values())
    
    if grand == 0:
        # If all portfolios have zero value, distribute contributions equally
        num_portfolios = len(values)
        if num_portfolios == 0:
            return {}
        return {p: 1.0 / num_portfolios for p in values}
    
    return {p: v / grand for p, v in values.items()}


# -----------------------------
# CORE SIMULATION
# -----------------------------
def run_realistic_projection(snapshot: dict,
                             portfolio_weights: dict,
                             monthly_total_contrib: float,
                             years: int = YEARS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (monthly_df, yearly_df) for a given total monthly contribution amount.
    Uses:
      - realistic price growth per ticker
      - realistic dividend evolution (including variability for HMAX/SMAX/DGS)
      - monthly contributions with hybrid allocation (Growth/Stable/HighYield)
      - inflation-adjusted values @ 2% per year
    """

    # Deep copy snapshot so we don't mutate the original
    state = {
        p: {t: info.copy() for t, info in holdings.items()}
        for p, holdings in snapshot.items()
    }

    months = years * 12
    dates = pd.date_range(BASE_DATE, periods=months + 1, freq="MS")
    rows = []

    for m in range(months + 1):
        date = dates[m]
        year_index = m // 12  # 0..years
        real_discount = (1 + ANNUAL_INFLATION) ** (m / 12)

        # Record snapshot FIRST (before applying any changes for this month)
        for portfolio, holdings in state.items():
            for ticker, info in holdings.items():
                annual_income_nominal = info["shares"] * info["div"]
                value_nominal = info["shares"] * info["price"]
                annual_income_real = annual_income_nominal / real_discount
                value_real = value_nominal / real_discount

                rows.append(
                    {
                        "Portfolio": portfolio,
                        "MonthIndex": m,
                        "YearIndex": year_index,
                        "Date": date,
                        "Ticker": ticker,
                        "Category": category_of(ticker),
                        "Shares": info["shares"],
                        "Price": info["price"],
                        "Value": value_nominal,
                        "Income": annual_income_nominal,
                        "RealValue": value_real,
                        "RealIncome": annual_income_real,
                        "MonthlyTotalContribution": monthly_total_contrib,
                    }
                )

        # Skip evolution on the last month (we just need to record the final state)
        if m == months:
            continue

        # Apply contributions, price evolution, and DRIP for months 0..months-1
        for portfolio, holdings in state.items():
            # Portfolio-level contribution this month
            port_contrib = monthly_total_contrib * portfolio_weights[portfolio]

            # Group tickers by category
            cat_tickers = {"Growth": [], "Stable": [], "HighYield": []}
            for ticker in holdings.keys():
                cat_tickers[category_of(ticker)].append(ticker)

            # Calculate effective weights - redistribute empty category weights proportionally
            active_cats = {cat: tickers for cat, tickers in cat_tickers.items() if tickers}
            if active_cats:
                # Sum of original weights for categories that have tickers
                active_weight_sum = sum(TARGET_CAT_WEIGHTS[cat] for cat in active_cats)
                # Scale weights so they sum to 1.0
                effective_weights = {
                    cat: TARGET_CAT_WEIGHTS[cat] / active_weight_sum
                    for cat in active_cats
                }
            else:
                effective_weights = {}

            # Allocate contributions by category, then equally by ticker
            for cat_name, tickers in active_cats.items():
                cat_weight = effective_weights[cat_name]
                cat_contrib = port_contrib * cat_weight
                per_ticker_contrib = cat_contrib / len(tickers)
                for ticker in tickers:
                    info = holdings[ticker]
                    info["shares"] += per_ticker_contrib / info["price"]

            # Monthly price + dividend evolution + DRIP
            for ticker, info in holdings.items():
                annual_price_growth = info["price_g"] + info["nav_g"]
                monthly_price_factor = (1 + annual_price_growth) ** (1 / 12)
                info["price"] *= monthly_price_factor

                # Dividend behaviour
                if ticker in {"HMAX", "SMAX"}:
                    # ±10%/year variability, smoothed monthly
                    noise_annual = RNG.uniform(-0.10, 0.10)
                    noise_month = noise_annual / 12
                    info["div"] *= (1 + noise_month)
                elif ticker == "DGS":
                    # Base negative drift plus ±15% variability
                    base_monthly_div_g = (1 + info["div_g"]) ** (1 / 12) - 1
                    noise_annual = RNG.uniform(-0.15, 0.15)
                    noise_month = noise_annual / 12
                    info["div"] *= (1 + base_monthly_div_g + noise_month)
                else:
                    # Deterministic dividend growth
                    monthly_div_g = (1 + info["div_g"]) ** (1 / 12) - 1
                    info["div"] *= (1 + monthly_div_g)

                # Monthly dividend DRIP
                monthly_div_cash = info["shares"] * (info["div"] / 12.0)
                info["shares"] += monthly_div_cash / info["price"]

    monthly_df = pd.DataFrame(rows)
    yearly_df = monthly_df[monthly_df["MonthIndex"] % 12 == 0].copy()
    return monthly_df, yearly_df


# -----------------------------
# ARCHIVE EXISTING FILES
# -----------------------------
def archive_existing_files():
    """
    Archives existing output files to an archive subdirectory.
    Removes old files in archive, then moves current output files there.
    Retries with pause if files cannot be deleted or moved (e.g., open in Excel).
    """
    archive_dir = OUTPUT_DIR / "archive"
    
    # Create archive directory if it doesn't exist
    archive_dir.mkdir(exist_ok=True)
    
    # Remove existing files in archive with retry
    for existing_file in archive_dir.glob("*.xlsx"):
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                existing_file.unlink()
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Warning: Cannot delete {existing_file.name} "
                          f"(may be open in Excel).")
                    input("Please close the file and press Enter to retry...")
                else:
                    print(f"Error: Failed to delete {existing_file.name} after "
                          f"{max_retries} attempts: {e}")
                    raise
    
    # Move current output files to archive with retry
    moved_count = 0
    for output_file in OUTPUT_DIR.glob("*.xlsx"):
        if output_file.is_file():
            dest = archive_dir / output_file.name
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    shutil.move(str(output_file), str(dest))
                    moved_count += 1
                    break
                except (PermissionError, OSError) as e:
                    if attempt < max_retries - 1:
                        print(f"Warning: Cannot move {output_file.name} "
                              f"(may be open in Excel).")
                        input("Please close the file and press Enter to retry...")
                    else:
                        print(f"Error: Failed to move {output_file.name} after "
                              f"{max_retries} attempts: {e}")
                        raise
    
    if moved_count > 0:
        print(f"Archived {moved_count} file(s) to {archive_dir}")


# -----------------------------
# RUN ALL SCENARIOS & SAVE
# -----------------------------
def main():
    # 0. Archive existing output files
    archive_existing_files()
    
    # 1. Load holdings snapshot from CSV
    snapshot = load_snapshot_from_file(HOLDINGS_CSV)

    # 2. Compute portfolio weights (for splitting total monthly contrib)
    portfolio_weights = compute_initial_portfolio_weights(snapshot)

    summary_rows = []

    for level in CONTRIBUTION_LEVELS:
        print(f"Running scenario: ${level}/month")

        monthly_df, yearly_df = run_realistic_projection(
            snapshot=snapshot,
            portfolio_weights=portfolio_weights,
            monthly_total_contrib=level,
            years=YEARS,
        )

        # Save per-scenario Excel
        xl_path = OUTPUT_DIR / f"portfolio_10yr_realistic_M{level}.xlsx"
        with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
            monthly_df.to_excel(writer, sheet_name="Monthly", index=False)
            yearly_df.to_excel(writer, sheet_name="Yearly", index=False)

        # Scenario-level Year-10 totals (YearIndex == 10)
        yr10 = yearly_df[yearly_df["YearIndex"] == YEARS]
        total_value = yr10["Value"].sum()
        total_income = yr10["Income"].sum()
        total_real_value = yr10["RealValue"].sum()
        total_real_income = yr10["RealIncome"].sum()

        summary_rows.append(
            {
                "MonthlyContribution": level,
                "Year10NominalValue": total_value,
                "Year10NominalIncome": total_income,
                "Year10RealValue": total_real_value,
                "Year10RealIncome": total_real_income,
            }
        )

    # Master summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_xlsx = OUTPUT_DIR / "portfolio_10yr_realistic_summary.xlsx"

    with pd.ExcelWriter(summary_xlsx, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)


if __name__ == "__main__":
    main()
