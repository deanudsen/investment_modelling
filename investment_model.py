import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------
# CONFIG LOADING
# -----------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    """
    Validate configuration and warn about potential issues.

    Warns if any ticker in ticker_assumptions is not assigned to a category.
    """
    ticker_assumptions = config.get("ticker_assumptions", {})
    categories = config.get("categories", {})

    # Build set of all categorized tickers
    categorized = set()
    for cat_name, tickers in categories.items():
        categorized.update(tickers)

    # Check for uncategorized tickers
    for ticker in ticker_assumptions:
        if ticker not in categorized:
            logger.warning(
                f"Ticker '{ticker}' is in ticker_assumptions but not "
                f"assigned to any category. It will default to 'Stable'."
            )

    # Check for categorized tickers missing assumptions
    for ticker in categorized:
        if ticker not in ticker_assumptions:
            logger.warning(
                f"Ticker '{ticker}' is in categories but missing from "
                f"ticker_assumptions. This will cause an error if used."
            )


# -----------------------------
# BUILD CATEGORY SETS FROM CONFIG
# -----------------------------
def build_category_sets(config: dict) -> tuple[set, set, set]:
    """Build GROWTH, STABLE, HIGH_YIELD sets from config."""
    categories = config.get("categories", {})
    growth = set(categories.get("Growth", []))
    stable = set(categories.get("Stable", []))
    high_yield = set(categories.get("HighYield", []))
    return growth, stable, high_yield


def make_category_of(growth: set, stable: set, high_yield: set):
    """Create a category_of function with the given category sets."""
    def category_of(ticker: str) -> str:
        if ticker in growth:
            return "Growth"
        if ticker in stable:
            return "Stable"
        if ticker in high_yield:
            return "HighYield"
        return "Stable"
    return category_of


# -----------------------------
# BUILD SNAPSHOT FROM FILE
# -----------------------------
def load_snapshot_from_file(
    csv_path: str,
    ticker_assumptions: dict
) -> dict:
    """Load holdings snapshot from CSV file."""
    df = pd.read_csv(csv_path)

    # Make sure the columns we need exist
    required = {"Portfolio", "Ticker", "Quantity", "Price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in holdings file: {missing}"
        )

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
                f"Invalid Quantity for ticker '{ticker}' in portfolio "
                f"'{portfolio}': {shares}. Quantity must be positive."
            )
        if price <= 0:
            raise ValueError(
                f"Invalid Price for ticker '{ticker}' in portfolio "
                f"'{portfolio}': {price}. Price must be positive."
            )

        if ticker not in ticker_assumptions:
            raise ValueError(
                f"No assumptions entry in ticker_assumptions for ticker "
                f"'{ticker}'. Please add it to config.yaml."
            )

        assump = ticker_assumptions[ticker]

        # Validate ticker assumptions contain required keys
        required_keys = {"div", "price_g", "div_g", "nav_g"}
        missing_keys = required_keys - set(assump.keys())
        if missing_keys:
            raise ValueError(
                f"ticker_assumptions for '{ticker}' is missing required "
                f"keys: {missing_keys}"
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
    """Compute portfolio weights based on initial values."""
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
def run_realistic_projection(
    snapshot: dict,
    portfolio_weights: dict,
    monthly_total_contrib: float,
    config: dict,
    category_of,
    rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run projection simulation.

    Returns (monthly_df, yearly_df) for a given total monthly contribution.

    Uses:
      - realistic price growth per ticker
      - realistic dividend evolution (including variability for HMAX/SMAX/DGS)
      - monthly contributions with hybrid allocation (Growth/Stable/HighYield)
      - inflation-adjusted values @ 2% per year
    """
    years = config["years"]
    base_date = pd.Timestamp(config["base_date"])
    annual_inflation = config["annual_inflation"]
    target_cat_weights = config["category_weights"]

    # Deep copy snapshot so we don't mutate the original
    state = {
        p: {t: info.copy() for t, info in holdings.items()}
        for p, holdings in snapshot.items()
    }

    months = years * 12
    dates = pd.date_range(base_date, periods=months + 1, freq="MS")
    rows = []

    for m in range(months + 1):
        date = dates[m]
        year_index = m // 12  # 0..years
        real_discount = (1 + annual_inflation) ** (m / 12)

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

        # Skip evolution on the last month (we just need to record final state)
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

            # Calculate effective weights - redistribute empty category weights
            active_cats = {
                cat: tickers
                for cat, tickers in cat_tickers.items()
                if tickers
            }
            if active_cats:
                # Sum of original weights for categories that have tickers
                active_weight_sum = sum(
                    target_cat_weights[cat] for cat in active_cats
                )
                # Scale weights so they sum to 1.0
                effective_weights = {
                    cat: target_cat_weights[cat] / active_weight_sum
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
                    noise_annual = rng.uniform(-0.10, 0.10)
                    noise_month = noise_annual / 12
                    info["div"] *= (1 + noise_month)
                elif ticker == "DGS":
                    # Base negative drift plus ±15% variability
                    base_monthly_div_g = (1 + info["div_g"]) ** (1 / 12) - 1
                    noise_annual = rng.uniform(-0.15, 0.15)
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
def archive_existing_files(output_dir: Path, interactive: bool = True):
    """
    Archive existing output files to an archive subdirectory.

    Removes old files in archive, then moves current output files there.

    Args:
        output_dir: Path to the output directory
        interactive: If True, prompt user to retry on file errors.
                     If False, skip files that can't be accessed.
    """
    archive_dir = output_dir / "archive"

    # Create archive directory if it doesn't exist
    archive_dir.mkdir(exist_ok=True)

    # Remove existing files in archive with retry
    for existing_file in archive_dir.glob("*.xlsx"):
        max_retries = 3 if interactive else 1

        for attempt in range(max_retries):
            try:
                existing_file.unlink()
                break
            except (PermissionError, OSError) as e:
                if interactive and attempt < max_retries - 1:
                    logger.warning(
                        f"Cannot delete {existing_file.name} "
                        f"(may be open in Excel)."
                    )
                    input("Please close the file and press Enter to retry...")
                elif not interactive:
                    logger.warning(
                        f"Skipping delete of {existing_file.name}: {e}"
                    )
                    break
                else:
                    logger.error(
                        f"Failed to delete {existing_file.name} "
                        f"after {max_retries} attempts: {e}"
                    )
                    raise

    # Move current output files to archive with retry
    moved_count = 0
    for output_file in output_dir.glob("*.xlsx"):
        if output_file.is_file():
            dest = archive_dir / output_file.name
            max_retries = 3 if interactive else 1

            for attempt in range(max_retries):
                try:
                    shutil.move(str(output_file), str(dest))
                    moved_count += 1
                    break
                except (PermissionError, OSError) as e:
                    if interactive and attempt < max_retries - 1:
                        logger.warning(
                            f"Cannot move {output_file.name} "
                            f"(may be open in Excel)."
                        )
                        input(
                            "Please close the file and press Enter to retry..."
                        )
                    elif not interactive:
                        logger.warning(
                            f"Skipping archive of {output_file.name}: {e}"
                        )
                        break
                    else:
                        logger.error(
                            f"Failed to move {output_file.name} "
                            f"after {max_retries} attempts: {e}"
                        )
                        raise

    if moved_count > 0:
        logger.info(f"Archived {moved_count} file(s) to {archive_dir}")


# -----------------------------
# RUN ALL SCENARIOS & SAVE
# -----------------------------
def main(config_path: str = "config.yaml"):
    """Main entry point."""
    # Load and validate configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    # Extract config values
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)
    holdings_csv = config["holdings_csv"]
    contribution_levels = config["contribution_levels"]
    years = config["years"]
    interactive = config.get("interactive", True)

    # Build category sets and category_of function
    growth, stable, high_yield = build_category_sets(config)
    category_of = make_category_of(growth, stable, high_yield)

    # Initialize RNG with seed from config
    rng = np.random.default_rng(config["random_seed"])

    # Archive existing output files
    archive_existing_files(output_dir, interactive=interactive)

    # Load holdings snapshot from CSV
    logger.info(f"Loading holdings from {holdings_csv}")
    ticker_assumptions = config["ticker_assumptions"]
    snapshot = load_snapshot_from_file(holdings_csv, ticker_assumptions)

    # Compute portfolio weights (for splitting total monthly contrib)
    portfolio_weights = compute_initial_portfolio_weights(snapshot)

    summary_rows = []

    for level in contribution_levels:
        logger.info(f"Running scenario: ${level}/month")

        monthly_df, yearly_df = run_realistic_projection(
            snapshot=snapshot,
            portfolio_weights=portfolio_weights,
            monthly_total_contrib=level,
            config=config,
            category_of=category_of,
            rng=rng,
        )

        # Save per-scenario Excel
        xl_path = output_dir / f"portfolio_{years}yr_realistic_M{level}.xlsx"
        with pd.ExcelWriter(xl_path, engine="xlsxwriter") as writer:
            monthly_df.to_excel(writer, sheet_name="Monthly", index=False)
            yearly_df.to_excel(writer, sheet_name="Yearly", index=False)

        # Scenario-level Year-N totals (YearIndex == years)
        yr_final = yearly_df[yearly_df["YearIndex"] == years]
        total_value = yr_final["Value"].sum()
        total_income = yr_final["Income"].sum()
        total_real_value = yr_final["RealValue"].sum()
        total_real_income = yr_final["RealIncome"].sum()

        summary_rows.append(
            {
                "MonthlyContribution": level,
                f"Year{years}NominalValue": total_value,
                f"Year{years}NominalIncome": total_income,
                f"Year{years}RealValue": total_real_value,
                f"Year{years}RealIncome": total_real_income,
            }
        )

    # Master summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_xlsx = output_dir / f"portfolio_{years}yr_realistic_summary.xlsx"

    with pd.ExcelWriter(summary_xlsx, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    logger.info(f"All scenarios complete. Output saved to {output_dir}")


if __name__ == "__main__":
    main()
