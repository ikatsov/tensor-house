from tools.database import Database
from tools.forecast import get_forecast
from streamlit.logger import get_logger


def stock_demand_difference(sku: str, location: str, weeks: int, db: Database) -> int:
    get_logger(__name__).info("stock_demand_difference")

    # Estimate the demand for the requested number of weeks
    demand = get_forecast(sku=sku, location=location).head(weeks)['demand'].sum()

    # Lookup the current stock level
    current_stock_df = db.query_inventory(
        f"SELECT quantity FROM INVENTORY WHERE sku='{{sku}}' AND location='{{location}}'"
    )['quantity']

    current_stock = 0 if current_stock_df.empty else current_stock_df.iloc[0]

    # Calculate the difference
    overage = current_stock - demand

    get_logger(__name__).info(f"Overage for sku [{sku}] in [{location}] over next [{weeks}] weeks: [{overage}]")

    return overage
