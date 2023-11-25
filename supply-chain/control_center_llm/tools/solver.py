from tools.database import Database
from tools.forecast import get_forecast


def stock_demand_difference(sku: str, location: str, weeks: int, db: Database) -> int:
    # Estimate the demand for the requested number of weeks
    demand = get_forecast(sku=sku, location=location).head(weeks)['demand'].sum()
    # Lookup the current stock level
    current_stock_df = db.query_inventory(f"SELECT quantity FROM INVENTORY WHERE sku='{{sku}}' AND location='{{location}}'")['quantity']
    current_stock = 0 if current_stock_df.empty else current_stock_df.iloc[0]
    # Calculate the difference
    overage = current_stock - demand
    return overage
