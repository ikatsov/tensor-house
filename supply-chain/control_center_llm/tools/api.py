def search_documents(question: str) -> str:
    """Knowledge database searcher

    Searches a collection of documents that describe the supply chain topology, supplier properties, and products,
    and returns a relevant answer or summary.

    Returns
    -------
    str
        Answer or summary
    """


def query_inventory(sql: str) -> pd.DataFrame:
    """Inventory database

    Converts the natural language question to a SQL query, executes its against a relational database with the
    current information about stock levels in each location, and returns the query result.

    The database contains only one table:
    INVENTORY(sku STRING, brand STRING, location STRING, quantity INTEGER)

    Returns
    -------
    pd.DataFrame
        Query result, pandas DataFrame
    """


def query_suppliers(sql: str) -> pd.DataFrame:
    """Database with available suppliers and procurement costs

    Converts the natural language question to a SQL query, executes it against a relational database with the
    current information about supply options and procurement costs, and returns the query result.

    The database contains only one table:
    SUPPLIERS(sku STRING, supplier STRING, location STRING, unit_cost FLOAT)

    Returns
    -------
    pd.DataFrame
        Query result, pandas DataFrame
    """


def get_shipping_cost(source: str, destination: str) -> float:
    """Carrier service

    Estimates per-unit shipping costs for delivering from the source location to destination location.

    Returns
    -------
    float
        Unit shipping cost in dollars
    """


def get_forecast(sku: str, location: str) -> pd.DataFrame:
    """Demand forecasting service

    Forecasts how many units of a given SKU will be sold in a given location. Returns forecasted weekly sales
    numbers, 52 weeks ahead.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame indexed by date with one column: 'demand'. 52 weeks ahead.
        Dates corresponds to Mondays of each week.
    """


def stock_demand_difference(sku: str, location: str, weeks: int) -> int:
    """Estimates the difference between the available stock and forecasted demand

    Computes the demand forecast for the requested number of weeks, and then computes the
    difference between the available stock and forecasted value

    Returns
    -------
    int
        Number of units. Positive value corresponds to overage (stock greater than the expected demand),
        negative value corresponds to shortage (expected demand greater than stock)
    """


def print_answer(answer: str):
    """Print answer or data for the supply chain manager

    Function that can be used to print a message, answer, or data for the user. The function can be called
    multiple times to print different parts of the full answer.
    """


def print_table(df: pd.Dataframe):
    """Print pandas DataFrame as table

    Function that can be used to print structured data for the user. The function can be called
    multiple times to print different parts of the full answer.
    """


def show_line_chart(data: pd.DataFrame):
    """Display a line chart

    Visualizes a pandas DataFrame with two columns: date and value
    """
