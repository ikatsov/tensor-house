import pandas as pd
import sqlite3
from streamlit.logger import get_logger


class Database:
    INVENTORY_TABLE = "inventory"
    SUPPLIER_TABLE = "suppliers"

    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        self.anchor_connection = sqlite3.connect('file::memory:?cache=shared')

        for table_name in [self.INVENTORY_TABLE, self.SUPPLIER_TABLE]:
            df = pd.read_csv(f"{config['data_folder']}/{table_name}.csv")
            self.logger.info(f"Loaded table [{table_name}] with [{df.shape[0]}] rows")
            df.to_sql(table_name, self.anchor_connection, if_exists='replace', index=False)

    def query_inventory(self, sql: str) -> pd.DataFrame:
        return self._read_sql(sql)

    def query_suppliers(self, sql: str) -> pd.DataFrame:
        return self._read_sql(sql)

    def _read_sql(self, sql: str) -> pd.DataFrame:
        self.logger.info(f"Executing SQL [{sql}]")
        try:
            connection = sqlite3.connect('file::memory:?cache=shared')
            df = pd.read_sql(sql, connection)
        except Exception as e:
            self.logger.error(e)

        self.logger.info(f"SQL execution result:\n{df.to_string()}")
        return df