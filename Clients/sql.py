from typing import List, Tuple
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import connection as Connection
from psycopg2.extensions import cursor as Cursor

from util import convert_numpy


# SqlClient will hold a database connection (in a context) for fast access
# 
# Usage example:
#
# with SqlClient() as client:
#     client.execute("SELECT * FROM my_table WHERE id = %s", (1,))
#
# It is common to also use transactions if a batch of operations needs to be atomic.
#
# Transaction usage example:
#
# with SqlClient() as client:
#     with SqlTransaction(client) as transaction:
#         client.insert_batch("my_table", ["col1", "col2"], [(1, "a"), (2, "b")])
#         client.insert_batch("my_table", ["col1", "col2"], [(3, "c"), (4, "d")])
#
# In the above example if either of the insert_batch calls fail, no data will be inserted into the database.
#
class SqlClient:
    conn: Connection
    cur: Cursor

    def __init__(
        self,
        host="localhost",
        database="mydb",
        user="myuser",
        password="mysecretpassword",
        port=5432
    ):
        self.conn_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }
        self.conn = None


    def __enter__(self):
        self.conn = psycopg2.connect(**self.conn_params)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.cur:
            self.cur.close()

        if self.conn:
            self.conn.close()

    def execute(self, query, params=None):
        self.cur.execute(query, params or ())

    def insert_batch(self, table: str, columns: List[str], rows: List[Tuple[any]]):
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES %s"
        execute_values(self.cur, query, [[convert_numpy(c) for c in r] for r in rows])


class SqlTransaction:
    client: SqlClient
    
    def __init__(self, client: SqlClient):
        self.client = client

    def __enter__(self):
        self.client.cur.execute("BEGIN")
  
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None:
            self.client.conn.commit()  # Commit if no exception
        else:
            self.client.conn.rollback()  # Rollback on error

