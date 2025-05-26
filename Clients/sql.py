from typing import List, Tuple
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import connection as Connection
from psycopg2.extras import Json


def convert_numpy(obj):
    if isinstance(obj, dict):
        return Json({k: convert_numpy(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class SqlClient:
    def __init__(
        self,
        host="localhost",
        database="mydb",
        user="myuser",
        password="mysecretpassword",
        port=5432,
    ):
        self.conn_params = {
            "host": host,
            "database": database,
            "user": user,
            "password": password,
            "port": port,
        }
        self.conn = None
        self.cur = None

    def __enter__(self):
        self.conn = psycopg2.connect(**self.conn_params)
        self.cur = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cur:
            self.cur.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()  # Commit if no exception
            else:
                self.conn.rollback()  # Rollback on error
            self.conn.close()

    def execute(self, query, params=None):
        self.cur.execute(query, params or ())

    def insert_batch(self, table: str, columns: List[str], rows: List[Tuple[any]]):
        query = f"INSERT INTO {table} ({','.join(columns)}) VALUES %s"
        execute_values(self.cur, query, [[convert_numpy(c) for c in r] for r in rows])

    def fetchall(self):
        return self.cur.fetchall()

    def fetchone(self):
        return self.cur.fetchone()
