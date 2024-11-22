from time import sleep
import psycopg2
from psycopg2 import pool
import os


class MultiThreadSqlConnector:
    def __init__(self, min_connections=5, max_connections=20, retries = 5):
        self.retries = retries
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=min_connections,
            maxconn=max_connections,
            dbname=os.getenv("DATABASE"),
            user='root',
            password=os.getenv("PASSWORD"),
            host=os.getenv("DB_HOST")
        )

    def get_connection_with_retries(self):

        retry_delay = 1  # seconds

        for retry in range(self.retries):
            try:
                conn = self.pool.getconn()
                return conn
            except Exception as err:
                print(f"Connection attempt {retry + 1} failed: {err}")
                sleep(retry_delay)

        raise Exception("Max retries reached. Unable to get a connection.")
