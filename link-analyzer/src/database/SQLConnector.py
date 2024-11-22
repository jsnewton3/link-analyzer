# Core imports
from time import sleep
import psycopg2
from psycopg2 import pool


class SQLConnector:
    def __init__(self, db_data):
        """
        Initialize an SQLConnector object.

        Args:
            pool_name (str): The name of the connection pool.
            pool_size (int): The size of the connection pool.
            host (str): The PostgreSQL server host.
            username (str): The PostgreSQL username.
            password (str): The PostgreSQL password.
            database (str): The PostgreSQL database name.
        """
        self.host = db_data.get("host", None)
        self.username = db_data.get("username", None)
        self.password = db_data.get("password", None)
        self.database = db_data.get("database", None)
        self.pool_name = db_data.get("pool_name", None)
        self.pool_size = db_data.get("pool_size", None)
        self.connection_pool = None

    def create_connection_pool(self):
        """
        Create the PostgreSQL connection pool.
        """
        self.connection_pool = pool.SimpleConnectionPool(
            self.pool_size,
            self.pool_size,
            host=self.host,
            user=self.username,
            password=self.password,
            database=self.database,
        )

    def get_connection(self, max_attempts=10, retry_interval=1):
        """
        Get a connection from the connection pool, waiting if necessary.

        Args:
            max_attempts (int, optional): The maximum number of attempts to get a connection. Defaults to 10.
            retry_interval (float, optional): The interval between connection attempts in seconds. Defaults to 0.5.

        Returns:
            psycopg2.extensions.connection: A connection object.
        """
        if self.connection_pool is None:
            self.connection_pool = self.create_connection_pool()

        for _ in range(max_attempts):
            try:
                connection = self.connection_pool.getconn()
                return connection
            except psycopg2.pool.PoolError:
                print("Connection pool is empty. Waiting for a connection...")
                sleep(retry_interval)

        raise Exception(f"Failed to get a connection after {max_attempts} attempts.")

    def close_connection_pool(self):
        """
        Close the connection pool and return all connections to it.
        """
        if self.connection_pool:
            self.connection_pool.closeall()
            self.connection_pool = None

    def connect_with_retry(self, max_attempts=5, retry_interval=5):
        """
        Connect to the PostgreSQL server with retry.

        Args:
            max_attempts (int, optional): The maximum number of connection attempts. Defaults to 5.
            retry_interval (int, optional): The interval between connection attempts in seconds. Defaults to 5.
        """
        attempt = 1
        while attempt <= max_attempts:
            try:
                self.create_connection_pool()
                return True
            except psycopg2.Error as e:
                print(
                    f"Failed to connect to PostgreSQL (Attempt {attempt}/{max_attempts}): {e}"
                )
                print(f"Retrying in {retry_interval} seconds...")
                sleep(retry_interval)
                attempt += 1

        print(f"Failed to establish a connection after {max_attempts} attempts.")
        return False

    def execute_query(self, query, params=None):
        """
        Execute an SQL query.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): The parameters for the query. Defaults to None.

        Returns:
            list: The result of the query.
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            if params is not None:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            result = cursor.fetchall() if cursor.description else []
            cursor.close()
            return result
        finally:
            if connection is not None:
                connection.commit()
                self.connection_pool.putconn(connection)

    def select_data(self, table, columns="*", where=None, where_values=None):
        """
        Select data from the table.

        Args:
            table (str): The name of the table.
            columns (str, optional): The columns to select. Defaults to "*".
            where (str, optional): The WHERE clause of the query. Defaults to None.
            where_values (tuple, optional): The values for the WHERE clause. Defaults to None.

        Returns:
            list: The selected data as a list of tuples.
        """
        query = f"SELECT {columns} FROM {table}"
        if where:
            query += f" WHERE {where}"

        return self.execute_query(query, where_values)

    def insert_data(self, table, data):
        """
        Insert data into the table.

        Args:
            table (str): The name of the table.
            data (list): The list of dictionaries representing the data to insert.
        """
        columns = data[0].keys()
        query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join(['%s'] * len(columns))})"

        for row in data:
            values = tuple(row.values())
            self.execute_query(query, values)

    def delete_data(self, table, where=None, where_values=None):
        """
        Delete data from the table.

        Args:
            table (str): The name of the table.
            where (str, optional): The WHERE clause of the query. Defaults to None.
            where_values (tuple, optional): The values for the WHERE clause. Defaults to None.
        """
        query = f"DELETE FROM {table}"
        if where:
            query += f" WHERE {where}"
        # print(query, where_values)
        self.execute_query(query, where_values)

    def update_data(self, table, set_values, where=None, where_values=None):
        """
        Update data in the table.

        Args:
            table (str): The name of the table.
            set_values (dict): The dictionary representing the column-value pairs to set.
            where (str, optional): The WHERE clause of the query. Defaults to None.
            where_values (tuple, optional): The values for the WHERE clause. Defaults to None.
        """
        set_columns = ", ".join([f"{column} = %s" for column in set_values.keys()])
        query = f"UPDATE {table} SET {set_columns}"
        if where:
            query += f" WHERE {where}"
        values = tuple(set_values.values())
        if where_values:
            values += tuple(where_values)
        self.execute_query(query, values)
