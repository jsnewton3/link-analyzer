import select
import psycopg2
import psycopg2.extensions
import asyncio
import os

def __init__(self, db_data):
    """
    Initialize an SQLConnector object.

    Args:
        host (str): The PostgreSQL server host.
        username (str): The PostgreSQL username.
        password (str): The PostgreSQL password.
        database (str): The PostgreSQL database name.
    """
    self.host = db_data.get("host", None)
    self.username = db_data.get("username", None)
    self.password = db_data.get("password", None)
    self.database = db_data.get("database", None)

cnct = psycopg2.connect(DSN)
cnct.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

cursor = cnct.cursor()
cursor.execute("LISTEN channel;")
print("Pending notification from 'channel'...")

def listener():
    while True:
        if select.select([cnct],[],[],5) == ([],[],[]):
            print("Channel timed out.")
        else:
            cnct.poll()
            while cnct.notifies:
                notif = cnct.notifies.pop(0)
                print("Received NOTIFY:", notif.pid, notif.channel, notif.payload)




