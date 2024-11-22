import logging
logger = logging.getLogger(__name__)
import threading
import time
from database.multithread_sql_connector import MultiThreadSqlConnector
from controllers.analysis_udp_stream import Destination_UDP
from psycopg2 import sql
import datetime
import numpy as np
import json


class Monitoring_Session(object):
    def __init__(self, analysis_obj, analysis_target_json,
                 db_connection_pool: MultiThreadSqlConnector):

        self.analysis_obj = analysis_obj
        self.db_targets = analysis_target_json
        # Calculate the required sleep time between database queries base on the query frequency provided by the
        # target json
        self.sleep_time = 1/self.db_targets['query_freq']
        print('Sleep times: ' + str(self.sleep_time))
        self.db_connection_err = None
        self.upd_connection_err = None
        self.current_query_time = None
        self.old_query_time = None

        # Create the database connection from the multi-threaded connection pool provided by psycopg2
        try:
            self.db_connection = db_connection_pool.get_connection_with_retries()
        except Exception as e:
            logging.error(e)
            self.upd_connection_err = e




    def start(self):

        # Set the current query data time and perform initial query to get the latest entry
        self.current_query_time = datetime.datetime.now()  #+ datetime.timedelta(hours=4)
        self.old_query_time = None
        data = np.array(self.construct_all_query())
        data = data[data[:, 0].argsort()[::-1]]
        self.data_to_analyze = []
        self.data_to_analyze.append(data[0, :])

        # Programmatically construct the recurrent query object based on the parameters passed in the
        # analysis target json.
        self.db_query = self.construct_recurrent_query()

        # start the recurrent query, analysis and send to udp thread
        # session_thread = threading.Thread(target=self.monitor_client_session, args=(session_id, client))
        # Create a udp connection over which analysis results will be streamed

        self.udp_stream = Destination_UDP()
        self.session_thread = threading.Thread(target=self.do_monitoring_task)
        self.session_thread.daemon = True
        self.session_thread.start()
        # self.train_window = None
        return self.udp_stream.port

    def do_monitoring_task(self):

        while True:
            addr_msg = self.udp_stream.server_socket.recvfrom(self.udp_stream.buffer)
            if addr_msg[0].decode('utf-8') == "Listening":
                self.client_port = addr_msg[1]
                logger.info("Message received from client on port " + str(self.client_port) + "Client is listening")
                print("client is listening")
                break

        anal_window = []
        while True:
            # set current and old query times
            self.old_query_time = self.current_query_time
            self.current_query_time = datetime.datetime.now() #+ datetime.timedelta(hours=4)
            # Get the cursor object from the connection
            cur = self.db_connection.cursor()

            # Execute the query with the time window parameters
            cur.execute(self.db_query, (self.old_query_time, self.current_query_time))

            # Fetch the data from the cursor object
            try:
                if not cur.description:
                    raise Exception("No recent data")
                data = cur.fetchall()
                for row in data:
                    if not data:
                        continue
                    row = [datetime.datetime.timestamp(row[0]), float(row[1])]
                    anal_window.append(row)

                time.sleep(self.sleep_time)
            except Exception as e:
                logging.exception("Unable to fetch data, or data is empty")
            if len(anal_window)>=self.analysis_obj.window:
                anal_window = np.array(anal_window)
                prediction = self.analysis_obj.predict(anal_window[:self.analysis_obj.window])
                anal_window = []
                # data =
                predict_jsons = [self.to_dict(row) for row in prediction]
                # self.search_socket.sendto(str.encode(json.dumps(data)), self.client_port)
                self.udp_stream.server_socket.sendto(str.encode(json.dumps(predict_jsons)), self.client_port)


    def to_dict(self, row):
        json_out = {}
        field_key = self.db_targets['field']
        time_key = "retrieved_at"
        json_out[time_key] = str(row[0])
        json_out[field_key] = float(row[1])
        return json_out




    def construct_recurrent_query(self):
        table = self.db_targets["table"]
        column = sql.Identifier(self.db_targets["field"])
        schema = self.db_targets["schema"]
        time_col_name = "retrieved_at"
        time_col = sql.Identifier(time_col_name)
        try:
            poop = (sql.SQL("SELECT {fields} FROM {schema}.{table} WHERE {retrieved_at} BETWEEN %s AND %s").
                   format(fields=sql.SQL(',').join([time_col, column]), schema=sql.Identifier(schema),
                          table=sql.Identifier(table), retrieved_at=time_col))
            return poop
        except Exception as e:
            logging.exception(e)


    def construct_all_query(self):
        table = self.db_targets["table"]
        column = sql.Identifier(self.db_targets["field"])
        schema = self.db_targets["schema"]
        print(schema)
        print(table)
        time_col_name =  "retrieved_at"
        time_col = sql.Identifier(time_col_name)
        try:
            query = sql.SQL("SELECT {fields} FROM {schema}.{table}").format(
                fields=sql.SQL(',').join([time_col, column]), schema=sql.Identifier(schema),
                table=sql.Identifier(table))
        except Exception as e:
            logging.exception(e)

        # Get the cursor object from the connection
        cur = self.db_connection.cursor()

        # Execute the query with the time window parameters
        cur.execute(query)

        # Fetch the data from the cursor object
        data = None
        try:
            if not cur.description:
                raise Exception("No recent data")
            data = cur.fetchall()
            return data
        except Exception as e:
            logging.exception("Unable to fetch data, or data is empty")

