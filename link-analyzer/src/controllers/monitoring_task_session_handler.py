from dotenv import load_dotenv
import os
import uuid
import inspect
from models import analyses
import logging
logger = logging.getLogger(__name__)
import time
import traceback
import importlib
from controllers.analysis_session import Monitoring_Session
from database.multithread_sql_connector import MultiThreadSqlConnector



class Monitor_Task_Session_Handler(object):
    def __init__(self,):
        self._err = None
        load_dotenv()
        logs_path = os.getenv("ANALYSIS_SERVER_LOG_PATH")
        log_dir_path = os.path.join(logs_path, 'analysis_server_logs')
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
        log_filename = "server_log_" + time.strftime("%m-%d-%Y_%H:%M:%S")
        log_filepath = os.path.join(log_dir_path, log_filename)
        logging.basicConfig(filename=log_filepath, encoding='utf-8', level=logging.INFO)
        self.link_monitor_sessions = {}
        self._analysis_type_dict = self.inspect_available_analyses()
        # Start the database multithreaded connection pool
        self.database_connector_pool = MultiThreadSqlConnector()

    @property
    def analysis_type_dict(self):
        self._err = None
        # self.err = None
        return self._analysis_type_dict

    def inspect_available_analyses(self):

        '''

        Returns: A dict of class name strings and class type value corresponding to the currently available monitoring
        processes.
        '''
        self._err = None
        # self.err = None
        available_analyses = {}
        # Inspect available class available in the analyses' module. Loads the class and stores it in a dynamic Enum
        # with class name string key.
        try:
            for name, class_obj in inspect.getmembers(analyses):
                if inspect.isclass(class_obj) and class_obj.__module__=='models.analyses':
                    moule_class_str = class_obj.__module__.split('.', 1)[1]
                    anal_mod = __import__(class_obj.__module__)
                    anal_mod = importlib.import_module(class_obj.__module__)
                    anal_class_ = getattr(anal_mod, class_obj.__name__)
                    available_analyses[class_obj.__name__] = anal_class_
            # Creates an enum for type checking

            return available_analyses
        except Exception as e:
            tb = traceback.format_exc()
            msg = 'Error checking for analysis types'
            self.handle_error(msg, tb, 500)
            self._err = msg

    def add_monitor_task(self, monitor_type, config_json, target_json):
        self._err = None
        # self.err = None
        try:
            monitor_class = self._analysis_type_dict[monitor_type]
            # Creates the analysis object from the class type stored
            monitor_obj = monitor_class.deserialize(config_json)
            # Stores the analysis object in a dict with a unique uuid key
            session_id = str(uuid.uuid4())
            session = Monitoring_Session(monitor_obj, target_json, self.database_connector_pool)
            self.link_monitor_sessions[session_id] = session
            return {'uuid': session_id}
        except KeyError as k:
            tb = traceback.format_exc()
            msg = 'Invalid analysis type'
            self.handle_error(msg, tb, 406, k)
            # self._err = msg

        except TypeError as k:
            tb = traceback.format_exc()
            msg = "Error initializing Arima filter. Invalid config parameters"
            self.handle_error(msg, tb, 406, err=k)
            # self._err = msg

        except Exception as e:
            tb = traceback.format_exc()
            msg = "Unanticipated error"
            self.handle_error(msg, tb, 400, err=e)
            # self._err = msg

    def start_monitoring(self, uuid):
        self._err = None
        session = self.link_monitor_sessions[uuid]
        try:
            listen_port = session.start()
        except Exception as e:
            msg = "Error starting UDP stream",
            tb = traceback.format_exc()
            code = 500
            self.handle_error(msg=msg, tb=tb, code=code, err=e)
        else:
            return {'port': listen_port}





    def handle_error(self, msg, tb, code, err ):
        err_json = {'msg':msg, 'code': code,'tb':tb}
        logging.exception(err)
        self._err = err_json




# if __name__ == "__main__":
    # a = Monitor_Task_Session_Handler()
    # with open('/home/phat/PycharmProjects/athena-local/link_analyzer/src/network-monitor-server/tests/test_jsons/arima_config') as file:
    #     ariam_config = json.load(file)
    # b= a._analysis_type_dict
    # aa = a.add_monitor_task('Arima', ariam_config['config'])
    # c = a._err

