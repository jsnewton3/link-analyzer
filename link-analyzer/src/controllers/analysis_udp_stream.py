import socket
import logging
logger = logging.getLogger(__name__)
import os
from dotenv import load_dotenv


class Destination_UDP(object):
    def __init__(self):
        # load_dotenv("../../../.env")
        self.server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        os.getenv('ANALYSIS_STREAM_BUFFER')
        self.buffer = int(os.getenv("ANALYSIS_STREAM_BUFFER"))
        self.host = "0.0.0.0"
        self.port = self.assign_anal_return_port()
        logger.info("Initiated client socket " + str((self.host, self.port)))

    # def bind_socket(self):


    def assign_anal_return_port(self):
        sock_max = os.getenv("ANALYSIS_STREAM_PORT_MAX")
        sock_min = os.getenv("ANALYSIS_STREAM_PORT_MIN")
        port = int(sock_min)
        while port <= int(sock_max):
            try:
                self.server_socket.bind((self.host, port))
                return port
            except OSError as e:
                port += 1
        raise IOError('no free ports')

Destination_UDP()