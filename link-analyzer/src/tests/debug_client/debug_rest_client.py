import sys
import json
import requests
import socket

def listen(port, buffer, client_socket):
    print("Listening on " + str(port))
    while True:
        addr_msg = client_socket.recvfrom(buffer)
        msg = addr_msg[0]
        print(str(msg))

with open(
        "/home/joseph.newton@rdte.nswc.navy.mil/Documents/ath4/athena-local-development/link-analyzer/src/tests/test_jsons/lstm_config.json") as f:
    test_jsons = json.load(f)
a=1
print(test_jsons)
res = requests.put("http://localhost:4001/monitor/config", json=test_jsons)
uuid_str = res.json()['uuid']
start_req_json = res.json()
res_start = requests.get("http://localhost:4001/monitor/start", json=start_req_json)

port = res_start.json()['port']
client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
buffer = 65536
addr = ("127.0.0.1", port)
# socket.con
client_socket.sendto(str.encode("Listening"), addr)
listen(port, buffer, client_socket)