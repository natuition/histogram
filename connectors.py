#!/usr/bin/env python

import telnetlib
import re
import socket
import json


class SmoothieConnector:

    def __init__(self, host, verbose=False, allow_comments=False):
        self._host = host
        self._allow_comments = allow_comments
        self._verbose = verbose
        self._tn = None

    def connect(self):
        self._tn = telnetlib.Telnet(self._host)
        # read Smoothie startup prompt
        self._tn.read_until("Smoothie command shell".encode('ascii'))

    def disconnect(self):
        # send disconnect command
        self._tn.write("exit".encode('ascii') + b"\n")
        self._tn.read_all()
        self._tn.close()

    def send(self, command: str):
        if not self._allow_comments:
            command = re.sub("[ ]*;.*", '', command)  # remove everything after ;
            command = command.strip()  # send only the bare necessity

        if len(command) > 0:
            if self._verbose:
                print("Streaming g-code to " + self._host)

            self._tn.write(command.encode('ascii') + b"\n")

            if self._verbose:
                print("Sent code: " + command.strip())

    def receive(self):
        return self._tn.read_some().decode()


class PythonConnectorServer:

    def __init__(self, host, port, bufsize=4096):
        self._host = host
        self._port = port
        self._buffer_size = bufsize
        self._socket = socket.socket()
        self._socket.bind((self._host, self._port))
        self._incoming_connection = None
        self._incoming_address = None

    def wait_connection(self):
        self._socket.listen(1)
        self._incoming_connection, self._incoming_address = self._socket.accept()

    def close_connection(self):
        self._incoming_connection.close()

    def receive(self):
        received_data = self._incoming_connection.recv(self._buffer_size).decode()
        return json.loads(received_data) if received_data else None

    def send(self, data):
        self._incoming_connection.send(json.dumps(data).encode())


class PythonConnectorClient:

    def __init__(self, host, port, bufsize=4096):
        self._host = host
        self._port = port
        self._buffer_size = bufsize
        self._socket = socket.socket()

    def connect(self):
        self._socket.connect((self._host, self._port))

    def disconnect(self):
        self._socket.close()

    def send(self, data):
        self._socket.send(json.dumps(data).encode())

    def receive(self):
        received_data = self._socket.recv(self._buffer_size).decode()
        return json.loads(received_data) if received_data else None


def _test_SmoothieConnector():
    g_code = "G0 X10 Y5 F200"
    host = "192.168.1.222"
    sc = SmoothieConnector(host, verbose=True)
    sc.connect()
    print("Connection established. Sending data.")
    sc.send(g_code)
    while True:
        resp = sc.receive()
        if resp.count("ok".encode('ascii')) > 0:
            print("Response: " + resp)
            break
    sc.disconnect()
    print("Connection closed normally.")


if __name__ == "__main__":
    # _test_PythonConnectorServer()
    # _test_SmoothieConnector()
    pass
