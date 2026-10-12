"""Loopback-only, EBI-only opaque TLS CONNECT relay for an SSH reverse tunnel.

This is a one-shot extraction transport, not a system proxy. TLS certificates
are still validated by the original requests client on the server. No browser
cookies/tokens, HTTP contents, or credentials are inspected or logged.
"""
import argparse
import select
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class RelayServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = False


class RelayHandler(BaseHTTPRequestHandler):
    allowed_hosts = {'ftp.ebi.ac.uk', 'www.ebi.ac.uk'}
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_CONNECT(self):
        try:
            host, port = self.path.rsplit(':', 1)
            if host.lower() not in self.allowed_hosts or int(port) != 443:
                self.send_error(403, 'Only official EBI TLS transport is allowed')
                return
            upstream = socket.create_connection((host, 443), timeout=20)
            upstream.settimeout(None)
            self.send_response(200, 'Connection established')
            self.end_headers()
            peers = [self.connection, upstream]
            last_data = time.monotonic()
            with upstream:
                while time.monotonic() - last_data < 120:
                    readable, _, exceptional = select.select(peers, [], peers, 10)
                    if exceptional:
                        return
                    for source in readable:
                        payload = source.recv(256*1024)
                        if not payload:
                            return
                        target = upstream if source is self.connection else self.connection
                        target.sendall(payload)
                        last_data = time.monotonic()
        except (OSError, ValueError):
            self.close_connection = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=17888)
    ap.add_argument('--hours', type=float, default=12)
    args = ap.parse_args()
    if not 1024 <= args.port <= 65535 or args.hours <= 0:
        raise ValueError('Invalid loopback relay limits')
    server = RelayServer(('127.0.0.1', args.port), RelayHandler)
    deadline = threading.Timer(args.hours*3600, server.shutdown)
    deadline.daemon = True
    deadline.start()
    print(f'EBI_LOOPBACK_RELAY port={args.port} max_hours={args.hours}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
