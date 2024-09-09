import json
import socket
import struct

with open('dynamic_logconfig.json', 'r') as f:
    log_config = json.load(f)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(('localhost', 9999))
    bytes = json.dumps(log_config).encode('utf-8')
    sock.send(struct.pack('>L', len(bytes)))
    sock.send(bytes)