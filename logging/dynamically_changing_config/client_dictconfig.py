import json
import socket
import struct

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    
    'loggers': {
        'root': {
            'level': 'DEBUG',
            'handlers': ['consoleHandler']
        }
    },

    'filters': {   
        'specificLoggerFilter': {
            '()' : '__main__.filter_maker',
            'logger': 'test1'
        }
    },

    'handlers': {
        'consoleHandler': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simpleFormatter',
            'stream': 'ext://sys.stdout',
            'filters': ['specificLoggerFilter']
        }
    },

    'formatters': {
        'simpleFormatter': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(('localhost', 9999))
    bytes = json.dumps(LOGGING_CONFIG).encode('utf-8')
    sock.send(struct.pack('>L', len(bytes)))
    sock.send(bytes)