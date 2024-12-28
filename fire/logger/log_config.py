import json
import logging.config
import traceback

def log_config():
    '''
    load the log configuration from the logconfig.json file
    '''
    try:
        conf_thread = logging.config.listen(9999)
        conf_thread.start()
    except Exception as e:
        print(traceback.format_exc())
        raise e

log_config()