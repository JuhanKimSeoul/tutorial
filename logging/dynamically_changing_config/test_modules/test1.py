import logging
import logging.config
import threading
from pathlib import Path
import time

class Test1:
    f_name = None
    f_dir = None
    f = None
    logger = None

    def __init__(self):
        self.f_name = 'logging_config_for_test1.ini'
        self.f_dir = Path(__file__)
        self.f = self.f_dir.parent.parent / self.f_name
        self.logger = logging.getLogger('test1')
        self.logger.debug(f"Initial configuration loaded : {self.f_name} and logger started.")

    def start_logging_server(self):
        try:
            while True:
                self.logger.debug("Logging debug at regular intervals.")
                self.logger.info("Logging info at regular intervals.")
                # Simulating work with a delay
                time.sleep(1)

        except KeyboardInterrupt:
            # Stop the logging server when the program is interrupted
            logging.config.stopListening()
            self.logger.info("Logging server stopped.")

    def run(self):
        logging_thread = threading.Thread(target=self.start_logging_server) 
        logging_thread.start()

        