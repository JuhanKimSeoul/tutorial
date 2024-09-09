import logging
from datetime import datetime

class CustomTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def __init__(self, filename, when='midnight', interval=1, backupCount=0, encoding=None, delay=False):
        self.baseFilename = filename
        super().__init__(filename, when, interval, backupCount, encoding, delay)
    
    def get_rotated_filename(self):
        # Generate a filename with the current date
        date_str = datetime.now().strftime('%Y_%m_%d')
        return f"{date_str}_{self.baseFilename}"

    def doRollover(self):
        # Rotate the log file
        self.baseFilename = self.get_rotated_filename()
        super().doRollover()