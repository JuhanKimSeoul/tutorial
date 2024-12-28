import logging
from datetime import datetime
from pathlib import Path

class CustomTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def __init__(self, filename, when='midnight', interval=1, backupCount=0, encoding=None, delay=False):
        filename = self.get_rotated_filename(filename)
        super().__init__(filename, when, interval, backupCount, encoding, delay)
    
    def get_rotated_filename(self, filename):
        # Generate a filename with the current date
        date_str = datetime.now().strftime('%Y_%m_%d')
        return str(Path(filename).parent / f'{Path(filename).name}_{date_str}.log')

    def doRollover(self):
        # Rotate the log file
        self.baseFilename = self.get_rotated_filename(self.baseFilename)
        super().doRollover()