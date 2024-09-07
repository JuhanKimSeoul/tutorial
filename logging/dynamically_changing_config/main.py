from test_modules.test1 import Test1
from test_modules.test2 import Test2
import logging.config
from pathlib import Path

def filter_maker(logger):
    def filter(record):
        return record.name == logger
    return filter

if __name__ == '__main__':
    f = Path(__file__).parent / 'original_config.ini'
    logging.config.fileConfig(str(f))
    conf_thread = logging.config.listen(9999)
    conf_thread.start()
    
    test1 = Test1()
    test2 = Test2()
    test1.run()
    test2.run()