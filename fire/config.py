import traceback
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
import os
from helpers.helper import deep_extend
from kombu.utils.url import safequote
import logging.config
import json 
from threading import Thread, Lock
import logging

class SupportedExchanges(BaseModel):
    binance:dict = {
        'name' : 'binance',
        'options' : {
            'enableRateLimit' : False
        },
        'type' : 'linear',
        'quote' : 'USDT'
    }
    upbit:dict = {
        'name' : 'upbit',
        'options' : {
            'enableRateLimit' : False
        },
        'type' : 'spot',
        'quote' : 'KRW'
    }
    bithumb:dict = {
        'name' : 'bithumb',
        'options' : {
            'enableRateLimit' : False
        },
        'type' : 'spot',
        'quote' : 'KRW'
    }
    bybit:dict = {
        'name' : 'bybit',
        'options' : {
            'enableRateLimit' : False
        },
        'type' : 'linear',
        'quote' : 'USDT'
    }

class DevSettings(BaseSettings):
    # model_config = SettingsConfigDict(env_file=str(Path(__file__).parent / 'dev.env'), env_file_encoding='utf-8', cli_parse_args=True) # pytest와 충돌하기 때문에 cli_parse_args 테스트할 때는 임시 삭제
    # model_config = SettingsConfigDict(env_file=str(Path(__file__).parent / 'dev.env'), env_file_encoding='utf-8') # config파일과 env파일이 다른 위치에 있을 때, 단독으로 config.py를 실행시키면, env파일을 못찾아온다.
    model_config = SettingsConfigDict(env_file='dev.env', env_file_encoding='utf-8')

    aws_access_key_id: str
    aws_secret_access_key: str
    aws_redis_server_endpoint: str
    aws_redis_server_port: int
    aws_sqs_url: str
    aws_sqs_name: str
    app_host: str
    app_port: int
    jwt_secret_key: str
    jwt_algorithm: str
    user_db_url: str

    #const variables should be added here!
    supported_exchanges: SupportedExchanges = SupportedExchanges()

    #dynamic variables should be added here! 
    def dynamic_config(self):
        return {
            'env' : os.getenv('ENV') if os.getenv('ENV') else 'dev',
            'log_level' : os.getenv('LOG_LEVEL'),
            'redis_url' : f'redis://{self.aws_redis_server_endpoint}:{self.aws_redis_server_port}',
            'broker_url' : f'sqs://{safequote(self.aws_access_key_id)}:{safequote(self.aws_secret_access_key)}@'
        }

class TestSettings(DevSettings):
    model_config = SettingsConfigDict(env_file='test.env', env_file_encoding='utf-8')

class ProdSettings(DevSettings):
    model_config = SettingsConfigDict(env_file='prod.env', env_file_encoding='utf-8')

    
def get_config():
    '''
        singleton lazy-loading pattern
    '''
    global settings

    if settings is None:
        if os.getenv('ENV') == 'prod':
            settings = ProdSettings()
        elif os.getenv('ENV') == 'test':
            settings = TestSettings()
        else:
            settings = DevSettings()

        settings = deep_extend(settings.model_dump(), settings.dynamic_config())
    
    return settings

def start_log_config_listener_thread(settings)->Thread:
    with lock:
        with open('./logger/logconfig.json', 'r+') as f:
            log_config = json.load(f)
            log_config['loggers'][""]['level'] = settings['log_level']
            f.seek(0)
            f.write(json.dumps(log_config, indent=4))
            logger.info("Successfully write the log level input into logconfig.json file")

        logging.config.dictConfig(log_config)
        logger.info("log configuration set complete")

    try:
        logchg_thread = logging.config.listen(9999)
        logchg_thread.daemon = True # not a daemon process, it's a daemon thread
        logchg_thread.start()
        logger.info("new log file listening...")
    except Exception as e:
        print(traceback.format_exc())
        raise e

lock = Lock()
logger = logging.getLogger(__name__)
settings = None
    
if __name__ == '__main__':
    settings = get_config()
    thread = start_log_config_listener_thread(settings)