import json
from pydantic_settings import BaseSettings
import yaml
import os
import logging.config

ENV = os.getenv("ENV", "development")
DEBUGYN = os.getenv("DEBUG", "False") == "True"

class Settings(BaseSettings):
    DEBUG: bool = DEBUGYN
    API_KEY: str
    API_SECRET: str
    APP_HOST: str
    APP_PORT: int
    WRITER_DB_URL: str
    READER_DB_URL: str

    class Config:
        env_file = f".env.{ENV}"

class DevConfig(Settings):
    ...

class TestConfig(Settings):
    ...

class ProdConfig(Settings):
    DEBUG: bool = DEBUGYN

def load_yaml_config():
    '''
    for constants that are not sensitive, we can store them in a config.yaml file
    load the configuration from the config.yaml file
    '''
    with open("config.yaml") as f:
        return yaml.safe_load(f)
    
def load_log_config():
    '''
    load the log configuration from the logconfig.json file
    '''
    with open('./logger/logconfig.json', 'r') as f:
        log_config = json.load(f)

    logging.config.dictConfig(log_config)
    conf_thread = logging.config.listen(9999)
    conf_thread.start()

def get_config():
    if ENV == "development":
        return DevConfig()
    elif ENV == "test":
        return TestConfig()
    elif ENV == "production":
        return ProdConfig()
    else:
        raise ValueError(f"Invalid environment: {ENV}")

