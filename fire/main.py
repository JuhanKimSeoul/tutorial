import json
import os
import click
import uvicorn
from config import get_config, start_log_config_listener_thread
import logging
import asyncio
import signal

logger = logging.getLogger(__name__)
settings = None

@click.command()
@click.option(
    "--env",
    type=click.Choice(["dev", "test", "prod"]),
    default="dev",
    help="The environment to run the server in",
)
@click.option(
    "--loglevel",
    type=click.Choice(['INFO', 'DEBUG']),
    default='DEBUG',
    help="log level"
)
def main(env, loglevel):
    global settings

    os.environ["ENV"] = env
    os.environ["LOG_LEVEL"] = str(loglevel)

    settings = get_config()
    start_log_config_listener_thread(settings) 
    logger.info('app configuration :')
    logger.info(json.dumps(settings, indent=4)) 

    config = uvicorn.Config(
        app="app.server:app",
        host=settings['app_host'],
        port=settings['app_port'],
        reload=False,
        # reload=True if env == "development" else False,
        workers=1,
        lifespan="on"
    )

    server = uvicorn.Server(config)

    loop = asyncio.get_event_loop()

    loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(handle_shutdown(server)))
    loop.add_signal_handler(signal.SIGTSTP, lambda: asyncio.create_task(handle_shutdown(server)))

    loop.run_until_complete(server.serve())

async def handle_shutdown(server):
    logger.info("Signal Caught, shutting down app gracefully...")
    await server.shutdown()

if __name__ == "__main__":
    main()