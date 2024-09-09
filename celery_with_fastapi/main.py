import os
import click
import uvicorn

@click.command()
@click.option(
    "--env",
    type=click.Choice(["development", "test", "production"]),
    default="development",
    help="The environment to run the server in",
)
@click.option(
    "--debug",
    type=click.BOOL,
    is_flag=True,
    default=False,
)
def main(env, debug):
    os.environ["ENV"] = env
    os.environ["DEBUG"] = str(debug)

    from config import get_config, load_log_config
    config = get_config()
    load_log_config()

    uvicorn.run(
        app="app.server:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=True if env == "development" else False,
        workers=1
    )
    
if __name__ == "__main__":
    main()