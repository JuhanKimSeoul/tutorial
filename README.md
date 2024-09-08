# Tutorial

## 1. Get used to Flask

- ✅ Make a simple HTTP Restful API Service with MVC pattern using Python, Flask
- ☑️ Adapt pytest and get used to it by using it on the project above(HTTP Restful API Service with MVC pattern)





## 2. How to set logger in production

### Motivation

When operating a real service, there occurs several bugs and the developers should fix the bug while still running the app.
In order to fix the bug, they should have to debug the services and it is hard to dynamically change the log level or filter some logs.
And in most cases for production mode, we have to erase the unnecessary logging to be stored for efficiency and this makes it harder to
find the bugs. It is a good time to use a configuration server for the real services!

### Features

- ✅ logging with multiple threads
- ✅ dynamically change the log configuration not interrupting the main app(service) running.

```python
# main.py
import logging.config
logging.config.listen(9999)
```

```python
# client.py
LOGGING_CONFIG = {
    ...
}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.connect(('localhost', 9999))
    bytes = json.dumps(LOGGING_CONFIG).encode('utf-8')
    sock.send(struct.pack('>L', len(bytes)))
    sock.send(bytes)
```

```bash
2024-09-07 18:51:37,531 - test1 - INFO - Logging info at regular intervals.
2024-09-07 18:51:38,530 - test2 - INFO - Logging info at regular intervals.
2024-09-07 18:51:38,536 - test1 - INFO - Logging info at regular intervals.
2024-09-07 18:51:39,535 - test2 - INFO - Logging info at regular intervals.
2024-09-07 18:51:39,540 - test1 - INFO - Logging info at regular intervals.
2024-09-07 18:51:40,537 - test2 - INFO - Logging info at regular intervals.
2024-09-07 18:51:40,543 - test1 - INFO - Logging info at regular intervals.
2024-09-07 18:51:41,545 - test1 - DEBUG - Logging debug at regular intervals. # dynamically change the configuration.
2024-09-07 18:51:41,545 - test1 - INFO - Logging info at regular intervals. # main.py still running, only test1.py has been recorded and also different logging level.
2024-09-07 18:51:42,551 - test1 - DEBUG - Logging debug at regular intervals.
2024-09-07 18:51:42,551 - test1 - INFO - Logging info at regular intervals.
2024-09-07 18:51:43,555 - test1 - DEBUG - Logging debug at regular intervals.
2024-09-07 18:51:43,555 - test1 - INFO - Logging info at regular intervals.
2024-09-07 18:51:44,556 - test1 - DEBUG - Logging debug at regular intervals.
2024-09-07 18:51:44,556 - test1 - INFO - Logging info at regular intervals.
2024-09-07 18:51:45,560 - test1 - DEBUG - Logging debug at regular intervals.
2024-09-07 18:51:45,560 - test1 - INFO - Logging info at regular intervals.
```

⚠️ if you set and get the typical logger using \_\_name\_\_, it will not go well when dynamically changing the logger because \_\_name\_\_ includes the parent directory. This makes tricky if you are going to still use that. You will see the main.py will stop when sending the socket message from the client to change the config. It will be much easier to use user-defined logger name like just 'test1'.

```python
logger = logging.getLogger(__name__) # logger : test_modules.test1

logger = logging.getLogger('test1') # use user-defined name instead
```

### log.config.fileconfig 🆚 log.config.dictconfig

- log.config.fileconfig : can't dynamically make instance of user-defined filter(function or class) to adapt into the logger. You can see and check the client_fileconfig.py.

- log.config.dictconfig : can dynamically make instance of user-defined filter(function or class) to adapt into the logger. You can see and check the client_dictconfig.py.

- If you still want to use file type, then you can make a json file and use json.load. In that way, you can use file so that your source code doesn't need to be changed and also dynamically change the logger configuration.






## 3. fastapi app incorporated with celery

### overview

Seperate the restful web api(FastAPI) service from constantly running backend server which is for requesting and accepting the data from the another API server. However we don't need the another backend server. The celery can replace it especially with this case of program which is running an infinite scheduled program. As we configure the broker and backend server(DB), the celery(beat) should send a scheduled task to a broker(rabbitmq) and then the broker convey it to an allocated celery(worker) to make it. This flow can be visualized and well-managed by flower packages which provides a web page to easily change the configuration dynamically such as worker pool, task rate limit and etc.

### main packages

- fastapi : for web application
- celery : replacing a backend server to process tasks independent of the main application
- flower : visualize and manage the whole celery process
- uvicorn : ASGI for performance and handling concurrent requests

### main techniques

- redis
- rabbitmq

### running fastapi web application

```python
uvicorn main:app --reload # running fastapi web application
```

### connect the flower app to celery and broker

```python
celery -A core flower --broker=amqp://localhost # running flower app, core is the name of file in which celery app located.
# fastapi web server runs at default 8000 port
```

```bash
# the results
# default port : 5555
INFO:flower.command:Visit me at http://0.0.0.0:5555
INFO:flower.command:Broker: amqp://guest:**@localhost:5672//
INFO:flower.command:Registered tasks:
['celery.accumulate',
 'celery.backend_cleanup',
 'celery.chain',
 'celery.chord',
 'celery.chord_unlock',
 'celery.chunks',
 'celery.group',
 'celery.map',
 'celery.starmap',
 'core.fetch_market_data',
 'core.fetch_orderbook_data']
```

### running a celery worker

```python
celery -A core worker --loglevel=info --logfile=./worker.log
```

```bash
 -------------- celery@Kims-MacBook-Pro.local v5.4.0 (opalescent)
--- ***** -----
-- ******* ---- macOS-14.6.1-arm64-arm-64bit 2024-09-09 00:55:10
- *** --- * ---
- ** ---------- [config]
- ** ---------- .> app:         tasks:0x1085a5210
- ** ---------- .> transport:   amqp://guest:**@localhost:5672//
- ** ---------- .> results:     redis://localhost:6379/0
- *** --- * --- .> concurrency: 12 (prefork)
-- ******* ---- .> task events: OFF (enable -E to monitor tasks in this worker)
--- ***** -----
 -------------- [queues]
                .> celery           exchange=celery(direct) key=celery


[tasks]
  . core.fetch_market_data
  . core.fetch_orderbook_data
```

### running a celery scheduler

```python
celery -A core beat --loglevel=info --logfile=./beat.log
```

```bash
celery beat v5.4.0 (opalescent) is starting.
__    -    ... __   -        _
LocalTime -> 2024-09-09 00:55:26
Configuration ->
    . broker -> amqp://guest:**@localhost:5672//
    . loader -> celery.loaders.app.AppLoader
    . scheduler -> celery.beat.PersistentScheduler
    . db -> celerybeat-schedule
    . logfile -> ./beat.log@%INFO
    . maxinterval -> 5.00 minutes (300s)
```
