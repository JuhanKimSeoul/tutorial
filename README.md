# Tutorial

## Get used to Flask

- ✅ Make a simple HTTP Restful API Service with MVC pattern using Python, Flask
- ☑️ Adapt pytest and get used to it by using it on the project above(HTTP Restful API Service with MVC pattern)

## How to set logger in production

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

## Get used to Spring(Java)

- ☑️ Make a simple shoppingmall application using Spring, Java
