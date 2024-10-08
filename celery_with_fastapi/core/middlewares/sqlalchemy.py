from uuid import uuid4
from starlette.types import ASGIApp, Scope, Receive, Send
from core.db.session import 

class SQLAlchemyMiddleware:
    def __init__(self, app:ASGIApp)->None:
        self.app = app

    def __call__(self, scope:Scope, receive:Receive, send:Send)->None:
        session_id = str(uuid4())
