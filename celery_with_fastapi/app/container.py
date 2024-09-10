from dependency_injector.containers import DeclarativeContainer, WiringConfiguration
from dependency_injector.providers import Configuration, Factory, Singleton

from app.user.adapter.persistence.repository_adapter import UserRepositoryAdapter
from app.user.adapter.persistence.sqlalchemy.repositoryImpl import UserRepositoryImpl

class Container(DeclarativeContainer):
    wiring_config = WiringConfiguration(packages=["app"])

    user_repository = Singleton(UserRepositoryImpl)
    user_repository_adapter = Factory(UserRepositoryAdapter, repository=user_repository)