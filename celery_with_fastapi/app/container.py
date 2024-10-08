from dependency_injector.containers import DeclarativeContainer, WiringConfiguration
from dependency_injector.providers import Configuration, Factory, Singleton

from app.user.adapter.persistence.repository_adapter import UserRepositoryAdapter
from app.user.adapter.persistence.sqlalchemy.repositoryImpl import UserRepositoryImpl
from app.user.application.service.user_service import UserService

class Container(DeclarativeContainer):
    wiring_config = WiringConfiguration(packages=["app"])

    user_repository = Singleton(UserRepositoryImpl)
    user_repository_adapter = Factory(UserRepositoryAdapter, repository=user_repository)
    user_service = Factory(UserService, repository=user_repository_adapter)