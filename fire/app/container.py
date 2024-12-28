from dependency_injector.containers import DeclarativeContainer, WiringConfiguration
from dependency_injector.providers import Factory, Singleton

from app.arbitrage.application.service import ArbitrageService
from app.user.persistence.sqlalchemy.user import UserSqlalchemyRepo
from app.user.persistence.repository_adapter import UserRepoAdapter
from app.user.application.service import UserService

class Container(DeclarativeContainer):
    wiring_config = WiringConfiguration(packages=['app'])

    user_repo = Singleton(UserSqlalchemyRepo)
    user_repo_adapter = Factory(UserRepoAdapter, user_repo=user_repo)
    user_service = Factory(UserService, repository=user_repo_adapter)

    arbitrage_service = Factory(ArbitrageService)