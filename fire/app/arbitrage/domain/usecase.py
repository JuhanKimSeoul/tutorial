from abc import ABC, abstractmethod

class ArbitrageUseCase(ABC):
    @abstractmethod
    async def get_kimchi_premium(self):
        pass