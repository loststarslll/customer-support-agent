from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """所有业务 Skill 的统一接口。"""

    name: str
    description: str

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """执行 Skill，并返回统一格式结果。"""
        raise NotImplementedError
