from src.skills.base import BaseSkill
from src.skills.faq_skill import FAQSkill
from src.skills.order_query_skill import OrderQuerySkill


_SKILLS: dict[str, BaseSkill] = {
    "faq_search": FAQSkill(),
    "order_query": OrderQuerySkill(),
}


def get_skill(skill_name: str) -> BaseSkill:
    """按名称获取已经注册的 Skill。"""

    if skill_name not in _SKILLS:
        raise ValueError(
            f"未注册的Skill：{skill_name}"
        )

    return _SKILLS[skill_name]


def list_skills() -> list[dict[str, str]]:
    """列出当前Agent拥有的全部Skill。"""

    return [
        {
            "name": skill.name,
            "description": skill.description,
        }
        for skill in _SKILLS.values()
    ]


if __name__ == "__main__":
    for skill in list_skills():
        print(skill)
