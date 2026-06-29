from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import lru_cache

from pydantic import BaseModel, Field


class SkillExpansion(BaseModel):
    parent_skill: str
    expanded_skills: List[str]


class SkillExpansionResponse(BaseModel):
    skills: List[SkillExpansion]


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a resume skill expansion engine.

For each input skill:
- Expand it into closely related technical skills
- Keep hierarchy (parent -> children)
- Do NOT return unrelated skills

Return format must be STRICT JSON:
Each parent skill must map to a list of expanded skills.

Rules:
- Keep expansions realistic (no hallucinated unrelated fields)
- Do not repeat duplicates
- Keep lowercase
"""
    ),
    (
        "human",
        "Skills: {skills}"
    )
])


@lru_cache(maxsize=50)
def get_skill_expansion_chain(api_key):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )

    structured_llm = llm.with_structured_output(SkillExpansionResponse)

    chain = prompt | structured_llm

    return chain

def expand_jd_skills_llm(skills, api_key):
    """
    Expand JD skills using Gemini.

    Returns:
        {
            expanded_skill: parent_skill
        }
    """

    try:
        chain = get_skill_expansion_chain(api_key)

        result = chain.invoke({
            "skills": ", ".join(skills)
        })

        expanded_skill_map = {}

        jd_lower = {s.lower() for s in skills}

        for group in result.skills:

            parent = group.parent_skill.strip()

            for child in group.expanded_skills:

                child = child.strip().lower()

                # Don't include the original JD skill
                if child not in jd_lower:
                    expanded_skill_map.setdefault(child, parent)

        return expanded_skill_map

    except Exception as e:
        print(f"[Skill Expansion Error] {e}")

        return {}

