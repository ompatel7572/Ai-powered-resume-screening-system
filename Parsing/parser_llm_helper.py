from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables

from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI



class Education(BaseModel):
    degree: str = Field(description="Degree, certification or qualification")
    location: str = Field(
        description="Institution, university, school or provider")
    result: str = Field(
        description="CGPA, GPA, percentage, grade or score. Return 'N/A' if unavailable.")


class Experience(BaseModel):
    job_title: str = Field(description="Job title")
    company: str = Field(description="Company or organization")
    duration: str = Field(
        description="Employment duration. Return 'N/A' if unavailable.")


class Reference(BaseModel):
    name: str = Field(
        description="Reference person's name. Return 'N/A' if unavailable.")
    designation: str = Field(
        description="Job title or designation. Return 'N/A' if unavailable.")
    organization: str = Field(
        description="Organization or company. Return 'N/A' if unavailable.")
    email: str = Field(
        description="Email address. Return 'N/A' if unavailable.")
    phone: str = Field(
        description="Phone number. Return 'N/A' if unavailable.")


class ResumeSections(BaseModel):
    education: List[Education]
    experience: List[Experience]
    references: List[Reference]


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert resume parser.

The input contains only the Education, Experience and References sections of a resume.

Extract the information exactly as written.

Rules:
- Do not invent missing information.
- If a field is unavailable, return "N/A".
- Preserve chronological order.
- Education includes degrees, certifications and academic qualifications.
- For education:
    - degree = qualification name
    - location = institution or provider
    - result = CGPA, GPA, percentage, grade or score Only send numbers with symbols
- For experience:
    - job_title = position
    - company = employer
    - duration = Formated employment dates 
- For References:
- Extract every reference listed.
- For each reference extract:
    - name
    - designation
    - organization
    - email
    - phone
- If the resume only says "References available upon request" or no references are present, return an empty list.
- Do not infer or fabricate any values..

"""
    ),
    (
        "human",
        """
Education

{education}

Experience

{experience}

References

{references}
"""
    )
])


@lru_cache(maxsize=100)
def get_chain(api_key: str):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )

    return prompt | llm.with_structured_output(ResumeSections)
# Initialize Gemini model



def extract_complex_details(chain, sectioned_dict_data):


    try:
        result = chain.invoke({
            "education": sectioned_dict_data.get("education", ""),
            "experience": sectioned_dict_data.get("experience", ""),
            "references": sectioned_dict_data.get("references", ""),
        })        
        return result.model_dump()

    except Exception as e:
        print(f"LLM extraction failed: {e}")

        return {
            "education": [],
            "experience": [],
            "references": []
        }
