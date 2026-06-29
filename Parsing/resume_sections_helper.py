from rapidfuzz import process, fuzz


CANONICAL_SECTIONS = {

    "summary": [
        "SUMMARY",
        "PROFESSIONAL SUMMARY",
        "CAREER SUMMARY",
        "PROFILE",
        "PROFESSIONAL PROFILE",
        "EXECUTIVE SUMMARY",
        "ABOUT ME",
        "ABOUT",
        "OBJECTIVE",
        "CAREER OBJECTIVE",
        "PROFESSIONAL OBJECTIVE",
        "PERSONAL PROFILE",
        "INTRODUCTION",
    ],

    "skills": [
        "SKILLS",
        "TECHNICAL SKILLS",
        "CORE SKILLS",
        "KEY SKILLS",
        "PROFESSIONAL SKILLS",
        "HARD SKILLS",
        "SOFT SKILLS",
        "TECHNICAL EXPERTISE",
        "CORE COMPETENCIES",
        "COMPETENCIES",
        "EXPERTISE",
        "TECHNOLOGIES",
        "TOOLS",
        "TOOLS & TECHNOLOGIES",
        "TECH STACK",
        "TECHNICAL PROFICIENCIES",
        "PROFICIENCIES",
        "AREAS OF EXPERTISE",
        "PROGRAMMING LANGUAGES",
    ],

    "education": [
        "EDUCATION",
        "ACADEMIC BACKGROUND",
        "ACADEMIC HISTORY",
        "ACADEMIC QUALIFICATIONS",
        "ACADEMIC DETAILS",
        "EDUCATIONAL BACKGROUND",
        "EDUCATIONAL QUALIFICATIONS",
        "EDUCATIONAL DETAILS",
        "QUALIFICATIONS",
        "SCHOLASTIC DETAILS",
        "SCHOLASTIC BACKGROUND",
        "ACADEMICS",
        "DEGREES",
        "FORMAL EDUCATION",
        "UNIVERSITY EDUCATION",
        "SCHOOLING",
    ],

    "experience": [
        "EXPERIENCE",
        "WORK EXPERIENCE",
        "PROFESSIONAL EXPERIENCE",
        "EMPLOYMENT HISTORY",
        "EMPLOYMENT",
        "WORK HISTORY",
        "CAREER HISTORY",
        "CAREER EXPERIENCE",
        "JOB HISTORY",
        "JOB EXPERIENCE",
        "INDUSTRY EXPERIENCE",
        "RELEVANT EXPERIENCE",
        "PRACTICAL EXPERIENCE",
        "PROFESSIONAL BACKGROUND",
        "POSITIONS HELD",
        "ROLES",
        "INTERNSHIP",
        "INTERNSHIPS",
        "INTERNSHIP EXPERIENCE",
        "INTERNSHIPS",
        "SUMMER INTERNSHIP",
        "TRAINING EXPERIENCE",
        "INDUSTRIAL TRAINING",
    ],

    # "internships": [
    #     "INTERNSHIP",
    #     "INTERNSHIPS",
    #     "INTERNSHIP EXPERIENCE",
    #     "INTERNSHIPS",
    #     "SUMMER INTERNSHIP",
    #     "TRAINING EXPERIENCE",
    #     "INDUSTRIAL TRAINING",
    # ],

    "projects": [
        "PROJECTS",
        "PROJECT",
        "PROJECT EXPERIENCE",
        "PROJECT WORK",
        "ACADEMIC PROJECTS",
        "PROFESSIONAL PROJECTS",
        "TECHNICAL PROJECTS",
        "PERSONAL PROJECTS",
        "SIDE PROJECTS",
        "KEY PROJECTS",
        "SELECTED PROJECTS",
        "MAJOR PROJECTS",
        "RELEVANT PROJECTS",
        "CAPSTONE PROJECT",
        "CAPSTONE PROJECTS",
        "OPEN SOURCE PROJECTS",
        "INDEPENDENT PROJECTS",
    ],

    "certifications": [
        "CERTIFICATIONS",
        "CERTIFICATION",
        "CERTIFICATES",
        "CERTIFICATE",
        "LICENSES",
        "LICENSE",
        "LICENSES & CERTIFICATIONS",
        "CERTIFICATIONS & LICENSES",
        "PROFESSIONAL CERTIFICATIONS",
        "TECHNICAL CERTIFICATIONS",
        "CREDENTIALS",
        "ACCREDITATIONS",
        "PROFESSIONAL CREDENTIALS",
    ],

    # "courses": [
    #     "COURSES",
    #     "COURSEWORK",
    #     "RELEVANT COURSEWORK",
    #     "RELATED COURSEWORK",
    #     "ONLINE COURSES",
    #     "TRAINING",
    #     "PROFESSIONAL TRAINING",
    #     "WORKSHOPS",
    #     "WORKSHOP",
    #     "BOOTCAMPS",
    #     "BOOTCAMP",
    #     "CONTINUING EDUCATION",
    #     "MOOCS",
    #     "E-LEARNING",
    # ],

    "research": [
        "RESEARCH",
        "RESEARCH EXPERIENCE",
        "RESEARCH PROJECTS",
        "RESEARCH WORK",
    ],

    "publications": [
        "PUBLICATIONS",
        "PUBLICATION",
        "RESEARCH PUBLICATIONS",
        "RESEARCH PAPERS",
        "PAPERS",
        "JOURNAL ARTICLES",
        "ARTICLES",
        "CONFERENCE PAPERS",
        "PRESENTATIONS",
        "POSTER PRESENTATIONS",
        "WHITE PAPERS",
        "WHITEPAPERS",
        "PATENTS",
        "PATENT",
    ],


    "extracurricular": [
        "EXTRACURRICULAR",
        "EXTRACURRICULAR ACTIVITIES",
        "ACTIVITIES",
        "CAMPUS ACTIVITIES",
        "CO-CURRICULAR ACTIVITIES",
        "CLUBS",
        "ORGANIZATIONS",
        "STUDENT ACTIVITIES",
    ],

    "languages": [
        "LANGUAGES",
        "LANGUAGE",
        "LANGUAGE PROFICIENCY",
        "SPOKEN LANGUAGES",
    ],

    "interests": [
        "INTERESTS",
        "HOBBIES",
        "HOBBIES & INTERESTS",
        "PERSONAL INTERESTS",
        "OTHER INTERESTS",
    ],

    "references": [
        "REFERENCES",
        "PROFESSIONAL REFERENCES",
        "REFEREES",
        "REFERENCE",
        "RECOMMENDATIONS",
    ],

    "additional": [
        "ADDITIONAL INFORMATION",
        "ADDITIONAL DETAILS",
        "OTHER INFORMATION",
        "MISCELLANEOUS",
        "PERSONAL DETAILS",
        "PERSONAL INFORMATION",
    ],
}

LOOKUP = {}

for canonical, aliases in CANONICAL_SECTIONS.items():
    for alias in aliases:
        LOOKUP[alias] = canonical


def normalize_heading(text, threshold=85):

    text = text.upper().strip().rstrip(":")

    match = process.extractOne(
        text,
        LOOKUP.keys(),
        scorer=fuzz.token_set_ratio
    )

    if match is None:
        return None

    alias, score, _ = match

    if score >= threshold:
        return LOOKUP[alias]

    return None


def get_canonical_heading(block):
    if block.label != "section_header":
        return None

    return normalize_heading(block.text)


def split_resume_into_sections(blocks):

    sections = {key: [] for key in CANONICAL_SECTIONS}
    sections["HEADER"] = []

    current_section = "HEADER"

    for block in blocks:
        canonical = get_canonical_heading(block)

        if canonical:
            current_section = canonical
            continue

        sections.setdefault(current_section, []).append(block.text.strip())

    return {
        k: "\n".join(filter(None, v))
        for k, v in sections.items()
    }
