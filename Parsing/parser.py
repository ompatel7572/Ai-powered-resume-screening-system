import math
import unicodedata

from flask import g
from flask_login import current_user
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter1d
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
import os
import re
import json
import phonenumbers


from Parsing.resume_sections_helper import split_resume_into_sections
from Parsing.parser_llm_helper import extract_complex_details, get_chain


def append_to_json(filename, data):
    """Safely appends each dictionary from a list to a JSON file."""
    if not isinstance(data, list):
        data = [data]  # Ensure data is always a list of dictionaries

    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []  # Handle cases where the file isn't a list
            except json.JSONDecodeError:
                existing_data = []  # Handle empty file case

            existing_data.extend(data)  # Append individual dicts

            f.seek(0)
            json.dump(existing_data, f, indent=4)
            f.truncate()  # Remove any trailing data if file was longer
    else:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)  # Write the list directly


# For skillNER to perform properly


def preprocess_resume_text(text):
    """
    Clean resume text before SkillNER extraction.
    """

    # Remove email addresses
    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove phone numbers
    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " ",
        text
    )

    # Replace separators with spaces
    text = re.sub(r"[|•▪►◆■]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# problem in skillner module


def extract_skills(skill_extractor, text, threshold=0.85):
    """
    Extract skills from a resume using SkillNER with dynamic chunking.

    If SkillNER fails on a chunk, that chunk is skipped while the
    remaining chunks are still processed.
    """

    words = text.split()
    n_words = len(words)

    if n_words == 0:
        return []

    # Aim for ~4 chunks
    max_words = max(1000, math.ceil(n_words / 4))

    BLACKLIST = {"com", "www", "e"}

    skills = set()

    for start in range(0, n_words, max_words):

        chunk_words = words[start:start + max_words]

        # Remove a few words from the end to avoid SkillNER edge-case
        if len(chunk_words) > 20:
            chunk_words = chunk_words[:-20]

        chunk = " ".join(chunk_words)
        chunk += " End of resume."

        try:
            annotations = skill_extractor.annotate(chunk)

            full_matches = [
                skill["doc_node_value"]
                for skill in annotations["results"].get("full_matches", [])
            ]

            partial_matches = [
                skill["doc_node_value"]
                for skill in annotations["results"].get("ngram_scored", [])
                if skill.get("score", 0) >= threshold
            ]

            skills.update(full_matches)
            skills.update(partial_matches)

        except (IndexError, ValueError):
            print(f"Skipping chunk {start // max_words + 1}")
            continue

    return sorted(
        skill.title()
        for skill in skills
        if skill.lower() not in BLACKLIST
    )


def extract_phone_number(text):
    """
    Extract the first valid phone number from text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str | None
    """

    try:
        for match in phonenumbers.PhoneNumberMatcher(text, None):
            return phonenumbers.format_number(
                match.number,
                phonenumbers.PhoneNumberFormat.E164
            )

    except phonenumbers.NumberParseException:
        pass

    return None


def extract_email(text):
    """
    Extract the first email address from text.

    Parameters
    ----------
    text : str

    Returns
    -------
    str | None
    """

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        re.IGNORECASE
    )

    match = EMAIL_PATTERN.search(text)

    return match.group() if match else None


def extract_name(nlpT, blocks):

    for block in blocks:

        doc = nlpT(block.text)

        person = next(
            (ent.text for ent in doc.ents if ent.label_ == "PERSON"),
            None
        )

        if person:
            return person

    return None


def parse_resume(resume_str_list, resume_block_list, skill_extractor, nlpT):
    """Parses multiple resumes' other details using Gemini data."""

    all_combined_data = []

    chain = get_chain(g.llm_api_key)

    for data_str, data_block in zip(resume_str_list, resume_block_list):

        preprocess_str_data = preprocess_resume_text(data_str)
        sectioned_dict_data = split_resume_into_sections(data_block)

        other_data = {
            'name': extract_name(nlpT, data_block),
            'email': extract_email(data_str),
            'phone': extract_phone_number(data_str),
            'skills':  extract_skills(skill_extractor, preprocess_str_data, 0.9),
        }

        llm_data = extract_complex_details(chain, sectioned_dict_data)

        # Combine extracted data
        combined_data = {**other_data, **llm_data}

        all_combined_data.append(combined_data)

    # return all_combined_data
    return all_combined_data


def extract_layout_blocks(doc):

    @dataclass
    class LayoutBlock:
        text: str
        label: str
        heading: str | None
        x: float
        y: float
        width: float
        height: float
        page: int

    blocks = []

    for span in doc.spans["layout"]:
        layout = span._.layout

        blocks.append(
            LayoutBlock(
                text=span.text.strip(),
                label=span.label_,
                heading=span._.heading,
                x=layout.x,
                y=layout.y,
                width=layout.width,
                height=layout.height,
                page=layout.page_no,
            )
        )

    # ** Sorting based on page row col  **

    blocks.sort(key=lambda b: (b.page, b.y, b.x))
    return blocks


def detect_multi_columns(blocks):

    # -------------------------------------------------
    # Estimate page width
    # -------------------------------------------------

    page_width = max(
        b.x + b.width
        for b in blocks
    )

    resolution = 1000

    profile = np.zeros(resolution)

    scale = resolution / page_width

    # -------------------------------------------------
    # Paint each block
    # -------------------------------------------------

    for b in blocks:

        start = int(b.x * scale)
        end = int((b.x + b.width) * scale)

        profile[start:end] += 1

    profile = gaussian_filter1d(profile, sigma=8)

    x = np.linspace(0, page_width, resolution)

    # plt.figure(figsize=(12, 4))

    # plt.plot(x, profile)

    # plt.grid(True)

    # plt.show()

    valleys, _ = find_peaks(
        -profile,
        prominence=profile.max()*0.2
    )

    # print(valleys)

    if len(valleys) > 0:
        return True
    else:
        return False

    # if this is true and detects multi then pass resume to llm with coridantes and ask to retuen strcutured cordiantes :


def extract_multi_columns(blocks):

    # Use block centers instead of left edges
    X = np.array([
        [b.x + b.width / 2]
        for b in blocks
    ])

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X)

    # Order left/right
    centers = kmeans.cluster_centers_.flatten()
    order = np.argsort(centers)

    mapping = {
        order[0]: 0,   # left
        order[1]: 1    # right
    }

    labels = np.array([mapping[l] for l in labels])

    left_column = []
    right_column = []

    for block, label in zip(blocks, labels):

        if label == 0:
            left_column.append(block)
        else:
            right_column.append(block)

    left_column.sort(key=lambda b: (b.page, b.y))
    right_column.sort(key=lambda b: (b.page, b.y))

    left_chars = sum(len(b.text) for b in left_column)
    right_chars = sum(len(b.text) for b in right_column)

    if left_chars >= right_chars:
        blocks = left_column + right_column
    else:
        blocks = right_column + left_column

    return blocks


def get_resume_text(paths, layout):
    resume_str_list = []
    resume_block_list = []

    for doc in layout.pipe(paths):
        # edu, exp, ref, other = extract_section(
        #     doc)  # Extract structured sections

        blocks = extract_layout_blocks(doc)

        multi_col = detect_multi_columns(blocks)
        if multi_col is True:
            # print('Inside Multi')
            blocks = extract_multi_columns(blocks)

        result_lst = []
        for i in blocks:
            result_lst.append(i.text)

        result_str = "\n".join(result_lst)

        resume_str_list.append(result_str)
        resume_block_list.append(blocks)
    # print(edu, exp, ref, other)

    return resume_str_list, resume_block_list

