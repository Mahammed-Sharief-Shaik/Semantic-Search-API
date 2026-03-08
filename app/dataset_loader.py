import os
import tarfile
from pathlib import Path


DATA_DIR = Path("data")


# def extract_dataset(filename):
#     """
#     Extracts the tar.gz dataset if not already extracted.
#     """

#     dataset_path = DATA_DIR / filename
#     extract_path = DATA_DIR / filename.replace(".tar.gz", "")

#     if extract_path.exists():
#         print("Dataset already extracted.")
#         return extract_path

#     print("Extracting dataset...")

#     with tarfile.open(dataset_path, "r:gz") as tar:
#         tar.extractall(DATA_DIR)

#     return extract_path

def extract_dataset(tar_path, extract_path="data/mini_newsgroups"):

    if os.path.exists(extract_path):
        print("Dataset already extracted.")
        return extract_path

    print("Extracting dataset...")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("data")

    return extract_path


def load_documents(dataset_folder):
    """
    Loads all documents from dataset folders.
    Each subfolder represents a category.
    """

    documents = []
    labels = []

    for category in os.listdir(dataset_folder):

        category_path = os.path.join(dataset_folder, category)

        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            try:
                with open(file_path, "r", encoding="latin1") as f:
                    text = f.read()
                    
                    # Intially added all as it is so lot of noise
                    # documents.append(text)
                    # labels.append(category)

                    # Now only adding the main content
                    cleaned_text = clean_document(text)
                    if len(cleaned_text) > 50:
                        documents.append(cleaned_text)
                        labels.append(category)

            except Exception:
                continue

    return documents, labels



import re


def clean_document(text):
    """
    Cleans a raw newsgroup article by removing metadata,
    quoted replies, and signatures while preserving
    the semantic body content.
    """

    # -------------------------------------------------
    # 1. Extract Subject line (if present)
    # -------------------------------------------------

    subject_match = re.search(r"Subject:\s*(.*)", text)

    subject = ""

    if subject_match:
        subject = subject_match.group(1).strip()

    # -------------------------------------------------
    # 2. Remove header section
    # -------------------------------------------------
    # Headers end at the first empty line

    parts = text.split("\n\n", 1)

    if len(parts) > 1:
        body = parts[1]
    else:
        # maybe header is absent
        body = text

    # -------------------------------------------------
    # 3. Remove quoted replies
    # -------------------------------------------------

    body_lines = body.split("\n")

    cleaned_lines = []

    for line in body_lines:

        # remove lines starting with >
        if line.strip().startswith(">") or line.strip().startswith("|>"):
            continue

        cleaned_lines.append(line)

    body = "\n".join(cleaned_lines)

    # -------------------------------------------------
    # 4. Remove signatures
    # -------------------------------------------------
    # common signature indicators:
    # --
    # __
    # ASCII separators

    body = re.split(r"\n--\s*\n", body)[0]

    # -------------------------------------------------
    # 5. Remove excessive whitespace
    # -------------------------------------------------

    body = re.sub(r"\s+", " ", body)

    body = body.strip()

    # -------------------------------------------------
    # 6. Combine Subject + Body
    # -------------------------------------------------

    cleaned_text = subject + " " + body

    cleaned_text = cleaned_text.strip()

    return cleaned_text