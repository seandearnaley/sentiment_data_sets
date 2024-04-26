import hashlib
from pathlib import Path


def read_message_from_file(file_path: str) -> str:
    with Path(file_path).open(encoding="utf-8") as file:
        return file.read()


def generate_record_id(title, description):
    record_string = f"{title}{description}"
    return hashlib.md5(record_string.encode()).hexdigest()


def load_processed_records(file_path):
    try:
        with open(file_path, "r") as file:
            records = file.read().splitlines()
            # Ensure all lines are treated as individual record IDs, not as a JSON array
            return set(records)
    except FileNotFoundError:
        return set()


def save_processed_records(processed_records, file_path):
    with open(file_path, "w") as file:
        for record_id in processed_records:
            file.write(f"{record_id}\n")
