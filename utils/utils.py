import hashlib
from pathlib import Path


def read_message_from_file(file_path: str) -> str:
    with Path(file_path).open(encoding="utf-8") as file:
        return file.read()


def generate_record_id(title, description):
    record_string = f"{title}{description}"
    return hashlib.md5(record_string.encode()).hexdigest()


def load_processed_records(file_path):
    file = Path(file_path)
    if not file.exists():
        file.touch()  # Create the file if it doesn't exist
        return set()
    with file.open("r", encoding="utf-8") as f:
        records = f.read().splitlines()
        return set(records)


def save_processed_records(processed_records, file_path):
    with open(file_path, "w") as file:
        for record_id in processed_records:
            file.write(f"{record_id}\n")
