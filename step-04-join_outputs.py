"""
This script joins the outputs from the articles, tweets, and Financial PhraseBank datasets.
"""

import csv
import json
import os
import unicodedata
from typing import IO, Dict, List, Optional

from pydantic import ValidationError

from utils.sentiment_response import SentimentResponse


def read_csv_file(file_path: str) -> List[Dict[str, str]]:
    """Reads a CSV file and returns a list of dictionaries if the format is correct."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["Sentence", "JSON"]:
                print(f"Skipping file with incorrect format: {file_path}")
                return []
            return [sanitize_row(row) for row in reader]
    except UnicodeDecodeError:
        print(f"Unicode decode error in file: {file_path}")
        return []


def sanitize_row(row: Dict[str, str]) -> Dict[str, str]:
    """Sanitizes the data in a row to ensure it is valid UTF-8, normalized, and free of known ambiguous characters."""
    sanitized_row = {}
    for key, value in row.items():
        # Normalize to NFC form
        normalized_value = unicodedata.normalize("NFC", value)
        # Replace common ambiguous characters
        normalized_value = normalized_value.replace("“", '"').replace("”", '"')
        normalized_value = normalized_value.replace("‘", "'").replace("’", "'")
        normalized_value = normalized_value.replace(
            "—", "-"
        )  # Replace em-dash with hyphen
        # Encode to UTF-8 to ensure compatibility
        sanitized_row[key] = normalized_value.encode("utf-8", "replace").decode("utf-8")
    return sanitized_row


def validate_json(json_str: str) -> Optional[Dict[str, str]]:
    """Validates JSON string using Pydantic model and returns the JSON if valid."""
    try:
        SentimentResponse.model_validate_json(json_str)
        return json.loads(json_str)
    except ValidationError as e:
        print(f"Validation error: {e.json()}")
    except json.JSONDecodeError:
        print("JSON decoding error")
    return None


def write_to_csv(outfile: IO[str], sentence: str, json_str: str):
    """Writes a sentence and JSON string to a CSV file."""
    try:
        writer = csv.writer(outfile)
        writer.writerow([sentence, json_str])
    except UnicodeEncodeError:
        print(f"Unicode encode error with data: {sentence}, {json_str}")


def process_files(input_files: List[str], output_file_path: str) -> int:
    """Processes all input files and writes valid records to an output CSV file."""
    record_count = 0
    with open(output_file_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["Article", "JSON"])  # Write header
        for file_path in input_files:
            for row in read_csv_file(file_path):
                valid_json = validate_json(row["JSON"])
                if valid_json is not None:
                    write_to_csv(outfile, row["Sentence"], row["JSON"])
                    record_count += 1
    return record_count


def get_csv_files(directory_path: str) -> List[str]:
    """Returns a list of CSV file paths in the given directory."""
    return [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(".csv")
    ]


def main(directory_path: str, output_file_path: str):
    """Main function to process all CSV files in the directory."""
    csv_files = get_csv_files(directory_path)
    total_records = process_files(csv_files, output_file_path)
    print(f"Total valid CSV records processed: {total_records}")


if __name__ == "__main__":
    directory_path = "data/outputs"
    output_file_path = "data/Combined_Articles.csv"
    main(directory_path, output_file_path)
