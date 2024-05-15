import csv
import json
import random
import unicodedata
from typing import Dict


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


def read_and_transform_csv(input_file_path: str, output_json_path: str):
    data = []
    with open(input_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sanitized_row = sanitize_row(row)
            try:
                # Parse the JSON string from the sanitized CSV to ensure it's valid and properly escaped
                output_json = json.loads(sanitized_row["JSON"])
                # Re-serialize to JSON string to ensure proper formatting and escaping
                output_json_string = json.dumps(output_json)
            except json.JSONDecodeError:
                # Handle cases where the JSON data is invalid
                output_json_string = "{}"  # Use an empty JSON object as a fallback

            entry = {
                "instruction": "Analyze the sentiment of this financial news article.",
                "input": sanitized_row["Article"],
                "output": output_json_string,
            }
            data.append(entry)

    # Randomize the order of data entries
    random.shuffle(data)

    with open(output_json_path, mode="w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)


if __name__ == "__main__":
    input_csv_path = "data/Combined_Articles.csv"
    output_json_path = "data/financial_sentiment_hf_dataset.json"
    read_and_transform_csv(input_csv_path, output_json_path)
