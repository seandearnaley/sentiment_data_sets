import csv
import random
from typing import List, Tuple


def read_csv(file_path: str) -> List[List[str]]:
    """Reads a CSV file and returns a list of rows."""
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        return list(reader)


def split_data(
    data: List[List[str]], split_ratio: float
) -> Tuple[List[List[str]], List[List[str]]]:
    """Splits the data into training and testing sets based on the split ratio."""
    random.shuffle(data)  # Randomize the data
    split_index = int(len(data) * split_ratio)
    return data[:split_index], data[split_index:]


def write_csv(data: List[List[str]], file_path: str):
    """Writes a list of rows to a CSV file."""
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)


def main(input_file_path: str, split_ratio: float):
    """Main function to read, split, and write the CSV data."""
    data = read_csv(input_file_path)
    headers = data[0]  # Assuming the first row is the header
    data_without_headers = data[1:]  # Remove the header for splitting

    training_data, testing_data = split_data(data_without_headers, split_ratio)
    training_data.insert(0, headers)  # Reinsert the header
    testing_data.insert(0, headers)

    write_csv(training_data, "data/training_data.csv")
    write_csv(testing_data, "data/testing_data.csv")
    print(
        f"Data split into {len(training_data)-1} training records and {len(testing_data)-1} testing records."
    )


if __name__ == "__main__":
    input_file_path = "data/Combined_Articles.csv"
    split_ratio = 0.8
    main(input_file_path, split_ratio)
