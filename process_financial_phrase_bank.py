import csv
import json
import os


def process_file(file_path, confidence_score):
    output_data = []

    with open(file_path, "r", encoding="ISO-8859-1") as file:
        for line in file:
            sentence, sentiment = line.strip().split("@")
            sentiment_mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            numeric_sentiment = sentiment_mapping[sentiment]
            reasoning = f"The sentiment is {sentiment} based on the financial context of the sentence."
            output_item = {
                "reasoning": reasoning,
                "sentiment": round(numeric_sentiment, 2),
                "confidence": confidence_score,
            }
            output_data.append((sentence, json.dumps(output_item)))

    return output_data


# Define confidence scores for each agreement level
confidence_scores = {
    "Sentences_AllAgree.txt": 1.0,
    "Sentences_75Agree.txt": 0.8,
    "Sentences_66Agree.txt": 0.6,
    "Sentences_50Agree.txt": 0.4,
}

# Set the base folder path
base_folder = "data/FinancialPhraseBank-v1.0"

# Initialize a list to collect all data
all_output_data = []

# Process the dataset files
for file_name, confidence_score in confidence_scores.items():
    file_path = os.path.join(base_folder, file_name)
    output_data = process_file(file_path, confidence_score)
    all_output_data.extend(output_data)

# Save all the output data to a single CSV file
output_file = os.path.join("data/outputs/", "All_Sentences_output.csv")
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "JSON"])
    writer.writerows(all_output_data)
