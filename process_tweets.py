import csv
import json


def process_file(file_path):
    output_data = []
    sentiment_mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sentence = row["text"]
            sentiment = row["airline_sentiment"]
            confidence = float(row["airline_sentiment_confidence"])
            numeric_sentiment = sentiment_mapping[sentiment]
            reasoning = (
                f"The sentiment is {sentiment} based on the content of the tweet."
            )
            output_item = {
                "reasoning": reasoning,
                "sentiment": round(numeric_sentiment, 2),
                "confidence": round(confidence, 2),
            }
            output_data.append((sentence, json.dumps(output_item)))

    return output_data


# Set the file path for the new dataset
file_path = "data/airline_tweaks/Tweets.csv"

# Process the new dataset file
output_data = process_file(file_path)

# Save the output data to a new CSV file
output_file = "data/outputs/Processed_Tweets_output.csv"
with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Sentence", "JSON"])
    writer.writerows(output_data)
