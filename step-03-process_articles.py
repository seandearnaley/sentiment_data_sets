"""
This script processes articles dataset, processes synthetic outputs from multiple providers using litellm and saves the output to a new CSV file.

https://newsdata.io/datasets
"""

import csv
import datetime
import json
import os
import time
from dataclasses import dataclass
from typing import Any, List

import litellm
from dotenv import load_dotenv
from litellm import Choices, ModelResponse, completion
from pydantic import ValidationError

from utils.sentiment_response import SentimentResponse
from utils.utils import (
    generate_record_id,
    load_processed_records,
    read_message_from_file,
    save_processed_records,
)

load_dotenv()

litellm.drop_params = True

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or "YOUR_API_KEY"
os.environ["PERPLEXITY_API_KEY"] = os.getenv("PERPLEXITY_API_KEY") or "YOUR_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY") or "YOUR_API_KEY"
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or "YOUR_API_KEY"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY") or "YOUR_API_KEY"


print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
print("PERPLEXITY_API_KEY:", os.getenv("PERPLEXITY_API_KEY"))
print("ANTHROPIC_API_KEY:", os.getenv("ANTHROPIC_API_KEY"))
print("COHERE_API_KEY:", os.getenv("COHERE_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("MISTRAL_API_KEY:", os.getenv("MISTRAL_API_KEY"))


alternate_models = [
    {
        "model": "groq/llama3-70b-8192",
        "api_key": os.getenv("GROQ_API_KEY"),
    },
    {
        "model": "perplexity/llama-3-70b-instruct",
        "api_key": os.getenv("PERPLEXITY_API_KEY"),
    },
    {
        "model": "perplexity/llama-3-70b-instruct",
        "api_key": os.getenv("PERPLEXITY_API_KEY"),
    },
    {
        "model": "openai/gpt-3.5-turbo-0125",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    {
        "model": "openai/gpt-4-turbo-2024-04-09",
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    # {
    #     "model": "mistral/mistral-large-latest",
    #     "api_key": os.getenv("MISTRAL_API_KEY"),
    # },
    # {
    #     "model": "anthropic/claude-3-sonnet-20240229",
    #     "api_key": os.getenv("ANTHROPIC_API_KEY"),
    # },
]


system_message = read_message_from_file("messages/system_message.txt")

# Get the current date and time in the desired format
current_datetime = datetime.datetime.utcnow().strftime("%A %B %d %Y %H:%M:%S UTC")

# Replace the {date_msg} placeholder with the current date and time
system_message = system_message.replace("{date_msg}", current_datetime)


message = read_message_from_file("messages/user_message.txt")


def call_api(model, messages, max_tokens, temperature, api_key):
    """
    Makes a single API call to the specified model.
    """
    return completion(
        model=model,
        stream=False,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_key=api_key,
    )


def retry_api_call(model, messages, max_tokens, temperature, api_key, retry_delays):
    """
    Retries API call based on the specified retry delays.
    """
    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            print(f"Attempt {attempt}: Calling API with model {model}")
            response = call_api(model, messages, max_tokens, temperature, api_key)
            print("API call successful.")
            return response
        except litellm.exceptions.APIConnectionError:
            if attempt == len(retry_delays):
                print("Final attempt failed.")
                raise Exception("API call failed after multiple retries.")
            print(f"API connection error occurred. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            print(f"An error occurred: {e}")
            raise


def extract_response(response):
    """
    Extracts and validates the response from the API call.
    """
    if (
        isinstance(response, ModelResponse)
        and "choices" in response
        and isinstance(response.choices, list)
        and len(response.choices) > 0
    ):
        choice = response.choices[0]
        if (
            isinstance(choice, Choices)
            and hasattr(choice, "message")
            and hasattr(choice.message, "content")
        ):
            return choice.message.content
        else:
            raise TypeError("Response 'choices' missing 'message' or 'content'.")
    else:
        raise TypeError(
            "Unexpected response structure, expected 'choices' in response."
        )


def make_api_call(initial_messages, model_index, temperature, num_ctx):
    """
    Generic function to make an API call to a model with a retry mechanism.
    Alternates between models based on the model_index.
    """
    retry_delays = [30, 60, 90]  # seconds
    model_details = alternate_models[model_index]
    api_key = model_details["api_key"]
    model = model_details["model"]
    print()
    print(f"Making API call to model: {model} with index {model_index}")
    print()
    response = retry_api_call(
        model, initial_messages, num_ctx, temperature, api_key, retry_delays
    )
    return extract_response(response)


@dataclass
class QueryParams:
    article: str
    temperature: float = 0.1
    num_ctx: int = 2048
    model_index: int = 0  # Default to the first model


def query_llm(params: QueryParams) -> str:
    initial_messages: List[Any] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
        {
            "role": "assistant",
            "content": "yes, lets begin, please provide your text and I will respond with json",
        },
    ]

    initial_messages.append({"role": "user", "content": params.article})

    response = make_api_call(
        initial_messages,
        model_index=params.model_index,
        temperature=params.temperature,
        num_ctx=params.num_ctx,
    )
    return response


def write_header_if_needed(output_file, writer):
    file_exists = os.path.exists(output_file)
    should_write_header = not file_exists or os.stat(output_file).st_size == 0
    if should_write_header:
        writer.writerow(["Article", "Sentiment"])


def process_article(
    title,
    description,
    processed_records,
    writer,
    processed_records_file,
    save_interval,
    count,
    model_index,
):
    article = title + "\n" + description
    record_id = generate_record_id(title, description)
    if record_id in processed_records:
        print(f"Skipping already processed article: {record_id}")
        return count  # Skip processing if record has been processed

    attempts = 0
    while attempts < 3:
        try:
            print(f"Processing article: {record_id}")
            sentiment_json = query_llm(
                QueryParams(article=article, model_index=model_index)
            )
            SentimentResponse.model_validate_json(sentiment_json)
            writer.writerow([article, sentiment_json])
            processed_records.add(record_id)
            count += 1
            if count % save_interval == 0:
                print()  # New line before
                print(f"Saving Processed {count} articles.")
                print()  # New line after
                save_processed_records(processed_records, processed_records_file)
            break  # Exit the loop if validation is successful
        except ValidationError as e:
            print(f"Validation error: {e.json()}")
            attempts += 1
        except json.JSONDecodeError:
            print("Failed to decode JSON.")
            attempts += 1
    if attempts == 3:
        print("Failed to process article after 3 attempts:", article)
    return count


def process_file(
    file_path, output_file, processed_records, processed_records_file, save_interval=10
):
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        with open(output_file, "a", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            write_header_if_needed(output_file, writer)

            count = 0
            model_index = 0
            for row in reader:
                title = row["title"]
                description = row["description"]
                count = process_article(
                    title,
                    description,
                    processed_records,
                    writer,
                    processed_records_file,
                    save_interval,
                    count,
                    model_index,
                )

                # Update model_index for the next article
                model_index = (model_index + 1) % len(alternate_models)

            # Save any remaining records that were not saved due to the interval
            save_processed_records(processed_records, processed_records_file)


def process_all_files(directory_path, output_file, processed_records_file):
    processed_records = load_processed_records(processed_records_file)
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            process_file(
                file_path, output_file, processed_records, processed_records_file
            )
    save_processed_records(processed_records, processed_records_file)


# Example usage
directory_path = "data/inputs/articles"

# hash file that keeps track of processed records
processed_records_file = "data/Processed_Records.txt"

output_file = "data/outputs/Processed_Articles_output_70b.csv"
process_all_files(directory_path, output_file, processed_records_file)
