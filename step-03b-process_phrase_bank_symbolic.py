"""
This script processes the Financial PhraseBank dataset and saves the output to a CSV file.

https://huggingface.co/datasets/financial_phrasebank
"""

import csv
import datetime
import os
import re
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
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY") or "YOUR_API_KEY"

print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
print("PERPLEXITY_API_KEY:", os.getenv("PERPLEXITY_API_KEY"))
print("ANTHROPIC_API_KEY:", os.getenv("ANTHROPIC_API_KEY"))
print("COHERE_API_KEY:", os.getenv("COHERE_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("MISTRAL_API_KEY:", os.getenv("MISTRAL_API_KEY"))


system_message = read_message_from_file("messages/system_message_symbolic.txt")

# Get the current date and time in the desired format
current_datetime = datetime.datetime.utcnow().strftime("%A %B %d %Y %H:%M:%S UTC")

# Replace the {date_msg} placeholder with the current date and time
system_message = system_message.replace("{date_msg}", current_datetime)


message = read_message_from_file("messages/user_message_symbolic.txt")

print("system_message:", system_message)
print("message:", message)


def call_api(messages, max_tokens, temperature):
    """
    Makes a single API call to the specified model.
    """
    return completion(
        model="ollama/lstep/neuraldaredevil-8b-abliterated:q8_0",
        stream=False,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        api_base="http://localhost:11434",
        response_format={"type": "json_object"},
    )


def retry_api_call(messages, max_tokens, temperature, retry_delays):
    """
    Retries API call based on the specified retry delays.
    """
    for attempt, delay in enumerate(retry_delays, start=1):
        try:
            print(f"Attempt {attempt}: Calling API")
            response = call_api(messages, max_tokens, temperature)
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
    import re

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
            content = choice.message.content

            # Regular expression to find JSON object
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json_match.group(0).strip()
            else:
                return content.strip()
        else:
            raise TypeError("Response 'choices' missing 'message' or 'content'.")
    else:
        raise TypeError(
            "Unexpected response structure, expected 'choices' in response."
        )


def make_api_call(initial_messages, temperature, max_tokens):
    """
    Generic function to make an API call to a model with a retry mechanism.
    """
    retry_delays = [30, 60, 90]  # seconds
    response = retry_api_call(initial_messages, max_tokens, temperature, retry_delays)
    return extract_response(response)


@dataclass
class QueryParams:
    article: str
    temperature: float = 0.2
    max_tokens: int = 800
    model_index: int = 0  # Default to the first model


def query_llm(params: QueryParams) -> str:
    initial_messages: List[Any] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
        {
            "role": "assistant",
            "content": "yes, lets begin, please provide your text and I will respond with a valid JSON object string adhering to the schema.",
        },
    ]

    initial_messages.append({"role": "user", "content": params.article})

    response = make_api_call(
        initial_messages,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
    )
    return response


def write_header_if_needed(output_file, writer):
    file_exists = os.path.exists(output_file)
    should_write_header = not file_exists or os.stat(output_file).st_size == 0
    if should_write_header:
        writer.writerow(["Article", "Sentiment"])


def clean_input(phrase: str) -> str:
    """
    Removes control characters from the input phrase.
    """
    return re.sub(r"[\x00-\x1F]+", "", phrase)


def process_phrase(
    phrase,
    processed_records,
    writer,
    processed_records_file,
    save_interval,
    count,
):
    phrase = clean_input(phrase)
    record_id = generate_record_id(phrase, None)
    if record_id in processed_records:
        print(f"Skipping already processed phrase: {record_id}")
        return count  # Skip processing if record has been processed

    attempts = 0
    while attempts < 3:
        try:
            print(f"Processing phrase: {record_id}")
            sentiment_json = query_llm(QueryParams(article=phrase))
            SentimentResponse.model_validate_json(sentiment_json)
            writer.writerow([phrase, sentiment_json])
            processed_records.add(record_id)

            if (
                count is not None
                and save_interval is not None
                and count % save_interval == 0
            ):
                save_processed_records(processed_records, processed_records_file)

            # Ensure count is not None
            if count is None:
                count = 0

            return count + 1
        except ValidationError as e:
            print(f"Validation error: {e}")
            attempts += 1
            time.sleep(1)  # Wait before retrying
        except Exception as e:
            print(f"An error occurred: {e}")
            attempts += 1
            time.sleep(1)  # Wait before retrying


def main():
    input_file = "data/inputs/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
    output_file = "data/outputs/processed_phrases_jun1_24.csv"
    processed_records_file = "data/processed_phrase_records.txt"
    save_interval = 5

    processed_records = load_processed_records(processed_records_file)

    with open(input_file, "r", encoding="ISO-8859-1") as infile:
        with open(output_file, "a", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            write_header_if_needed(output_file, writer)

            count = 0
            for line in infile:
                phrase = line.strip()
                if phrase:
                    count = process_phrase(
                        phrase,
                        processed_records,
                        writer,
                        processed_records_file,
                        save_interval,
                        count,
                    )

    save_processed_records(processed_records, processed_records_file)


if __name__ == "__main__":
    main()
