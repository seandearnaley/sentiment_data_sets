import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import litellm
from dotenv import load_dotenv
from litellm import Choices, ModelResponse, completion
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()  # This loads the environment variables from a .env file

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", None),
    api_key=os.getenv("OPENAI_API_KEY", "ollama"),  # api_key is required
)

# from openai.types.chat import ChatCompletionMessageParam
litellm.drop_params = True


class SentimentResponse(BaseModel):
    reasoning: str = Field(
        ...,  # This indicates that the 'reasoning' field must be provided
        description="A brief description explaining the logic used to determine the numeric sentiment value.",
        min_length=1,
    )
    sentiment: float = Field(
        ...,  # This indicates that the 'sentiment' field must be provided
        description="A floating-point representation of the sentiment of the news article, rounded to two decimal places. Scale ranges from -1.0 (negative) to 1.0 (positive), where 0.0 represents neutral sentiment.",
        ge=-1.0,
        le=1.0,
    )
    confidence: float = Field(
        ...,  # This indicates that the 'confidence' field must be provided
        description="A floating-point representation of the confidence in the sentiment analysis, ranging from 0.0 (least confident) to 1.0 (most confident), rounded to two decimal places.",
        ge=0.0,
        le=1.0,
    )


def read_message_from_file(file_path: str) -> str:
    with Path(file_path).open(encoding="utf-8") as file:
        return file.read()


system_message = read_message_from_file("messages/system_message.txt")
message = read_message_from_file("messages/user_message.txt")


def make_api_call(
    initial_messages: List[Any],
    model: str,
    # stop_signals: List[str] = ["TERMINATE"],
    temperature: float = 0.1,
    num_ctx: int = 4096,
) -> str:
    """
    Generic function to make an API call to Ollama.
    """
    response = completion(
        model=model,
        stream=False,
        messages=initial_messages,
        max_tokens=num_ctx,
        temperature=temperature,
        # stop=stop_signals,
        # response_format={"type": "json_object"},
    )

    print("response:", response)

    # Ensure response is an instance of ModelResponse
    if not isinstance(response, ModelResponse):
        raise TypeError("Expected response to be a ModelResponse instance.")

    # Check if response is structured as expected

    # Check if response is structured as expected
    if (
        "choices" in response
        and isinstance(response.choices, list)
        and len(response.choices) > 0
    ):
        choice = response.choices[0]
        # Check if the choice is of type Choices and has a message attribute
        if (
            isinstance(choice, Choices)
            and hasattr(choice, "message")
            and hasattr(choice.message, "content")
        ):
            final_json_string = choice.message.content
            return final_json_string
        else:
            raise TypeError("Response 'choices' missing 'message' or 'content'.")
    else:
        raise TypeError(
            "Unexpected response structure, expected 'choices' in response."
        )


@dataclass
class QueryParams:
    article: str
    model: str
    temperature: float = 0.1
    num_ctx: int = 4096


def query_llm(params: QueryParams) -> str:
    initial_messages: List[Any] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": message},
        {
            "role": "assistant",
            "content": "yes, lets begin, please provide your <article></article> and I will respond with json",
        },
    ]

    initial_messages.append({"role": "user", "content": params.article})

    return make_api_call(
        initial_messages,
        params.model,
        temperature=params.temperature,
        num_ctx=params.num_ctx,
    )


def process_file(file_path, output_file):
    with open(file_path, "r", encoding="utf-8") as file, open(
        output_file, "a", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(file)
        writer = csv.writer(outfile)
        for row in reader:
            title = row["title"]
            description = row["description"]
            article = title + "\n" + description
            attempts = 0
            while attempts < 3:
                try:
                    sentiment_json = query_llm(
                        QueryParams(
                            article=article,
                            model="ollama/llama3:8b-instruct-q8_0",
                        )
                    )
                    # Validate the JSON response using Pydantic V2 method
                    sentiment_data = SentimentResponse.model_validate_json(  # noqa
                        sentiment_json
                    )
                    writer.writerow([article, sentiment_json])
                    break  # Exit the loop if validation is successful
                except ValidationError as e:
                    print(f"Validation error: {e.json()}")
                    attempts += 1
                except json.JSONDecodeError:
                    print("Failed to decode JSON.")
                    attempts += 1
            if attempts == 3:
                print("Failed to process article after 3 attempts:", article)


def process_all_files(directory_path):
    output_file = "data/outputs/Processed_Articles_output.csv"
    # Ensure the output file has the header before appending data
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Sentence", "JSON"])

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            process_file(file_path, output_file)


# Set the directory path for the articles
directory_path = "data/articles"

# Process all CSV files in the directory
process_all_files(directory_path)
