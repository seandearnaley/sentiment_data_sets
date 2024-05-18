# Sentiment Data Sets Project

This project involves processing various datasets for sentiment analysis, combining them, and preparing them for use in machine learning models. Each step involves processing a different type of data and integrating them into a unified format. This README provides a comprehensive guide to understanding and adapting the code for your own needs.

## Project Structure
```
sentiment_data_sets/
│
├── data/
│   ├── inputs/
│   │   ├── airline_tweaks/
│   │   ├── articles/
│   │   ├── FinancialPhraseBank-v1.0/
│   ├── outputs/
│   └── Combined_Articles.csv
│
├── messages/
├── utils/
├── step-01-process_tweets.py
├── step-02-process_financial_phrase_bank.py
├── step-03-process_articles.py
├── step-04-join_outputs.py
├── step-05-build_hf_dataset_sharegpt.py
├── pyproject.toml
└── README.md
```

## Step-by-Step Guide

### Step 1: Processing Tweets
**File: `step-01-process_tweets.py`**

This script processes the airline tweets dataset and saves the output to a new CSV file.

**Features:**
- **Sentiment Mapping:** Converts sentiment labels (positive, neutral, negative) to numeric values (1.0, 0.0, -1.0).
- **Data Processing:** Reads the input CSV, processes each tweet, and constructs a JSON object with sentiment, confidence, and reasoning.
- **Output:** Saves processed data to a new CSV file.

**Dataset Source:**
- [Twitter Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

### Step 2: Processing Financial PhraseBank
**File: `step-02-process_financial_phrase_bank.py`**

This script processes the Financial PhraseBank dataset and saves the output to a CSV file.

**Features:**
- **Confidence Scores:** Assigns different confidence scores based on the agreement level of sentiment annotations.
- **Sentiment Mapping:** Converts sentiment labels to numeric values.
- **Data Processing:** Reads the dataset, processes each phrase, and constructs a JSON object.
- **Output:** Combines processed data from different agreement levels into a single CSV file.

**Dataset Source:**
- [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank)

### Step 3: Processing Articles
**File: `step-03-process_articles.py`**

This script processes a dataset of news articles, generates synthetic outputs using various language models, and saves the output to a CSV file.

**Features:**
- **API Integration:** Uses multiple AI models (e.g., OpenAI GPT-3.5, GPT-4) for sentiment analysis.
- **Retry Mechanism:** Implements a retry mechanism to handle API call failures.
- **Data Validation:** Ensures that the generated JSON responses are valid using Pydantic models.
- **Output:** Saves processed articles with their sentiment analysis to a CSV file.

**Dataset Source:**
- [News Articles](https://newsdata.io/datasets)

### Step 4: Joining Outputs
**File: `step-04-join_outputs.py`**

This script combines the outputs from the tweets, financial phrases, and articles datasets into a single CSV file.

**Features:**
- **Data Sanitization:** Ensures all data is in a consistent format and free from encoding issues.
- **JSON Validation:** Validates JSON strings to ensure they meet the expected format.
- **Output:** Combines valid records from all processed datasets into one CSV file.

### Step 5: Building HuggingFace Dataset
**File: `step-05-build_hf_dataset_sharegpt.py`**

This script converts the combined dataset into a format suitable for uploading to HuggingFace for sharing and training models.

**Features:**
- **Data Transformation:** Reads the combined CSV file, sanitizes the data, and transforms it into a JSON format.
- **Dataset Structure:** Organizes data into a conversation format suitable for training models on HuggingFace.
- **Output:** Saves transformed data to a JSON file ready for upload.

### Utility Scripts
**Files: `utils/sentiment_response.py`, `utils/utils.py`**

These utility scripts provide helper functions and classes used across the main scripts:
- **SentimentResponse:** A Pydantic model for validating JSON responses.
- **File Utilities:** Functions for reading messages, generating record IDs, and loading/saving processed records.

## Dependencies
This project uses Python 3.12 and the following packages:
- `litellm`
- `openai`
- `python-dotenv`
- `pydantic`

These dependencies are managed using Poetry, as specified in the `pyproject.toml` file.

## Setting Up
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/sentiment-data-sets.git
   cd sentiment-data-sets
   ```

2. Install dependencies using Poetry:
   ```sh
   poetry install
   ```

3. Set up your environment variables:
   ```sh
   cp .env.example .env
   # Edit .env to include your API keys
   ```

4. Run the scripts in sequence to process and combine the datasets:
   ```sh
   python step-01-process_tweets.py
   python step-02-process_financial_phrase_bank.py
   python step-03-process_articles.py
   python step-04-join_outputs.py
   python step-05-build_hf_dataset_sharegpt.py
   ```

## Conclusion
By following this guide, you can adapt these scripts to process and analyze your own datasets, enabling you to build comprehensive sentiment analysis datasets for various applications. For full access to the code and datasets, refer to the provided links and file paths.