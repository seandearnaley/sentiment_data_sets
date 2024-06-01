from pydantic import BaseModel, Field


class SentimentResponse(BaseModel):
    reasoning: str = Field(
        ...,  # This indicates that the 'reasoning' field must be provided
        description="The symbolic logic used to determine the sentiment.",
        min_length=1,
    )
    conclusion: str = Field(
        ...,  # This indicates that the 'reasoning' field must be provided
        description="The conclusion of the symbolic reasoning.",
        min_length=1,
    )
    sentiment: float = Field(
        ...,  # This indicates that the 'sentiment' field must be provided
        description="A floating-point representation of the sentiment of the text, rounded to two decimal places. Scale ranges from -1.0 (negative) to 1.0 (positive), where 0.0 represents neutral sentiment.",
        ge=-1.0,
        le=1.0,
    )
    confidence: float = Field(
        ...,  # This indicates that the 'confidence' field must be provided
        description="A floating-point representation of the confidence in the sentiment analysis, ranging from 0.0 (least confident) to 1.0 (most confident), rounded to two decimal places.",
        ge=0.0,
        le=1.0,
    )
