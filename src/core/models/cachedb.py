from typing import Dict, List

from pydantic import BaseModel, Field


class CacheBase(BaseModel):
    """
    Represents the base structure for caching question-answer pairs along with their embeddings and identifiers.

    Attributes:
        qdrant_id (str): ID associated with Qdrant Vector DB.
        question (str): Question asked by the user.
        embedding (List): Embedding of the question.
        answer (str): Complete answer from the RAG (Retrieval-Augmented Generation model).
        response_text (str): The response text derived from the answer.
    """

    qdrant_id: str = Field(None, title="Qdrant ID",
                           description="ID associated with Qdrant Vector DB.")
    question: str = Field(None, title="Question",
                          description="Question asked by user")
    embedding: List = Field(None, title="embedding",
                           description="Embedding of the question.")
    answer: Dict = Field(None, title="Answer",
                        description="Complete answer from the RAG.")
    response_text: str = Field(
        None, title="Response", description="The response text from the answer.")

    def to_json(self):
        """
        Serializes the model to a JSON string, excluding attributes with None values.

        Returns:
            str: JSON string representation of the model.
        """
        return self.model_dump_json(exclude_none=True)

    def serialize(self):
        """
        Provides a dictionary representation of the model, suitable for serialization.

        Returns:
            dict: A dictionary with the model's data.
        """
        return {
            "qdrant_id": self.qdrant_id,
            "question": self.question,
            "embedding": self.embedding,
            "answer": self.answer,
            "response_text": self.response_text
        }

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
