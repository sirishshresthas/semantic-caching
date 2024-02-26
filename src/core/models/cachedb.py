from typing import List, Optional

from bson import ObjectId
from pydantic import BaseModel, Field

from .PyObjectId import PyObjectId


class CacheBase(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    question: str = Field(None, title="Question",
                          description="Question asked by user")
    embedding: str = Field(None, title="embedding",
                           description="Embedding of the question.")
    answer: str = Field(None, title="Answer",
                        description="Complete answer from the RAG.")
    response_text: str = Field(
        None, title="Response", description="The response text from the answer.")

    def to_json(self):
        return self.model_dump_json(exclude_none=True)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class CacheList(BaseModel):
    cache_list: List[CacheBase]

    def to_json(self):
        return self.model_dump_json(exclude_none=True)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
