from typing import Any, Dict

from bson import ObjectId


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls) -> Dict[str, Any]:
        # Return a modified schema for PyObjectId
        return {
            "type": "string",
            "title": "PyObjectId",
            "description": "MongoDB ObjectId custom type for Pydantic models",
        }