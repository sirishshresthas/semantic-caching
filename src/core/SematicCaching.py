import json
import time
import uuid
from typing import Any, Dict

import numpy as np
import pymongo
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.make_predictions import make_prediction
from src.core.models import CacheBase
from src.core.services import CacheService
from src.core.VectorStorage import VectorDB

cache_service = CacheService()


class SemanticCaching(object):
    def __init__(self, json_file: str = 'cache.json', threshold: int = 0.3, encoder_model: str = 'all-mpnet-base-v2'):

        # Initialize Sentence Transformer model
        self.encoder = SentenceTransformer(encoder_model)

        # initialize Qdrant vectordb
        self.vectorDb = VectorDB(
            vector_size=self.encoder.get_sentence_embedding_dimension())

        # Set Euclidean distance threshold
        self.distance_threshold = threshold
        self.cache: CacheBase = CacheBase()
        self.load_cache()

    def load_cache(self):
        try:
            self.cache: CacheBase = cache_service.find()
            first_result = next(self.cache, None)
            if first_result is None:  # If no document is found, result is empty
                self.cache = CacheBase()
                print("Cache initialized as it was empty: ", self.cache)
        except pymongo.errors.PyMongoError as e:  
            print(f"Error accessing MongoDB: {e}")
            self.cache = CacheBase()
            print("Cache initialized due to MongoDB error: ", self.cache)


    def save_cache(self):
        cache_service.insert_one(self.cache.__dict__)

    def ask(self, question: str) -> str:

        start_time = time.time()
        try:
            embedding = self.encoder.encode(question).tolist()

            points = self.vectorDb.search(query_vector=embedding)
            print("Points: ", points)

            if points:
                point_id = points[0].id
                distance = points[0].score
                print("distance: ", distance)

                if distance >= self.distance_threshold:
                    print(
                        f'Found cache in row: {point_id} with score {distance}')
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time} seconds")

                    # Assuming you have a method to retrieve response text by id
                    return self.cache['response_text'][point_id]

                    # row_id = int(I[0][0])
                    # print(
                    #     f'Found cache in row: {row_id} with score {1 - D[0][0]}')
                    # end_time = time.time()
                    # elapsed_time = end_time - start_time
                    # print(f"Time taken: {elapsed_time} seconds")
                    # return self.cache['response_text'][row_id]

            # Handle the case when there are not enough results or Euclidean distance is not met
            answer, response_text = self.generate_answer(question)

            self.cache.id = str(uuid.uuid4())
            self.cache.question = question
            self.cache.embedding = embedding
            self.cache.answer = answer
            self.cache.response_text = response_text

            # self.index.add(embedding)
            self.vectorDb.upsert(embeddings=embedding, point_id = self.cache.id, metadata=None)
            self.save_cache()
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time} seconds")

            return response_text
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")

    def generate_answer(self, question: str) -> str:
        # Method to generate an answer using a separate function (make_prediction in this case)
        try:
            result = make_prediction(question)
            response_text = result['data']['response_text']

            return result, response_text
        except Exception as e:
            raise RuntimeError(f"Error during 'generate_answer' method: {e}")
