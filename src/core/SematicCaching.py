import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

import pymongo
from sentence_transformers import SentenceTransformer

from src.core import VectorStorage, models, requests, services
from src.core.utilities import setup_logging

setup_logging(log_file="cache.log")
logger = logging.getLogger(__name__)


class SemanticCaching(object):
    """
    A class for caching and retrieving semantic search results to improve efficiency and response time.
    """

    def __init__(self, threshold: int = 0.8, encoder_model: str = 'all-mpnet-base-v2', distance_metric: str = 'cosine'):
        """
        Initializes the SemanticCaching instance.

        Args:
            threshold (float): The similarity threshold for considering a cache hit.
            encoder_model (str): The model used by SentenceTransformer for encoding sentences.
            distance_metric (str): Distance metric to use in Qdrant collection while creating it. It defaults to cosine. Available are dot, cosine, euclidean, manhattan.
        """
        self.cache_service = services.CacheService()
        self.encoder = SentenceTransformer(encoder_model)
        self.vectorDb = VectorStorage.VectorDB(
            vector_size=self.encoder.get_sentence_embedding_dimension(), distance_metric=distance_metric)
        self._distance_threshold = threshold
        self.cache = models.CacheBase()

        logger.info("Initiating SemanticCaching.")
        logger.info(
            f"encoder_model={encoder_model}; vector_size={self.encoder.get_sentence_embedding_dimension()}; distance_threshold={self._distance_threshold}")

    @property
    def distance_threshold(self):
        """Returns the current distance threshold."""
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, threshold):
        """Sets a new distance threshold."""
        self._distance_threshold = threshold

    def is_cuda_available(self): 
        import torch
        is_cuda: bool = torch.cuda.is_available()
        print(f"Cuda is {'available' if is_cuda else 'not available'}")

    def save_cache(self):
        """Inserts the current cache state into the database."""
        try:
            logging.info("Inserting data to mongo db. Collection: cache")
            self.cache_service.insert_one(self.cache.serialize())
        except pymongo.errors.PyMongoError as e:
            logging.error(f"Error inserting data to MongoDB: {e}")

    def ask(self, question: str, use_llm: bool = False, model_id: str = "mistralai/Mistral-7B-v0.1") -> str:
        """
        Processes a question, checks for cache hits, and returns the corresponding answer.

        Args:
            question (str): The question to process.
            use_llm (bool): True if the response should come from LLM directly. Default is False.
            model_id (str): The Huggingface model id to download the model. This should be paired with use_llm

        Returns:
            str: The answer to the question.
        """
        if not use_llm:
            model_id = ""
        if use_llm and model_id != "":
            msg: str = f"Generating answer using {model_id}"
            logger.info(msg)
            print(msg)
            

        elif use_llm and model_id == "":
            msg: str = "Error: It seems you're attempting to generate an answer using the Large Language Model (LLM), but the specific Model ID required to initiate the process is missing. Providing the Model ID is crucial for accurate and targeted responses."

            logger.error(
                "Error: Missing Model ID for Large Language Model (LLM)")

            raise ValueError(msg)
        
        if not self.is_cuda_available(): 
                logger.error("Cuda is not available for this device. Defaulting to ARES API.")
                use_llm = False

        try:
            logger.info("Asking question")
            start_time = time.time()
            metadata: Dict = {}

            # encoding the question using sentence transformer
            embedding = self.encoder.encode(question).tolist()

            # identify the points from the vectors
            points = self.vectorDb.search(query_vector=embedding)

            logger.info("Evaluating similarity in the database.")
            return self.evaluate_similarity(points=points, start_time=start_time, question=question, embedding=embedding, metadata=metadata, model_id=model_id)
        except Exception as e:
            logger.error(f"Error during 'ask' method: {e}")
            raise

    def evaluate_similarity(self, points: List[Dict], start_time: float, question: str, embedding: List[float], metadata: Dict, model_id: str = "") -> str:
        """
        Evaluates similarity of the given points to the query and handles cache hits or misses accordingly.

        Args:
            points (List[Dict]): A list of points from the vector database search.
            start_time (float): The start time of the query processing for timing analytics.
            question (str): The original question asked.
            embedding (List[float]): The embedding vector of the question.
            metadata (Dict): The metadata to add to the vector db.

        Returns:
            str: The answer to the question, either from cache or freshly generated.
        """
        if points:
            point = points[0]
            score = point.score
            
            # identify the distance metric to compare score with threshold
            is_hit = False

            logger.info(
                f"Current distance metric set on vector db is {self.vectorDb.distance_metric.lower()}")
            if self.vectorDb.distance_metric.lower() in ['cosine', 'dot']:
                is_hit = score > self.distance_threshold

            else:
                is_hit = score <= self.distance_threshold

            if is_hit:
                result = self.handle_cache_hit(
                    point_id=point.id, distance=score)

                if not result:
                    print("Data doesn't seem to exist in the cache. Populating..")
                    logger.info(
                        "Data doesn't seem to exist in the cache. Populating..")
                    result = self.handle_cache_miss(
                        question=question, embedding=embedding, point_id=point.id, metadata=metadata, model_id=model_id)
                    logger.info("Result: ", result)
                else:
                    result = result['response_text']

                self.display_elapsed_time(start_time)
                return result

        result = self.handle_cache_miss(
            question, embedding, metadata=metadata, model_id=model_id)
        logger.info(f"Result: result")
        self.display_elapsed_time(start_time)
        return result

    def handle_cache_hit(self, point_id: str, distance: float) -> str:
        """
        Handles a cache hit by retrieving and returning the cached response.

        Args:
            point_id (str): The identifier of the cache hit point.
            distance (float): The distance or similarity score of the hit.

        Returns:
            Dict: The cached document, if found.
        """
        logger.info(
            f'Found cache hit: {point_id} with distance {distance}')
        print(f'Found cache hit: {point_id} with distance {distance}')

        response_cursor = self.cache_service.find(
            filter={"qdrant_id": point_id})

        response_document = next(response_cursor, None)
        return response_document

    def handle_cache_miss(self, question: str, embedding: List, point_id: str = None, metadata: Dict = None, model_id: str = '') -> str:
        """ 
        Handles a cache miss by generating an answer, adding it to the cache, and returning the response.

        Args:
            question (str): The question asked.
            embedding (List[float]): The embedding of the question.
            point_id (str, optional): The point ID if available. Defaults to None.
            metadata (Optional[Dict]): The metadata to add to the vector db.

        Returns:
            str: The generated answer to the question.
        """
        print("Cache not found. Fetching data..")
        logger.info("Cache not found. Fetching data .. ")
        answer, response_text = self.generate_answer(
            question, model_id)

        self.add_to_cache(question, embedding, answer,
                          response_text, point_id=point_id, metadata=metadata)
        return response_text

    def add_to_cache(self, question: str, embedding: List[float], answer: str, response_text: str, point_id: str = None, metadata: Dict = None) -> None:
        """
        Adds a new entry to the cache.

        Args:
            question (str): The question asked.
            embedding (List[float]): The embedding of the question.
            answer (str): The answer to the question.
            response_text (str): The response text to be cached.
            point_id (str, optional): An optional point ID. Generates a new UUID if None.
            metadata (Optional[Dict]): The metadata to add to the vector db.

        """
        logger.info("Storing data to cache")
        point_id = point_id if point_id else str(uuid.uuid4())
        self.cache.qdrant_id = point_id
        self.cache.question = question
        self.cache.embedding = embedding
        self.cache.answer = answer
        self.cache.response_text = response_text

        logger.info("Storing data to vector database.")
        self.vectorDb.upsert(embeddings=embedding,
                             point_id=point_id, metadata=metadata)
        self.save_cache()

    def generate_answer(self, question: str, model_id: str = "") -> Tuple[str, str]:
        """
        Generates an answer for a given question.

        Args:
            question (str): The question to answer.

        Returns:
            Tuple[str, str]: The answer and the response text.
        """

        if model_id != "":
            result = requests.get_answer(question, model_id=model_id)
            print(result)

        else:
            try:
                logger.info("Getting answer from Ares.")
                result = requests.get_answer(question)
                return result['data'], result['data']['response_text']
            except Exception as e:
                raise RuntimeError(f"Error during 'generate_answer': {e}")

    def display_elapsed_time(self, start_time):
        """
        Logs the elapsed time since the provided start time.

        Args:
            start_time (float): The start time to measure from.
        """
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time to respond: {elapsed_time} seconds")
        logging.info(f"Time to respond: {elapsed_time} seconds")
