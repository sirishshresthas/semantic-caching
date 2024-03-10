import logging
import time
import uuid
from typing import Dict, List, Tuple

import pymongo
import torch
from sentence_transformers import SentenceTransformer

from semantic_caching.core import VectorStorage, models, requests, services
from semantic_caching.core.utilities import ConsoleLogger, setup_logging

setup_logging(log_file="cache.log")
logger = logging.getLogger(__name__)

console_logger = ConsoleLogger()

class SemanticCaching(object):
    """
    A class for caching and retrieving semantic search results to improve efficiency and response time.
    """

    def __init__(self, threshold: int = 0.8):
        """
        Initializes the SemanticCaching instance.

        Args:
            threshold (float): The similarity threshold for considering a cache hit.
            encoder_model (str): The model used by SentenceTransformer for encoding sentences.
            distance_metric (str): Distance metric to use in Qdrant collection while creating it. It defaults to cosine. Available are dot, cosine, euclidean, manhattan.
        """
        console_logger.start_timer("Initializing mongodb..... ")

        self.cache_service = services.CacheService()
        self._distance_threshold = threshold
        self.cache = models.CacheBase()

        console_logger.end_timer()
        logger.info("Initiating SemanticCaching.")

    @property
    def distance_threshold(self):
        """Returns the current distance threshold."""
        return self._distance_threshold

    @distance_threshold.setter
    def distance_threshold(self, threshold):
        """Sets a new distance threshold."""
        self._distance_threshold = threshold

    @staticmethod
    def is_cuda_available() -> bool:
        is_cuda: bool = torch.cuda.is_available()
        logger.info(f"CUDA is {'available' if is_cuda else 'not available'}")
        return is_cuda

    def init_vector_db(self, distance_metric: str = 'cosine', encoder_model: str = 'all-mpnet-base-v2'):
        logger.info("Initiating vector db")
        
        ## sentence encoder
        console_logger.start_timer("Initializing encoder...")

        self.encoder = SentenceTransformer(encoder_model)

        console_logger.end_timer()
        
        ## vector database
        console_logger.start_timer("Initializing Qdrant vector db..... ")
        
        self.vectorDb = VectorStorage.VectorDB(
            vector_size=self.encoder.get_sentence_embedding_dimension(), distance_metric=distance_metric)

        console_logger.end_timer()

        logger.info(
            f"encoder_model={encoder_model}; vector_size={self.encoder.get_sentence_embedding_dimension()}; distance_threshold={self._distance_threshold}")


    def save_cache(self):
        """Inserts the current cache state into the database."""
        try:
            console_logger.start_timer("saving cache..... ")

            logging.info("Inserting data to mongo db. Collection: cache")
            self.cache_service.insert_one(self.cache.serialize())

            console_logger.end_timer()

        except pymongo.errors.PyMongoError as e:
            logging.error(f"Error inserting data to MongoDB: {e}")

    def ask(self, question: str, use_llm: bool = False, model_id: str = "mistralai/Mistral-7B-Instruct-v0.2") -> str:
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

        elif use_llm and model_id == "":
            msg: str = "Error: It seems you're attempting to generate an answer using the Large Language Model (LLM), but the specific Model ID required to initiate the process is missing. Providing the Model ID is crucial for accurate and targeted responses."

            logger.error(
                "Error: Missing Model ID for Large Language Model (LLM)")

            raise ValueError(msg)

        if not self.is_cuda_available():
            logger.error(
                "CUDA is not available for this device. Defaulting to ARES API.")
            use_llm = False

        try:

            metadata: Dict = {}

            # encoding the question using sentence transformer
            embedding = self.encoder.encode(question).tolist()

            console_logger.start_timer("Searching for the answer in Qdrant.")
            points = self.vectorDb.search(query_vector=embedding)
            console_logger.end_timer()

            logger.info("Evaluating similarity in the database.")
            return self.evaluate_similarity(points=points, question=question, embedding=embedding, metadata=metadata, model_id=model_id)
        except Exception as e:
            logger.error(f"Error during 'ask' method: {e}")
            raise

    def evaluate_similarity(self, points: List[Dict], question: str, embedding: List[float], metadata: Dict, model_id: str = "") -> str:
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
            # # identify the distance metric to compare score with threshold
            # is_hit = False

            # logger.info(
            #     f"Current distance metric set on vector db is {self.vectorDb.distance_metric.lower()}")
            # if self.vectorDb.distance_metric.lower() in ['cosine', 'dot']:
            #     is_hit = score > self.distance_threshold

            # else:
            #     is_hit = score <= self.distance_threshold

            # if is_hit:
            result = self.handle_cache_hit(
                point_id=point.id, distance=score)

            if not result:
                console_logger.start_timer("Data doesn't seem to exist in the cache. Populating..")
                logger.info(
                    "Data doesn't seem to exist in the cache. Populating..")
                result = self.handle_cache_miss(
                    question=question, embedding=embedding, point_id=point.id, metadata=metadata, model_id=model_id)
                logger.info("Result: ", result)
            else:
                result = result['response_text']

            return result

        result = self.handle_cache_miss(
            question, embedding, metadata=metadata, model_id=model_id)

        console_logger.end_timer()
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

        console_logger.start_timer("Retrieving from cache ... ")
        response_cursor = self.cache_service.find(
            filter={"qdrant_id": point_id})
        console_logger.end_timer()

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

        console_logger.start_time("Cache not found .. fetching data")
        answer, response_text = self.generate_answer(
            question, model_id)
        
        console_logger.end_timer()

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
        console_logger.start_timer("Storing vectors to Qdrant")
        self.vectorDb.upsert(embeddings=embedding,
                             point_id=point_id, metadata=metadata)
        
        console_logger.end_timer()
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
            return result['data'], result['data']['response_text']

        else:
            try:
                logger.info("Getting answer from Ares.")
                result = requests.get_answer(question)
                return result['data'], result['data']['response_text']
            except Exception as e:
                raise RuntimeError(f"Error during 'generate_answer': {e}")