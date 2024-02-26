import uuid
from typing import Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from requests.exceptions import ConnectionError, Timeout

from src.core.utilities import settings


class VectorDB(object):

    def __init__(self, vector_size: int = 300, payload: Optional[Dict] = None):
        if payload is None:
            payload = {}
        self.client = QdrantClient(
            url=settings.VECTORDB_URL, api_key=settings.VECTORDB_API_KEY)
        self._collection_name = settings.COLLECTION_NAME
        self.get_collection(vector_size=vector_size)
        self.payload = payload

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @collection_name.setter
    def collection_name(self, value: str):
        self._collection_name = value

    def get_collection(self, vector_size: int, distance: str = "cosine"):
        try:
            collection_response = self.client.get_collections()
            collections: List = [
                cols.name for cols in collection_response.collections]
            if self.collection_name not in collections:
                collection = self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=self.distance_metric(distance)),
                )
                return collection
            else:
                return self.client.get_collection(self.collection_name)
        except ConnectionError:
            raise ValueError("Failed to connect to the Qdrant server.")
        except Timeout:
            raise ValueError("The request to Qdrant server timed out.")
        except Exception as e:
            # This is a catch-all for any other exceptions not specifically caught above.
            raise ValueError(f"An unexpected error occurred: {e}")

    def upsert(self, embeddings: List[float], point_id: uuid = None, metadata=None):

        print(embeddings)

        if metadata is None:
            metadata = {}

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embeddings,
                        payload=metadata,
                )]
            )
        except ConnectionError:
            raise ValueError(
                "Failed to connect to the Qdrant server for upsert operation.")
        except Timeout:
            raise ValueError("The upsert request to Qdrant server timed out.")
        except Exception as e:
            # This is a catch-all for any other exceptions not specifically caught above.
            raise ValueError(
                f"An unexpected error occurred during upsert: {e}")

    def search(self, search_key: str = "", search_value: str = "", limit: int = 3, query_vector: List[List[float]] = [], hnsw_ef: int = 128, exact: bool = False) -> Optional[Dict]:

        try:
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                # query_filter=models.Filter(
                #     must=[
                #         models.FieldCondition(
                #             key=search_key,
                #             match=models.MatchValue(
                #                 value=search_value,
                #             ),
                #         )
                #     ]
                # ),
                # search_params=models.SearchParams(
                #     hnsw_ef=hnsw_ef, exact=exact),
                limit=limit,
            )
        except Exception as e:
            raise ValueError(e)

    def distance_metric(self, distance: str = "cosine") -> models.Distance:
        metric = {
            "cosine": models.Distance.COSINE,
            "euclidean": models.Distance.EUCLID,
            "dot": models.Distance.DOT,
            "manhattan": models.Distance.MANHATTAN
        }
        return metric.get(distance, models.Distance)
