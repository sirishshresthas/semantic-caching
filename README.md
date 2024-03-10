# Semantic Caching

## Overview
Semantic Caching enhances the efficiency of semantic search operations by caching query results, thereby reducing redundant computations and trips to the LLM for similar queries. This approach speeds up response times, decreases computational load, and minimizes costs. The project utilizes a sentence transformer model for semantic understanding and integrates with the Qdrant vector database for storing and searching embeddings.

## Features
- **Semantic Understanding**: Employs the `all-mpnet-base-v2` model to encode questions and grasp their semantics effectively. A different encoder can be used during instantiating `SemanticCaching` class.
- **Efficient Caching**: Caches query results and employs a similarity threshold to determine cache hits, thereby improving response times for similar queries.
- **Vector Database Integration**: Utilizes the Qdrant vector database to store question embeddings, facilitating efficient similarity searches.

## Setup

### Requirements
- Python 3.11+
- MongoDB - Connection string is required for local or cloud
- Qdrant Vector DB - URL and API Key are required
- ARES API - Create/retrieve an API Key from [Traversaal Ares API](https://api.traversaal.ai/). Visit the [documentation](https://docs.traversaal.ai/docs/intro) for further details.
- Sentence Transformers
- Any additional dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```
   git clone git@github.com:sirishshresthas/semantic-caching.git
   ```
2. Navigate into the project directory:
   ```
   cd semantic-caching
   ```
3. There are 3 ways to start using Semantic Caching:

    a. **Run Docker container:**
    - Install Docker by following the instructions [here](https://docs.docker.com/engine/install/).
    - Run the docker compose using the following command:
    ```
    docker compose up --build
    ```
    - Navigate to `localhost:8888` after Docker runs to begin using JupyterLab notebook.

    b. **Conda environment:**
    - If you want to use conda environment, download and install Miniconda or Anaconda. Then run the following command to create a conda environment:
    ```
    conda env create -f conda.yml
    ```

    c. **Install Python packages using pip:**
    ```
    pip install -r requirements.txt
    ```

    After installing the packages, open the `main.ipynb` file to see examples of how to use it.

### Environment Variables
The following environment variables are required in the `.env` file:

    ```
    ## Qdrant
    VECTORDB_URL= # Qdrant URL
    VECTORDB_API_KEY= # Qdrant API Key

    MONGODB_CONNECTION_STRING= # MongoDB connection string
    DATABASE_NAME= # MongoDB database name
    COLLECTION_NAME= # MongoDB collection name

    ARES_URL= # Ares URL
    ARES_API_KEY= # Ares API Key

    ```

## Usage

### Initializing the Semantic Caching System
To start using the semantic caching system, you need to instantiate the `SemanticCaching` class.

```python
from semantic_caching import SemanticCaching

semantic_caching = SemanticCaching()
```

## Adjusting Vector DB distance metric
While creating the Qdrant collection for the first time, the distance metric is currently set to `cosine` by default. If you're creating Qdrant collection for the first time, you have the option to set the distance metric to one of the following: `dot`, `cosine`, `euclidean`, or `manhattan`

### Asking Questions
To process a question through the caching system, use the `ask` method. This method checks if an answer to a similar question has already been cached. If so, it retrieves the answer from the cache; otherwise, it generates a new answer, caches it, and then returns it.

```python
question = "What is the capital of France?"
answer = semantic_caching.ask(question)
print(answer)
```

### Adjusting the Similarity Threshold
You can adjust the similarity threshold for determining cache hits according to your needs. A higher threshold requires greater similarity for a cache hit, whereas a lower threshold is more lenient.

```python
semantic_caching.distance_threshold = 0.9 
```
