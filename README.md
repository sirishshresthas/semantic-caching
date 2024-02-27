# Semantic Caching 

## Overview
The Semantic Caching is designed to enhance the efficiency of semantic search operations by caching the results of queries. This approach reduces the need for repeated computations and trip to LLM for similar queries, thus speeding up response times and decreasing computational load and cost. The project leverages a sentence transformer model for semantic understanding and Qdrant vector database for storing and searching embeddings.

## Features
- **Semantic Understanding**: Utilizes the `all-mpnet-base-v2` model to encode questions and understand their semantics.
- **Efficient Caching**: Caches query results and uses a similarity threshold to determine cache hits, improving response times for similar queries.
- **Vector Database Integration**: Employs Qdrant vector database to store question embeddings, facilitating efficient similarity searches.

## Setup

### Requirements
- Python 3.11+
- MongoDB - Connection string is required for local or cloud
- Qdrant Vector DB - URL and API Key are required
- ARES API - Create/retrieve an API Key from `https://api.traversaal.ai/`. Visit `https://docs.traversaal.ai/docs/intro` for docs
- Sentence Transformers
- Any additional dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```
   git clone git@github.com:sirishshresthas/sematic-caching.git
   ```
2. Navigate into the project directory:
   ```
   cd semantic-caching
   ```
3. There are 3 ways to use semantic-caching

    a. Run Docker container
    - Install Docker by following the instructions from [here](https://docs.docker.com/engine/install/)
    - run the docker compose using the following command
    ```
    docker compose up --build
    ```
    - Navigate to `localhost:8888` after docker runs to begin using Jupyterlab notebook. 

    b. If using conda, download and install miniconda or anaconda. Run the following command to create a conda environment. 
    ```
    conda env create -f conda.yml
    ```

    c. Install the required Python packages using pip:
    ```
    pip install -r requirements.txt
    ```

    After installing the packages, open `main.ipynb` file to see example how-to

### Environment Variable
Following environment variables are required in the .env file

    ```
    PROJECT_ENVIRONMENT=dev

    ## Quadrant
    VECTORDB_URL= #Qdrant URL
    VECTORDB_API_KEY= #Qdrant API Key

    MONGODB_CONNECTION_STRING= # Mongodb connection string
    DATABASE_NAME= # Mongdb database name
    COLLECTION_NAME= # Mongodb collection name

    ARES_URL= # Ares URL
    ARES_API_KEY=# Ares Api Key

    ```

## Usage

### Initializing the Semantic Caching System
To start using the semantic caching system, you need to instantiate the `SemanticCaching` class.

```python
from semantic_caching import SemanticCaching

semantic_caching = SemanticCaching()
```

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
semantic_caching.distance_threshold = 0.9  # Set a new threshold
```

### Saving Cache to Database
The caching system automatically saves cached entries to a MongoDB database. This process is handled internally but can be manually invoked if needed.

```python
semantic_caching.save_cache()
```

## Contributing
Contributions to the Semantic Caching Project are welcome! Please submit pull requests or open issues to propose changes or report bugs.

## License
Specify your project's license here.

---

Remember to replace `<repository-url>` with the actual URL of your project's repository and adjust any specific details according to your project's setup and requirements.