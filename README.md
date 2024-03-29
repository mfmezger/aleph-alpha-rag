[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

# RAG with Aleph Alpha & Langchain & Qdrant Vector Database


- [RAG with Aleph Alpha \& Langchain \& Qdrant Vector Database](#rag-with-aleph-alpha--langchain--qdrant-vector-database)
  - [Quickstart](#quickstart)
  - [Project Description](#project-description)
  - [Semantic Search](#semantic-search)
  - [Architecture](#architecture)
  - [Components](#components)
  - [Secret Management](#secret-management)
  - [Installation \& Development Backend](#installation--development-backend)
  - [Development Frontend](#development-frontend)
  - [Vector Database](#vector-database)
  - [Qdrant API Key](#qdrant-api-key)



## Quickstart
To run the complete system with docker use this command:

```bash
git clone https://github.com/mfmezger/aleph-alpha-rag.git
cd aleph-alpha-rag
```
Create a .env file from the .env-template and set the qdrant api key. For tests just set it to test.
QDRANT_API_KEY="test"

Then start the system with
```bash
  docker compose up -d
```

Then go to http://127.0.0.1:8001/docs or http://127.0.0.1:8001/redoc to see the API documentation.

Frontend: localhost:8501
Qdrant Dashboard: localhost:6333/dashboard


## Project Description
This project is a conversational aleph_alpha_rag that uses Aleph Alpha and OpenAI Large Language Models to generate responses to user queries. The aleph_alpha_rag also includes a vector database and a REST API built with FastAPI.

Features
- Uses Aleph Alpha to generate responses to user queries.
- Includes a vector database to store and retrieve information.
- Provides a REST API built with FastAPI for easy integration with other applications.
- Has a basic gui.

## Semantic Search
![Semantic Search Architecture](resources/search_flow.png)

Semantic search is an advanced search technique that aims to understand the meaning and context of a user's query, rather than matching keywords. It involves natural language processing (NLP) and machine learning algorithms to analyze and interpret user intent, synonyms, relationships between words, and the structure of content. By considering these factors, semantic search improves the accuracy and relevance of search results, providing a more intuitive and personalized user experience.

## Architecture
![Semantic Search Architecture](resources/Architecture.png)

## Components

Langchain is a library for natural language processing and machine learning. FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. A Vectordatabase is a database that stores vectors, which can be used for similarity searches and other machine learning tasks.

## Secret Management

Two ways to manage your api keys are available, the easiest approach is to sent the api token in the request as the token.
Another possiblity is to create a .env file and add the api token there.
If you use OpenAI from Azure or OpenAI directly you need to set the correct parameters in the .env file.


## Installation & Development Backend

First install Python Dependencies:

```bash
pip install poetry
poetry install
```

Start the complete system with:

```bash
docker compose up -d
```

To run the Qdrant Database local just run:

```bash
docker compose up qdrant
```


To run the Backend use this command in the root directory:

```bash
poetry run uvicorn aleph_alpha_rag.api:app --reload
```

To run the tests you can use this command:

```bash
poetry run coverage run -m pytest -o log_cli=true -vvv tests
```

To update all dependencies use this command:

```bash
poetry add langchain@latest streamlit@latest loguru@latest omegaconf@latest aleph-alpha-client@latest python-dotenv@latest httpx@latest python-multipart@latest fastapi@latest grpcio@latest grpcio-tools@latest qdrant-client@latest uvicorn@latest nltk@latest pypdfium2@latest lingua-language-detector@latest pyarrow@latest aiohttp@latest
```

## Development Frontend

To run the Frontend use this command in the root directory:

```bash
poetry run streamlit run gui.py --theme.base="dark"
```

## Vector Database

Qdrant Dashboard is available at http://127.0.0.1:6333/dashboard. There you need to enter the api key.



## Qdrant API Key
To use the Qdrant API you need to set the correct parameters in the .env file.
QDRANT_API_KEY is the API key for the Qdrant API.
And you need to change it in the qdrant.yaml file in the config folder.
