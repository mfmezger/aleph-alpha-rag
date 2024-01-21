"""The script to initialize the Qdrant db backend with aleph alpha."""

import os
from typing import Optional

from langchain_community.embeddings import AlephAlphaAsymmetricSemanticEmbedding
from langchain_community.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient, models

from aleph_alpha_rag.utils.configuration import load_config


@load_config(location="config/main.yml")
def get_db_connection(aleph_alpha_token: str, cfg: DictConfig, collection_name: Optional[str] = None) -> Qdrant:
    """Initializes a connection to the Qdrant DB.

    Args:
        cfg (DictConfig): The configuration file loaded via OmegaConf.
        aleph_alpha_token (str): The Aleph Alpha API token.

    Returns:
        Qdrant: The Qdrant DB connection.
    """
    embedding = AlephAlphaAsymmetricSemanticEmbedding(
        model=cfg.aleph_alpha_embeddings.model_name,
        aleph_alpha_api_key=aleph_alpha_token,
        normalize=cfg.aleph_alpha_embeddings.normalize,
        compress_to_size=cfg.aleph_alpha_embeddings.compress_to_size,
    )
    qdrant_client = QdrantClient(
        cfg.qdrant.url,
        port=cfg.qdrant.port,
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=cfg.qdrant.prefer_grpc,
    )

    if collection_name is None or collection_name == "":
        collection_name = cfg.qdrant.collection_name_aa

    logger.info(f"USING COLLECTION: {collection_name}")

    vector_db = Qdrant(client=qdrant_client, collection_name=collection_name, embeddings=embedding)
    logger.info("SUCCESS: Qdrant DB initialized.")

    return vector_db


def generate_collection(qdrant_client: QdrantClient, collection_name: str, embeddings_size: int):
    """Generate a collection for the Aleph Alpha Backend.

    Args:
        qdrant_client (_type_): _description_
        collection_name (_type_): _description_
        embeddings_size (_type_): _description_
    """
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
    )
    logger.info(f"SUCCESS: Collection {collection_name} created.")
