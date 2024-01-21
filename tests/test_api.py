"""API Tests."""
from io import BytesIO

from fastapi import UploadFile
from fastapi.testclient import TestClient

from aleph_alpha_rag.api import app

client: TestClient = TestClient(app)


def test_create_collection():
    """Test the create collection endpoint."""
    collection_name = "test_collection"
    embeddings_size = 5120
    response = client.post(f"/collection/create/{collection_name}/{embeddings_size}")
    assert response.status_code == 200
    assert response.json() == {"detail": f"SUCCESS: Collection {collection_name} created."}


def test_post_embedd_documents():
    """Test the post embedd documents endpoint."""
    # Create a dummy file for testing
    data = {"key": "value"}
    data_bytes = bytes(str(data), "utf-8")
    file = UploadFile("test_file.txt", BytesIO(data_bytes))

    response = client.post("/embeddings/documents", files={"files": (file, data_bytes, "text/plain")}, data={"token": "test_token", "collection_name": "test_collection"})

    assert response.status_code == 200
    assert response.json() == {"status": "success", "files": ["test_file.txt"]}
