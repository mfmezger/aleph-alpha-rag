"""API Tests."""

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
