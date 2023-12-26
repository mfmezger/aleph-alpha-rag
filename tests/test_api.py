"""API Tests."""

from fastapi.testclient import TestClient

from aleph_alpha_rag.api import app

client: TestClient = TestClient(app)


def test_post_explain_question_answer():
    """Test the explain question answer endpoint."""
    # Replace with valid data for your application
    data = {"qa": {"search": {"query": "Your test query", "token": "Your test token"}, "language": "detect", "history": 0}, "threshold_explain": 0.0}
    response = client.post("/explanation/explain-qa", json=data)
    assert response.status_code == 200
