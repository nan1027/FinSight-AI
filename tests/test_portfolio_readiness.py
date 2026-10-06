from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_root_serves_frontend_shell() -> None:
    response = client.get("/")
    assert response.status_code == 200, response.text
    assert "FinSight AI" in response.text
    assert "risk" in response.text.lower()


def test_cors_allows_browser_clients() -> None:
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200, response.text
    assert "access-control-allow-origin" in response.headers
