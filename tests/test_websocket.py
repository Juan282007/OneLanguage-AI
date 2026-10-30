import pytest
import jwt
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import web_service

TEST_JWT_SECRET = "t" * 64


@pytest.fixture(autouse=True)
def configure_test_jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)


def make_token(subject="user@example.com", expires_in=60, algorithm="HS256"):
    import time

    now = int(time.time())
    return jwt.encode(
        {"sub": subject, "iat": now, "exp": now + expires_in},
        TEST_JWT_SECRET,
        algorithm=algorithm,
    )


def authenticate(websocket):
    websocket.send_json({"type": "auth", "token": make_token()})


def test_default_allowed_origins_include_http_and_https(monkeypatch):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)

    assert web_service._allowed_origins() == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://localhost:5173",
        "https://127.0.0.1:5173",
    ]


def test_health_reports_loaded_model():
    response = TestClient(web_service.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["preprocessing_version"] == 7
    assert response.json()["labels"]


def test_websocket_rejects_disallowed_origin():
    with TestClient(web_service.app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/ws/recognize", headers={"origin": "https://untrusted.example"}
            ):
                pass
    assert rejected.value.code == 1008


def test_websocket_rejects_missing_origin():
    with TestClient(web_service.app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect("/ws/recognize"):
                pass
    assert rejected.value.code == 1008


@pytest.mark.parametrize("token", [None, "not-a-jwt", "expired"])
def test_websocket_rejects_missing_or_invalid_token(token):
    with TestClient(web_service.app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/ws/recognize", headers={"origin": "http://localhost:5173"}
            ) as websocket:
                websocket.send_json({"type": "auth", "token": token})
                websocket.receive_json()
    assert rejected.value.code == 1008


@pytest.mark.parametrize("algorithm", ["HS256", "HS384", "HS512"])
def test_websocket_accepts_backend_hmac_algorithms(algorithm):
    with TestClient(web_service.app) as client:
        with client.websocket_connect(
            "/ws/recognize", headers={"origin": "http://localhost:5173"}
        ) as websocket:
            websocket.send_json({"type": "auth", "token": make_token(algorithm=algorithm)})
            assert websocket.receive_json()["type"] == "ready"


def test_websocket_limits_connections_per_ip(monkeypatch):
    monkeypatch.setattr(web_service, "MAX_CONNECTIONS_PER_IP", 1)
    with TestClient(web_service.app) as client:
        with client.websocket_connect(
            "/ws/recognize", headers={"origin": "http://localhost:5173"}
        ) as first:
            authenticate(first)
            assert first.receive_json()["type"] == "ready"
            with pytest.raises(WebSocketDisconnect) as rejected:
                with client.websocket_connect(
                    "/ws/recognize", headers={"origin": "http://localhost:5173"}
                ) as second:
                    authenticate(second)
                    second.receive_json()
            assert rejected.value.code == 1008


def test_websocket_limits_frames_per_second(monkeypatch):
    monkeypatch.setattr(web_service, "MAX_FRAMES_PER_SECOND", 1)
    with TestClient(web_service.app) as client:
        with client.websocket_connect(
            "/ws/recognize", headers={"origin": "http://localhost:5173"}
        ) as websocket:
            authenticate(websocket)
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(b"invalid-jpeg")
            assert websocket.receive_json()["type"] == "error"
            with pytest.raises(WebSocketDisconnect) as rejected:
                websocket.send_bytes(b"invalid-jpeg")
                websocket.receive_json()
            assert rejected.value.code == 1008


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"invalid-jpeg", id="invalid-jpeg"),
        pytest.param(b"x" * (web_service.MAX_FRAME_BYTES + 1), id="too-large"),
    ],
)
def test_websocket_reports_invalid_camera_frames(payload):
    with TestClient(web_service.app) as client:
        with client.websocket_connect(
            "/ws/recognize", headers={"origin": "http://localhost:5173"}
        ) as websocket:
            authenticate(websocket)
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(payload)
            assert websocket.receive_json()["type"] == "error"
