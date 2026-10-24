import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import web_service


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
            assert websocket.receive_json()["type"] == "ready"
            websocket.send_bytes(payload)
            assert websocket.receive_json()["type"] == "error"
