# Copyright (c) Microsoft. All rights reserved.

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.testclient import TestClient


def make_client() -> TestClient:
    app = FastAPI()

    async def upload(label: str = Form(...), document: UploadFile = File(...)) -> dict[str, str]:
        content = await document.read()
        return {"label": label, "filename": document.filename or "", "content": content.decode()}

    app.post("/upload")(upload)
    return TestClient(app)


def test_multipart_upload_with_legacy_fastapi() -> None:
    with make_client() as client:
        response = client.post(
            "/upload", data={"label": "offline"}, files={"document": ("sample.txt", b"content", "text/plain")}
        )
    assert response.status_code == 200
    assert response.json() == {"label": "offline", "filename": "sample.txt", "content": "content"}


def test_multipart_requires_boundary() -> None:
    with make_client() as client:
        response = client.post("/upload", content=b"invalid", headers={"Content-Type": "multipart/form-data"})
    assert response.status_code == 400
    assert "boundary" in response.json()["detail"].lower()


def test_multipart_rejects_missing_file() -> None:
    with make_client() as client:
        response = client.post("/upload", files={"label": (None, "offline")})
    assert response.status_code == 422
