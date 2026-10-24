# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The LanceDB Authors

import base64
import io
import os
import socket
from types import SimpleNamespace

import pytest

from lancedb.embeddings import get_registry


FILENAMES = [
    "image#1.png",
    pytest.param(
        "image?1.png",
        marks=pytest.mark.skipif(os.name == "nt", reason="? is not a Windows filename"),
    ),
    "image;1.png",
    "image 1.png",
    "image%20literal.png",
]


@pytest.fixture
def image_embedding(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("requests")
    monkeypatch.setenv("JINA_API_KEY", "test-not-a-real-key")

    def reject_network(*args, **kwargs):
        raise AssertionError("Image path tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    sent = []

    def fake_post(session, url, json):
        sent.append(json)
        return SimpleNamespace(
            json=lambda: {"data": [{"index": 0, "embedding": [0.25, 0.75]}]}
        )

    monkeypatch.setattr("requests.Session.post", fake_post)
    # Truncating a literal #, ? or ; opens a different, valid image.
    Image.new("RGB", (2, 3), color="blue").save(tmp_path / "image", format="PNG")

    def make_image(filename):
        path = tmp_path / filename
        Image.new("RGB", (4, 2), color="red").save(path, format="PNG")
        return path

    def assert_image_sent():
        assert len(sent) == 1
        payload = sent[0]["input"]
        assert len(payload) == 1
        with Image.open(io.BytesIO(base64.b64decode(payload[0]["image"]))) as image:
            assert image.size == (4, 2)
            assert image.getpixel((0, 0)) == (255, 0, 0)

    return get_registry().get("jina").create(), make_image, assert_image_sent


@pytest.mark.parametrize("filename", FILENAMES)
@pytest.mark.parametrize("as_string", [False, True])
def test_jina_image_embedding_preserves_local_filename(
    image_embedding, filename, as_string
):
    function, make_image, assert_image_sent = image_embedding
    path = make_image(filename)

    result = function.generate_image_embedding(str(path) if as_string else path)

    assert result == [0.25, 0.75]
    assert_image_sent()


@pytest.mark.parametrize("filename", FILENAMES)
def test_jina_image_embedding_decodes_file_uri(image_embedding, filename):
    function, make_image, assert_image_sent = image_embedding
    path = make_image(filename)

    assert function.generate_image_embedding(path.as_uri()) == [0.25, 0.75]
    assert_image_sent()


@pytest.mark.parametrize("filename", FILENAMES)
def test_jina_source_embedding_accepts_literal_path(image_embedding, filename):
    function, make_image, assert_image_sent = image_embedding

    assert function.compute_source_embeddings([make_image(filename)]) == [[0.25, 0.75]]
    assert_image_sent()


def test_jina_query_embedding_accepts_literal_path(image_embedding):
    function, make_image, assert_image_sent = image_embedding

    assert function.compute_query_embeddings(make_image("image#1.png")) == [
        [0.25, 0.75]
    ]
    assert_image_sent()


def test_jina_image_embedding_preserves_http_url(image_embedding, monkeypatch):
    function, make_image, assert_image_sent = image_embedding
    path = make_image("remote.png")
    url = "https://example.invalid/image%20name.png?size=2#view"
    downloaded = []

    def fake_retrieve(requested):
        downloaded.append(requested)
        return path.read_bytes()

    monkeypatch.setattr("lancedb.embeddings.jinaai.url_retrieve", fake_retrieve)

    assert function.generate_image_embedding(url) == [0.25, 0.75]
    assert downloaded == [url]
    assert_image_sent()


@pytest.mark.parametrize("as_bytes", [False, True])
def test_jina_image_embedding_keeps_memory_inputs(image_embedding, as_bytes):
    Image = pytest.importorskip("PIL.Image")
    function, make_image, assert_image_sent = image_embedding
    path = make_image("memory.png")
    with Image.open(path) as image:
        value = path.read_bytes() if as_bytes else image
        assert function.generate_image_embedding(value) == [0.25, 0.75]
    assert_image_sent()


def test_jina_image_embedding_rejects_unsupported_uri(image_embedding):
    function, _, _ = image_embedding

    with pytest.raises(NotImplementedError, match="local and http"):
        function.generate_image_embedding("ftp://example.invalid/image.png")
