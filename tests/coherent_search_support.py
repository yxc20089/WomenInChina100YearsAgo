from __future__ import annotations

import gzip
import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar, final, override
from uuid import uuid4

import pytest
from tests.coherent_search_fake_models import AliasEnvelope, BulkAction
from tests.coherent_search_fake_filters import matches_dense_identity
from wic_history.coherent_search import (
    COHERENT_ALIAS,
    CoherentEmbedding,
    CoherentSource,
    FrozenProjectionManifest,
    ProjectionArticle,
)
from wic_history.evidence import Point, Polygon, SourcePointer
from wic_history.search import DEFAULT_ALIAS
from wic_history.semantic_repository import CoherentTextBundle, CoherentTextSegment


TestJson = str | int | float | bool | None | list["TestJson"] | dict[str, "TestJson"]


@final
class QueryEmbedder:
    model_name = "BAAI/bge-m3"
    model_revision = "model-revision"
    configuration_sha256 = "f" * 64

    def encode_query(self, query: str) -> list[float]:
        assert query == "女子教育"
        return [0.01] * 1024


class OpenSearchFake(BaseHTTPRequestHandler):
    documents: ClassVar[dict[str, str]] = {}
    aliases: ClassVar[dict[str, str]] = {}
    requests: ClassVar[list[tuple[str, str, TestJson]]] = []
    fail_bulk: ClassVar[bool] = False
    misleading_bulk: ClassVar[bool] = False
    bulk_envelope: ClassVar[TestJson] = None
    count_envelope: ClassVar[TestJson] = None
    refresh_envelope: ClassVar[TestJson] = None
    alias_lookup_envelope: ClassVar[TestJson] = None
    alias_update_envelope: ClassVar[TestJson] = None
    filter_dense_identity: ClassVar[bool] = False

    @override
    def log_message(self, format: str, *args: str | int | float) -> None:
        return

    def _json_body(self) -> str | None:
        length = int(self.headers.get("content-length", "0"))
        if length == 0:
            return None
        payload = self.rfile.read(length)
        if self.headers.get("content-encoding") == "gzip":
            payload = gzip.decompress(payload)
        return payload.decode()

    def _reply(self, status: int, value: TestJson) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        _ = self.wfile.write(payload)

    def _reply_search(self, body: str | None) -> None:
        documents = self.documents.values()
        if self.filter_dense_identity and body is not None and '"knn"' in body:
            documents = tuple(
                document
                for document in documents
                if matches_dense_identity(body, document)
            )
        hits = ",".join(
            f'{{"_index":{json.dumps(self.aliases[COHERENT_ALIAS])},"_score":1.5,"_source":{document}}}'
            for document in documents
        )
        payload = f'{{"hits":{{"hits":[{hits}]}}}}'.encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        _ = self.wfile.write(payload)

    def do_PUT(self) -> None:
        self.requests.append(("PUT", self.path, self._json_body()))
        self._reply(200, {"acknowledged": True})

    def do_GET(self) -> None:
        self.requests.append(("GET", self.path, None))
        if not self.path.startswith("/_alias/"):
            self._reply(200, {"version": {"number": "3.2.0"}})
            return
        if self.alias_lookup_envelope is not None:
            self._reply(200, self.alias_lookup_envelope)
            return
        alias = self.path.split("?")[0].rsplit("/", 1)[1]
        index = self.aliases.get(alias)
        if index is None:
            self._reply(404, {"error": "alias missing", "status": 404})
        else:
            self._reply(200, {index: {"aliases": {alias: {}}}})

    def do_POST(self) -> None:
        body = self._json_body()
        self.requests.append(("POST", self.path, body))
        if self.path.startswith("/_bulk"):
            self._bulk(body)
        elif self.path.endswith("/_count"):
            self._reply(
                200,
                self.count_envelope
                if self.count_envelope is not None
                else {"count": len(self.documents)},
            )
        elif self.path == "/_aliases":
            self._aliases(body)
        elif self.path.endswith("/_refresh"):
            self._reply(
                200,
                self.refresh_envelope
                if self.refresh_envelope is not None
                else {"_shards": {"total": 1, "successful": 1, "failed": 0}},
            )
        elif self.path.endswith("/_search"):
            self._reply_search(body)
        else:
            self._reply(200, {"_shards": {"successful": 1}})

    def _bulk(self, body: str | None) -> None:
        if self.bulk_envelope is not None:
            self._reply(200, self.bulk_envelope)
            return
        assert body is not None
        lines = body.strip().splitlines()
        items: list[TestJson] = []
        for position in range(0, len(lines), 2):
            action = BulkAction.model_validate_json(lines[position])
            self.documents[action.index.document_id] = lines[position + 1]
            status = 500 if self.fail_bulk or self.misleading_bulk else 201
            item: dict[str, TestJson] = {"status": status}
            if self.fail_bulk:
                item["error"] = {"type": "fixture_failure"}
            items.append({"index": item})
        self._reply(200, {"errors": self.fail_bulk, "items": items})

    def _aliases(self, body: str | None) -> None:
        if self.alias_update_envelope is not None:
            self._reply(200, self.alias_update_envelope)
            return
        assert body is not None
        envelope = AliasEnvelope.model_validate_json(body)
        for action in envelope.actions:
            if (
                action.remove is not None
                and action.remove.must_exist is True
                and self.aliases.get(action.remove.alias) != action.remove.index
            ):
                self._reply(404, {"error": "required alias target is missing"})
                return
        for action in envelope.actions:
            if action.remove is not None:
                _ = self.aliases.pop(action.remove.alias, None)
            elif action.add is not None:
                self.aliases[action.add.alias] = action.add.index
        self._reply(200, {"acknowledged": True})


@pytest.fixture
def search_server() -> Iterator[str]:
    OpenSearchFake.documents = {}
    OpenSearchFake.requests = []
    OpenSearchFake.fail_bulk = False
    OpenSearchFake.misleading_bulk = False
    OpenSearchFake.bulk_envelope = None
    OpenSearchFake.count_envelope = None
    OpenSearchFake.refresh_envelope = None
    OpenSearchFake.alias_lookup_envelope = None
    OpenSearchFake.alias_update_envelope = None
    OpenSearchFake.filter_dense_identity = False
    OpenSearchFake.aliases = {
        DEFAULT_ALIAS: "wic-regions-v2",
        COHERENT_ALIAS: "wic-coherent-units-build-old",
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), OpenSearchFake)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def manifest() -> FrozenProjectionManifest:
    revision_id, unit_id = uuid4(), uuid4()
    segments: list[CoherentTextSegment] = []
    sources: list[CoherentSource] = []
    content_parts = ("女子學校創立", "續招學生二十名")
    position = 0
    for sequence, text in enumerate(content_parts):
        region_id, page_id, text_id, selection_id = uuid4(), uuid4(), uuid4(), uuid4()
        end = position + len(text)
        segments.append(
            CoherentTextSegment(
                sequence,
                region_id,
                page_id,
                text_id,
                selection_id,
                0,
                len(text),
                position,
                end,
                text,
                "body",
                None,
            )
        )
        sources.append(
            CoherentSource(
                sequence,
                SourcePointer(
                    source_uri="s3://archive/volume.pdf",
                    source_sha256="a" * 64,
                    page_id=page_id,
                    derivative_id=uuid4(),
                    image_uri=f"s3://archive/page-{sequence}.png",
                    image_sha256="b" * 64,
                    evidence_tier="historian_selected_gold",
                    volume_number=7,
                    publication_year=1925 + sequence,
                    page_number=12 + sequence,
                    region_id=region_id,
                    text_version_id=text_id,
                    text_selection_id=selection_id,
                    polygon=Polygon(
                        points=[Point(x=0, y=0), Point(x=1, y=0), Point(x=1, y=1)]
                    ),
                    text_start=0,
                    text_end=len(text),
                ),
                (f"warning-{sequence}",),
            )
        )
        position = end + 1
    content = "\n".join(content_parts)
    content_sha256 = hashlib.sha256(content.encode()).hexdigest()
    bundle = CoherentTextBundle(
        revision_id, content, "c" * 64, tuple(segments), (), content_sha256, "e" * 64
    )
    embedding = CoherentEmbedding(
        revision_id,
        "coherent_unit_revision",
        "BAAI/bge-m3",
        "model-revision",
        "c" * 64,
        content_sha256,
        "f" * 64,
        tuple([0.01] * 1024),
    )
    article = ProjectionArticle(unit_id, "女子教育消息", bundle, tuple(sources))
    return FrozenProjectionManifest.freeze((article,), (embedding,))
