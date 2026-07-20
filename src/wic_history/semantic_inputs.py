"""Immutable coherent text and image inputs for semantic processing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from .source_provenance import Polygon

if TYPE_CHECKING:
    from uuid import UUID


class _PolygonPointInput(TypedDict):
    x: int | float
    y: int | float


class _PolygonMappingInput(TypedDict):
    points: list[_PolygonPointInput]


PolygonInput = Polygon | _PolygonMappingInput | None


@dataclass(frozen=True, slots=True)
class CoherentTextSegment:
    """One selected reviewed-text span with its source-image coordinates."""

    sequence_number: int
    region_id: UUID
    page_id: UUID
    text_version_id: UUID
    selection_id: UUID
    text_start: int
    text_end: int
    composite_start: int
    composite_end: int
    text: str
    role: str
    polygon: PolygonInput


@dataclass(frozen=True, slots=True)
class PageImageReference:
    """An immutable page derivative supplied with semantic text segments."""

    page_id: UUID
    derivative_id: UUID
    image_uri: str
    image_sha256: str
    media_type: str
    width: int
    height: int
    region_ids: tuple[UUID, ...]


@dataclass(frozen=True, slots=True)
class CoherentTextBundle:
    """Canonical reviewed text and its model-visible provenance inputs."""

    coherent_unit_revision_id: UUID
    content: str
    input_sha256: str
    segments: tuple[CoherentTextSegment, ...]
    page_images: tuple[PageImageReference, ...]
    content_sha256: str = ""
    multimodal_input_sha256: str = ""


def _polygon_identity(polygon: PolygonInput) -> _PolygonMappingInput | None:
    if polygon is None:
        return None
    if isinstance(polygon, Polygon):
        return {
            "points": [{"x": point.x, "y": point.y} for point in polygon.points],
        }
    return polygon


def semantic_multimodal_input_sha256(bundle: CoherentTextBundle) -> str:
    """Hash every text-segment and page-image field visible to the semantic model."""
    identity = {
        "reviewed_text_input_sha256": bundle.input_sha256,
        "segments": [
            {
                "region_id": str(item.region_id),
                "page_id": str(item.page_id),
                "text_version_id": str(item.text_version_id),
                "text_start": item.text_start,
                "text_end": item.text_end,
                "text": item.text,
                "role": item.role,
                "polygon": _polygon_identity(item.polygon),
            }
            for item in bundle.segments
        ],
        "page_images": [
            {
                "page_id": str(item.page_id),
                "derivative_id": str(item.derivative_id),
                "image_uri": item.image_uri,
                "image_sha256": item.image_sha256,
                "media_type": item.media_type,
                "width": item.width,
                "height": item.height,
                "region_ids": [str(value) for value in item.region_ids],
            }
            for item in bundle.page_images
        ],
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ).hexdigest()
