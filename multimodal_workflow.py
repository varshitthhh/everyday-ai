"""Standalone Gemma 4 multimodal workflow.

The workflow is deliberately independent of the application that inspired it:

1. Prepare each image and extract lightweight EXIF metadata.
2. Ask a local Gemma model for one structured photo-memory record.
3. Ask Gemma to synthesize the photo records into a trip memory.

Input photos are supplied by path and are never copied into this repository.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar

import ollama
from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import BaseModel, Field


SchemaT = TypeVar("SchemaT", bound=BaseModel)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "gemma4:e4b-128k")
DEFAULT_MAX_IMAGE_EDGE_PX = int(os.getenv("MAX_IMAGE_EDGE_PX", "1280"))


@dataclass(frozen=True)
class PreparedPhoto:
    photo_id: str
    image: bytes
    metadata: dict[str, Any]


class PhotoMemoryAnalysis(BaseModel):
    scene_summary: str = Field(min_length=1, max_length=500)
    memory_caption: str = Field(min_length=1, max_length=240)
    place_type: str = Field(min_length=1, max_length=120)
    visible_activities: list[str] = Field(default_factory=list, max_length=12)
    visible_objects: list[str] = Field(default_factory=list, max_length=20)
    sensory_details: list[str] = Field(default_factory=list, max_length=12)
    inferred_interest_signals: list[str] = Field(default_factory=list, max_length=12)
    mood: str = Field(min_length=1, max_length=120)
    uncertainty_notes: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)


class MemorableMoment(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=500)
    evidence_photo_ids: list[str] = Field(default_factory=list)


class TripMemorySynthesis(BaseModel):
    narrative_summary: str = Field(min_length=1, max_length=1000)
    inferred_interests: list[str] = Field(default_factory=list, max_length=16)
    recurring_themes: list[str] = Field(default_factory=list, max_length=16)
    memorable_moments: list[MemorableMoment] = Field(default_factory=list, max_length=20)
    evidence_photo_ids: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list, max_length=10)


def prepare_photo(
    image_path: Path,
    *,
    max_edge_px: int = DEFAULT_MAX_IMAGE_EDGE_PX,
) -> PreparedPhoto:
    """Resize an image for local inference and extract useful EXIF metadata."""

    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    try:
        with Image.open(image_path) as source:
            metadata = _extract_metadata(source, image_path)
            image = source.copy()
            image.thumbnail((max_edge_px, max_edge_px))
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="JPEG" if image.mode == "RGB" else "PNG")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image: {image_path}") from exc

    return PreparedPhoto(
        photo_id=image_path.name,
        image=output.getvalue(),
        metadata=metadata,
    )


def analyze_photo(
    prepared_photo: PreparedPhoto,
    *,
    model: str = DEFAULT_MODEL,
) -> PhotoMemoryAnalysis:
    """Turn one prepared image into a validated structured memory record."""

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _photo_system_instruction()},
            {
                "role": "user",
                "content": _photo_prompt(prepared_photo.metadata),
                "images": [prepared_photo.image],
            },
        ],
        format=_inline_schema_refs(PhotoMemoryAnalysis.model_json_schema()),
        options={"temperature": 0},
    )
    return PhotoMemoryAnalysis.model_validate_json(_response_content(response))


def synthesize_trip(
    photo_memories: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
) -> TripMemorySynthesis:
    """Synthesize validated photo records with a text-only Gemma call."""

    valid_photo_ids = {str(item["photo_id"]) for item in photo_memories}
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": _trip_system_instruction()},
            {"role": "user", "content": _trip_prompt(photo_memories)},
        ],
        format=_inline_schema_refs(TripMemorySynthesis.model_json_schema()),
        options={"temperature": 0},
    )
    result = TripMemorySynthesis.model_validate_json(_response_content(response))
    referenced_ids = set(result.evidence_photo_ids)
    referenced_ids.update(
        photo_id
        for moment in result.memorable_moments
        for photo_id in moment.evidence_photo_ids
    )
    invalid_ids = sorted(referenced_ids - valid_photo_ids)
    if invalid_ids:
        raise ValueError(f"Model referenced unknown photo IDs: {invalid_ids}")
    return result


def run_workflow(
    photo_dir: Path,
    *,
    model: str = DEFAULT_MODEL,
    max_edge_px: int = DEFAULT_MAX_IMAGE_EDGE_PX,
) -> tuple[list[dict[str, Any]], TripMemorySynthesis]:
    """Run the complete workflow over supported images in a directory."""

    photo_paths = sorted(
        path
        for path in photo_dir.expanduser().resolve().iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not photo_paths:
        raise ValueError(f"No supported images found in {photo_dir}")

    photo_memories: list[dict[str, Any]] = []
    for image_path in photo_paths:
        prepared = prepare_photo(image_path, max_edge_px=max_edge_px)
        analysis = analyze_photo(prepared, model=model)
        photo_memories.append(
            {
                "photo_id": prepared.photo_id,
                "metadata": prepared.metadata,
                "analysis": analysis.model_dump(),
            }
        )

    return photo_memories, synthesize_trip(photo_memories, model=model)


def _photo_system_instruction() -> str:
    return """You are a local multimodal travel-memory analyst.
Analyze the supplied travel photo and metadata. Use only the supplied inputs.
Do not invent exact places, dates, events, people, or coordinates.
Return exactly one JSON object matching the requested schema.
If evidence is weak, explain that in uncertainty_notes.""".strip()


def _trip_system_instruction() -> str:
    return """You are a local travel-memory synthesizer.
Synthesize the supplied photo records into a coherent trip memory.
Use only the supplied records and cite evidence using their photo_id values.
Do not add outside facts or invent events, places, dates, or people.
Return exactly one JSON object matching the requested schema.""".strip()


def _photo_prompt(metadata: dict[str, Any]) -> str:
    return "Photo metadata:\n" + json.dumps(metadata, indent=2, default=str)


def _trip_prompt(photo_memories: list[dict[str, Any]]) -> str:
    return "Photo memory records:\n" + json.dumps(photo_memories, indent=2, default=str)


def _response_content(response: Any) -> str:
    message = response["message"] if isinstance(response, dict) else response.message
    content = message["content"] if isinstance(message, dict) else message.content
    if not content:
        raise ValueError("Ollama returned an empty response")
    return str(content).strip()


def _inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    definitions = schema.get("$defs", {})

    def expand(value: Any) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if reference:
                return expand(definitions[reference.rsplit("/", 1)[-1]])
            return {
                key: expand(item)
                for key, item in value.items()
                if key != "$defs"
            }
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(schema)


def _extract_metadata(image: Image.Image, image_path: Path) -> dict[str, Any]:
    exif = image.getexif()
    tags = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
    gps = tags.get("GPSInfo", {})
    metadata: dict[str, Any] = {
        "filename": image_path.name,
        "captured_at": _first_value(tags, "DateTimeOriginal", "DateTimeDigitized", "DateTime"),
        "width": image.width,
        "height": image.height,
    }
    latitude = _gps_coordinate(gps, 2, 1)
    longitude = _gps_coordinate(gps, 4, 3)
    if latitude is not None:
        metadata["latitude"] = latitude
    if longitude is not None:
        metadata["longitude"] = longitude
    return metadata


def _first_value(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value:
            return str(value)
    return None


def _gps_coordinate(gps: Any, coordinate_key: int, reference_key: int) -> float | None:
    if not isinstance(gps, dict):
        return None
    coordinate = gps.get(coordinate_key)
    reference = gps.get(reference_key)
    if not coordinate or not reference or len(coordinate) != 3:
        return None
    try:
        degrees, minutes, seconds = (float(value) for value in coordinate)
        decimal = degrees + minutes / 60 + seconds / 3600
        return -decimal if str(reference).upper() in {"S", "W"} else decimal
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo_dir", type=Path, help="Directory containing input photos")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-edge-px", type=int, default=DEFAULT_MAX_IMAGE_EDGE_PX)
    parser.add_argument("--output", type=Path, default=Path("outputs/trip_memory.json"))
    args = parser.parse_args()

    photo_memories, trip_memory = run_workflow(
        args.photo_dir,
        model=args.model,
        max_edge_px=args.max_edge_px,
    )
    payload = {"photo_memories": photo_memories, "trip_memory": trip_memory.model_dump()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(photo_memories)} photo memories and one trip memory to {args.output}")


if __name__ == "__main__":
    main()
