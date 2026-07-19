#!/usr/bin/env python3
"""Reproducible local LLM smoke suite for historical-Chinese KG ingestion.

This is a technical diagnostic, not the project benchmark.  It exercises the
three model responsibilities independently and retains every raw response.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "surface": {"type": "string", "minLength": 1},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                    "category": {
                        "type": "string",
                        "enum": [
                            "person_name",
                            "person_reference",
                            "place_name",
                            "organization_name",
                            "school_name",
                            "publication_name",
                        ],
                    },
                    "form": {
                        "type": "string",
                        "enum": [
                            "full_name",
                            "alternate_name",
                            "latin_name",
                            "shortened_surname",
                            "title_reference",
                            "anonymous_descriptor",
                            "named_entity",
                        ],
                    },
                },
                "required": ["surface", "start", "end", "category", "form"],
            },
        },
        "local_coreference": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "anaphor_start": {"type": "integer", "minimum": 0},
                    "antecedent_start": {"type": "integer", "minimum": 0},
                    "decision": {
                        "type": "string",
                        "enum": ["SAME_LOCAL", "INSUFFICIENT"],
                    },
                },
                "required": ["anaphor_start", "antecedent_start", "decision"],
            },
        },
    },
    "required": ["mentions", "local_coreference"],
}

RESOLUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["SAME", "DIFFERENT", "INSUFFICIENT"],
        },
        "supporting_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "contradiction_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "decision",
        "supporting_evidence_ids",
        "contradiction_evidence_ids",
    ],
}

CANDIDATE_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "candidate_id": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "person_name",
                            "person_reference",
                            "place_name",
                            "organization_name",
                            "school_name",
                            "publication_name",
                        ],
                    },
                    "form": {
                        "type": "string",
                        "enum": [
                            "full_name",
                            "alternate_name",
                            "latin_name",
                            "shortened_surname",
                            "title_reference",
                            "anonymous_descriptor",
                            "named_entity",
                        ],
                    },
                },
                "required": ["candidate_id", "category", "form"],
            },
        },
        "coreference": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "left_candidate_id": {"type": "string"},
                    "right_candidate_id": {"type": "string"},
                    "decision": {
                        "type": "string",
                        "enum": ["SAME_LOCAL", "DIFFERENT_LOCAL", "INSUFFICIENT"],
                    },
                },
                "required": [
                    "left_candidate_id",
                    "right_candidate_id",
                    "decision",
                ],
            },
        },
    },
    "required": ["selections", "coreference"],
}

EXTRACTION_SYSTEM = """You extract occurrence-level mentions from Traditional Chinese newspaper text. The language is approximately 100 years old, not ancient Chinese. Return only named people, person references, named places, named organizations, named schools, and named publications. Do not extract occupations, generic locations, events, verbs, adjectives, products, or phrases. Every surface must copy an exact source substring. start and end are zero-based Python Unicode code-point offsets, with end exclusive. Record repeated occurrences separately. Resolve a shortened name or title only within this supplied coherent text. Never join a possible name to an adjacent verb or generic location merely because the characters could form a longer span. A one-character surname reference may corefer with a previously introduced full name only when the coherent passage supports it. Patterns such as PERSON，字NAME or PERSON，號NAME identify a style name or sobriquet when the syntax supports that reading. If local identity is unclear, return INSUFFICIENT. Do not infer global aliases."""

RESOLUTION_SYSTEM = """You compare exactly two supplied historical person profiles. Return SAME only when the supplied evidence positively identifies one person. Return DIFFERENT only for explicit separate-person evidence or facts that cannot both be true of one person, such as incompatible life dates or simultaneous mutually exclusive events. A different occupation or place is not a contradiction by itself because a person may change work or travel; without incompatible dates it requires INSUFFICIENT. A shared surname, shared name, occupation, city, or similarity score is not enough for SAME. An article-local shortened reference is not a global alias. Use only supplied evidence IDs. Do not choose a canonical record and do not invent facts."""

CANDIDATE_CLASSIFICATION_SYSTEM = """You classify supplied exact-span candidates from one coherent Traditional Chinese newspaper passage. The language is approximately 100 years old, not ancient Chinese. Select only supplied candidate IDs that are named people, person references, named places, named organizations, named schools, or named publications. Do not select occupations, generic locations, events, verbs, adjectives, products, phrases, or a longer candidate that incorrectly joins a name to a following verb or generic location. Resolve identity only within this passage. A one-character surname reference may corefer with a previously introduced full name only when the coherent passage supports it. Patterns such as PERSON，字NAME or PERSON，號NAME identify a style name or sobriquet when the syntax supports that reading. Never emit an ID that was not supplied."""

EXTRACTION_CASES = [
    {
        "id": "holbein_local_reference",
        "text": "霍爾平（Hans Holbein）為英國著名畫家。英皇時召霍臨宮中作畫。",
        "notes": "Must extract exact 霍, never 霍臨; local 霍 -> 霍爾平.",
    },
    {
        "id": "sun_names",
        "text": "孫文，字逸仙，後以孫中山之名著稱。",
        "notes": "Three person-name occurrences refer to one person in this text.",
    },
    {
        "id": "two_huos",
        "text": "霍女士在上海任教。另一位霍先生在北京任職，二人並無親屬關係。",
        "notes": "Two distinct person references; do not corefer them.",
    },
    {
        "id": "common_phrase_negative",
        "text": "愛寵情人，不僅悅目娛心。",
        "notes": "No named entities.",
    },
    {
        "id": "teacher_school",
        "text": "王女士任教於上海女子學校。",
        "notes": "Person reference 王女士 and school 上海女子學校.",
    },
    {
        "id": "repeated_wang_uncertain",
        "text": "王氏創辦女學。王氏在同頁另一則啟事中捐款，文中未說明是否同一人。",
        "notes": "Two exact 王氏 occurrences; identity is insufficient.",
    },
]

RESOLUTION_CASES = [
    {
        "id": "isolated_surname_not_global_alias",
        "left": {"evidence": [{"id": "L1", "surface": "霍"}]},
        "right": {
            "evidence": [
                {"id": "R1", "surface": "霍爾平"},
                {"id": "R2", "latin_name": "Hans Holbein"},
            ]
        },
        "expected": "INSUFFICIENT",
    },
    {
        "id": "reviewed_sun_aliases",
        "left": {
            "evidence": [
                {"id": "L1", "surface": "孫文"},
                {"id": "L2", "source_claim": "字逸仙"},
            ]
        },
        "right": {
            "evidence": [
                {"id": "R1", "surface": "孫中山"},
                {"id": "R2", "authority_id": "reviewed:sun-yat-sen"},
                {"id": "R3", "authority_alias": "孫文"},
            ]
        },
        "expected": "SAME",
    },
    {
        "id": "explicit_distinct_wangs",
        "left": {
            "evidence": [
                {"id": "L1", "surface": "王氏"},
                {"id": "L2", "source_claim": "與另一王氏並非同一人"},
            ]
        },
        "right": {
            "evidence": [
                {"id": "R1", "surface": "王氏"},
                {"id": "R2", "source_claim": "與前述王氏並非同一人"},
            ]
        },
        "expected": "DIFFERENT",
    },
    {
        "id": "same_name_conflicting_profiles",
        "left": {
            "evidence": [
                {"id": "L1", "surface": "陳淑英"},
                {"id": "L2", "occupation": "教師", "place": "上海"},
            ]
        },
        "right": {
            "evidence": [
                {"id": "R1", "surface": "陳淑英"},
                {"id": "R2", "occupation": "醫師", "place": "廣州"},
            ]
        },
        "expected": "INSUFFICIENT",
    },
]

CANDIDATE_CLASSIFICATION_CASES = [
    {
        "id": "bounded_holbein",
        "text": "霍爾平（Hans Holbein）為英國著名畫家。英皇時召霍臨宮中作畫。",
        "candidates": [
            {"id": "C1", "surface": "霍爾平", "start": 0, "end": 3},
            {"id": "C2", "surface": "Hans Holbein", "start": 4, "end": 16},
            {"id": "C3", "surface": "英國", "start": 18, "end": 20},
            {"id": "C4", "surface": "畫家", "start": 22, "end": 24},
            {"id": "C5", "surface": "英皇", "start": 25, "end": 27},
            {"id": "C6", "surface": "霍", "start": 29, "end": 30},
            {"id": "C7", "surface": "霍臨", "start": 29, "end": 31},
            {"id": "C8", "surface": "宮中", "start": 31, "end": 33},
        ],
        "expected_selected_ids": ["C1", "C2", "C3", "C5", "C6"],
        "expected_coreference": [
            {"left_candidate_id": "C1", "right_candidate_id": "C2", "decision": "SAME_LOCAL"},
            {"left_candidate_id": "C1", "right_candidate_id": "C6", "decision": "SAME_LOCAL"}
        ],
    },
    {
        "id": "bounded_sun_names",
        "text": "孫文，字逸仙，後以孫中山之名著稱。",
        "candidates": [
            {"id": "C1", "surface": "孫文", "start": 0, "end": 2},
            {"id": "C2", "surface": "字逸仙", "start": 3, "end": 6},
            {"id": "C3", "surface": "逸仙", "start": 4, "end": 6},
            {"id": "C4", "surface": "孫中山", "start": 9, "end": 12},
        ],
        "expected_selected_ids": ["C1", "C3", "C4"],
        "expected_coreference": [
            {"left_candidate_id": "C1", "right_candidate_id": "C3", "decision": "SAME_LOCAL"},
            {"left_candidate_id": "C1", "right_candidate_id": "C4", "decision": "SAME_LOCAL"},
        ],
    },
    {
        "id": "bounded_common_phrase_negative",
        "text": "愛寵情人，不僅悅目娛心。",
        "candidates": [
            {"id": "C1", "surface": "愛寵", "start": 0, "end": 2},
            {"id": "C2", "surface": "情人", "start": 2, "end": 4},
            {"id": "C3", "surface": "愛寵情人", "start": 0, "end": 4},
        ],
        "expected_selected_ids": [],
        "expected_coreference": [],
    },
    {
        "id": "bounded_repeated_wang",
        "text": "王氏創辦女學。王氏在同頁另一則啟事中捐款，文中未說明是否同一人。",
        "candidates": [
            {"id": "C1", "surface": "王氏", "start": 0, "end": 2},
            {"id": "C2", "surface": "王氏", "start": 7, "end": 9},
        ],
        "expected_selected_ids": ["C1", "C2"],
        "expected_coreference": [
            {"left_candidate_id": "C1", "right_candidate_id": "C2", "decision": "INSUFFICIENT"}
        ],
    },
]


def chat(base_url: str, model: str, system: str, user: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    grounded_user = {
        **user,
        "output_contract": {
            "instruction": "Return only one JSON value matching this schema exactly.",
            "json_schema": schema,
        },
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(grounded_user, ensure_ascii=False),
            },
        ],
        "format": schema,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "num_predict": 2048,
        },
        "keep_alive": "10m",
    }
    request = Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=600) as response:
        envelope = json.load(response)
    elapsed = time.perf_counter() - started
    content = envelope.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        parse_error = None
    except json.JSONDecodeError as error:
        parsed = None
        parse_error = str(error)
    return {
        "elapsed_seconds": elapsed,
        "raw_content": content,
        "response": parsed,
        "json_parse_error": parse_error,
        "ollama_metrics": {
            key: envelope.get(key)
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
        },
    }


def validate_extraction(text: str, response: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(response, dict):
        return ["top-level structured response is not an object"]
    if set(response) != {"mentions", "local_coreference"}:
        errors.append("top-level object has missing or unexpected keys")
    if not isinstance(response.get("mentions"), list):
        errors.append("mentions is not an array")
        return errors
    for index, mention in enumerate(response.get("mentions", [])):
        start = mention.get("start")
        end = mention.get("end")
        surface = mention.get("surface")
        if not isinstance(start, int) or not isinstance(end, int):
            errors.append(f"mention[{index}] has non-integer offsets")
        elif start < 0 or end > len(text) or start >= end:
            errors.append(f"mention[{index}] has invalid offsets {start}:{end}")
        elif text[start:end] != surface:
            errors.append(
                f"mention[{index}] surface mismatch: {surface!r} != {text[start:end]!r}"
            )
    return errors


def validate_candidate_classification(case: dict[str, Any], response: Any) -> list[str]:
    if not isinstance(response, dict):
        return ["top-level structured response is not an object"]
    errors: list[str] = []
    allowed = {candidate["id"] for candidate in case["candidates"]}
    selections = response.get("selections")
    coreference = response.get("coreference")
    if not isinstance(selections, list) or not isinstance(coreference, list):
        return ["response lacks selections/coreference arrays"]
    selected_ids = [selection.get("candidate_id") for selection in selections]
    unknown = sorted({item for item in selected_ids if item not in allowed})
    if unknown:
        errors.append(f"unknown selected candidate IDs: {unknown}")
    for relation in coreference:
        relation_ids = {
            relation.get("left_candidate_id"),
            relation.get("right_candidate_id"),
        }
        if not relation_ids <= allowed:
            errors.append(f"coreference uses unknown IDs: {sorted(relation_ids - allowed)}")
    if sorted(selected_ids) != sorted(case["expected_selected_ids"]):
        errors.append(
            f"selected IDs {sorted(selected_ids)} != expected {sorted(case['expected_selected_ids'])}"
        )
    observed_relations = {
        (
            relation.get("left_candidate_id"),
            relation.get("right_candidate_id"),
            relation.get("decision"),
        )
        for relation in coreference
    }
    expected_relations = {
        (
            relation["left_candidate_id"],
            relation["right_candidate_id"],
            relation["decision"],
        )
        for relation in case["expected_coreference"]
    }
    observed_same = {
        frozenset((left, right))
        for left, right, decision in observed_relations
        if decision == "SAME_LOCAL"
    }
    expected_same = {
        frozenset((left, right))
        for left, right, decision in expected_relations
        if decision == "SAME_LOCAL"
    }

    def transitive_closure(edges: set[frozenset[str]]) -> set[frozenset[str]]:
        graph: dict[str, set[str]] = {}
        for edge in edges:
            if len(edge) != 2:
                continue
            left, right = tuple(edge)
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
        closure: set[frozenset[str]] = set()
        visited: set[str] = set()
        for node in graph:
            if node in visited:
                continue
            stack = [node]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(graph.get(current, ()))
            visited.update(component)
            ordered = sorted(component)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1 :]:
                    closure.add(frozenset((left, right)))
        return closure

    observed_same_closure = transitive_closure(observed_same)
    expected_same_closure = transitive_closure(expected_same)
    if not expected_same <= observed_same_closure:
        errors.append("required SAME_LOCAL relations are not connected")
    if not observed_same <= expected_same_closure:
        errors.append("unexpected SAME_LOCAL relation crosses expected clusters")

    observed_non_same = {
        relation for relation in observed_relations if relation[2] != "SAME_LOCAL"
    }
    expected_non_same = {
        relation for relation in expected_relations if relation[2] != "SAME_LOCAL"
    }
    if observed_non_same != expected_non_same:
        errors.append("non-SAME coreference decisions do not match expected fixture")
    return errors


def parse_bare_candidate_ids(content: str) -> list[str] | None:
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[len("```json") : -len("```")].strip()
    if stripped == "[]":
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
        return parsed
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("selected_ids"), list)
        and all(isinstance(item, str) for item in parsed["selected_ids"])
    ):
        return parsed["selected_ids"]
    parts = [part.strip() for part in stripped.split(",")]
    if parts and all(part.startswith("C") and part[1:].isdigit() for part in parts):
        return parts
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "conditions": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 8192,
            "thinking": False,
            "structured_output": "native_ollama_json_schema",
        },
        "scope": "technical smoke suite; not an accuracy benchmark",
        "evaluation_controls": {
            "expected_answers_excluded_from_model_input": True,
            "case_notes_excluded_from_model_input": True,
            "fixture_specific_answers_excluded_from_system_prompt": True,
        },
        "extraction_cases": [],
        "candidate_classification_cases": [],
        "resolution_cases": [],
    }

    for case in EXTRACTION_CASES:
        result = chat(
            args.base_url,
            args.model,
            EXTRACTION_SYSTEM,
            {
                "task": "extract_mentions_and_local_coreference",
                "case_id": case["id"],
                "text": case["text"],
            },
            EXTRACTION_SCHEMA,
        )
        result.update(case)
        result["offset_validation_errors"] = validate_extraction(
            case["text"], result["response"]
        )
        artifact["extraction_cases"].append(result)

    for case in CANDIDATE_CLASSIFICATION_CASES:
        result = chat(
            args.base_url,
            args.model,
            CANDIDATE_CLASSIFICATION_SYSTEM,
            {
                "task": "classify_supplied_candidates",
                "case_id": case["id"],
                "text": case["text"],
                "candidates": case["candidates"],
            },
            CANDIDATE_CLASSIFICATION_SCHEMA,
        )
        result.update(case)
        result["validation_errors"] = validate_candidate_classification(
            case, result["response"]
        )
        bare_ids = parse_bare_candidate_ids(result["raw_content"])
        result["bare_selected_ids"] = bare_ids
        result["bare_selection_matches_expected"] = (
            bare_ids is not None
            and sorted(bare_ids) == sorted(case["expected_selected_ids"])
        )
        artifact["candidate_classification_cases"].append(result)

    for case in RESOLUTION_CASES:
        result = chat(
            args.base_url,
            args.model,
            RESOLUTION_SYSTEM,
            {
                "task": "compare_profiles",
                "case_id": case["id"],
                "left": case["left"],
                "right": case["right"],
            },
            RESOLUTION_SCHEMA,
        )
        result.update(case)
        result["matches_expected_decision"] = (
            isinstance(result["response"], dict)
            and result["response"].get("decision") == case["expected"]
        )
        result["bare_decision_matches_expected"] = (
            result["raw_content"].strip() == case["expected"]
        )
        artifact["resolution_cases"].append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
