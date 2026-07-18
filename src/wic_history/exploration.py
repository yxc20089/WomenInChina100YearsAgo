"""Evidence-linked exploratory leads over active OCR and candidate NER outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import Field

from .evidence import SourcePointer, StrictModel


class ExplorationCounts(StrictModel):
    active_pages: int = 0
    active_regions: int = 0
    text_characters: int = 0
    mean_ocr_confidence: float | None = None
    low_confidence_regions: int = 0
    active_ocr_runs: int = 0
    candidate_ner_runs: int = 0
    candidate_mentions: int = 0


class ThemeEvidence(StrictModel):
    text: str
    normalized_text: str | None = None
    confidence: float | None = None
    source: SourcePointer


class ThemeLead(StrictModel):
    theme_id: str
    title: str
    research_prompt: str
    matched_regions: int = 0
    matched_pages: int = 0
    year_start: int | None = None
    year_end: int | None = None
    examples: list[ThemeEvidence] = Field(default_factory=list)
    epistemic_label: str = "machine_observation_research_lead"


class NERRunSignal(StrictModel):
    run_id: UUID
    model_name: str
    model_revision: str
    dataset_id: str | None = None
    split_id: str | None = None
    max_regions: int | None = None
    input_regions: int | None = None
    candidate_mentions: int = 0
    cited_regions: int = 0
    mean_confidence: float | None = None


class NERAgreementSignal(StrictModel):
    left_run_id: UUID
    right_run_id: UUID
    left_model: str
    right_model: str
    left_candidates: int
    right_candidates: int
    exact_agreements: int
    candidate_jaccard: float
    epistemic_label: str = "model_disagreement_review_priority"


class ExplorationReport(StrictModel):
    generated_at: datetime
    counts: ExplorationCounts
    themes: list[ThemeLead]
    ner_runs: list[NERRunSignal]
    ner_agreements: list[NERAgreementSignal]
    warnings: list[str] = Field(default_factory=list)


THEMES: tuple[tuple[str, str, str, str], ...] = (
    (
        "elite_women_publics",
        "Elite women and public audiences",
        r"(士女|淑女|名媛|女界)",
        "Inspect how labels such as 士女, 淑女, 名媛, and 女界 frame women as audiences or publics.",
    ),
    (
        "women_education",
        "Women and education",
        r"(女學|女校|女生|女學生|女塾|女子學校)",
        "Trace schools, students, curricula, locations, and the language used around women's education.",
    ),
    (
        "women_work",
        "Women and work",
        r"(女工|工女|女傭|女職|職業婦女|女店員)",
        "Compare occupations, workplaces, wages, labor disputes, and the visibility of women's work.",
    ),
    (
        "performance_media",
        "Women in performance and media",
        r"(女.{0,12}(演|伶|影|戲)|(演|伶|影|戲).{0,12}女|女明星|女演員|女伶)",
        "Investigate women performers, cinema/theatre audiences, publicity, and cultural mediation.",
    ),
    (
        "marriage_family",
        "Marriage, kinship, and family",
        r"(婚姻|結婚|離婚|妻|寡婦|母親|女兒|媳婦|妾)",
        "Compare legal, social, and commercial language around marriage, kinship, and household roles.",
    ),
    (
        "women_consumption",
        "Women, advertising, and consumption",
        r"((女士|婦女|淑女|名媛|女界).{0,20}(廣告|出售|價|貨|用品|服|藥|粉|香)|(廣告|出售|價|貨|用品|服|藥|粉|香).{0,20}(女士|婦女|淑女|名媛|女界))",
        "Examine how products and advertisements construct women as consumers, authorities, or symbols.",
    ),
)


ACTIVE_OCR_CTE = """
active_ocr AS (
    SELECT region.region_id, region.run_id, region.page_id,
           region.raw_text, region.normalized_text, region.confidence,
           region.polygon, page.page_number, volume.volume_number,
           volume.publication_year, source.source_uri,
           source.sha256 AS source_sha256,
           derivative.derivative_id,
           derivative.image_sha256, derivative.evidence_tier
    FROM evidence.ocr_region region
    JOIN archive.page page USING (page_id)
    JOIN archive.volume volume USING (volume_id)
    JOIN archive.source_object source USING (source_object_id)
    JOIN evidence.page_ocr_selection selection
      ON selection.page_id = region.page_id
     AND selection.run_id = region.run_id
     AND selection.superseded_at IS NULL
    JOIN archive.page_derivative derivative
      ON derivative.derivative_id = selection.derivative_id
)
"""


COUNTS_SQL = f"""
WITH {ACTIVE_OCR_CTE}
SELECT count(DISTINCT page_id) AS active_pages,
       count(*) AS active_regions,
       COALESCE(sum(length(COALESCE(normalized_text, raw_text))), 0)
           AS text_characters,
       avg(confidence) AS mean_ocr_confidence,
       count(*) FILTER (WHERE confidence < 0.5) AS low_confidence_regions,
       count(DISTINCT run_id) AS active_ocr_runs,
       (SELECT count(DISTINCT input.run_id)
        FROM evidence.ner_run_input input
        WHERE input.source_ocr_run_id IN (SELECT DISTINCT run_id FROM active_ocr))
           AS candidate_ner_runs,
       (SELECT count(*)
        FROM evidence.entity_mention mention
        JOIN evidence.ocr_region region USING (region_id)
        WHERE region.run_id IN (SELECT DISTINCT run_id FROM active_ocr))
           AS candidate_mentions
FROM active_ocr
"""


THEME_SQL = f"""
WITH themes AS (
    SELECT * FROM unnest(%s::text[], %s::text[])
      AS theme(theme_id, pattern)
),
{ACTIVE_OCR_CTE},
matches AS (
    SELECT theme.theme_id, active_ocr.*,
           row_number() OVER (
               PARTITION BY theme.theme_id
               ORDER BY confidence DESC NULLS LAST, publication_year,
                        volume_number, page_number, region_id
           ) AS example_rank
    FROM themes theme
    JOIN active_ocr
      ON COALESCE(active_ocr.normalized_text, active_ocr.raw_text) ~ theme.pattern
),
aggregate AS (
    SELECT theme_id, count(*) AS matched_regions,
           count(DISTINCT page_id) AS matched_pages,
           min(publication_year) AS year_start,
           max(publication_year) AS year_end
    FROM matches GROUP BY theme_id
)
SELECT theme.theme_id,
       COALESCE(aggregate.matched_regions, 0) AS matched_regions,
       COALESCE(aggregate.matched_pages, 0) AS matched_pages,
       aggregate.year_start, aggregate.year_end,
       matches.region_id, matches.raw_text, matches.normalized_text,
       matches.confidence, matches.polygon, matches.source_uri,
       matches.source_sha256, matches.derivative_id,
       matches.image_sha256, matches.evidence_tier,
       matches.volume_number, matches.publication_year, matches.page_number
FROM themes theme
LEFT JOIN aggregate USING (theme_id)
LEFT JOIN matches
  ON matches.theme_id = theme.theme_id AND matches.example_rank <= %s
ORDER BY theme.theme_id, matches.example_rank
"""


NER_RUNS_SQL = f"""
WITH {ACTIVE_OCR_CTE},
active_runs AS (
    SELECT DISTINCT input.run_id, input.source_ocr_run_id, input.dataset_id,
           input.split_id
    FROM evidence.ner_run_input input
    WHERE input.source_ocr_run_id IN (SELECT DISTINCT run_id FROM active_ocr)
),
active_mentions AS (
    SELECT mention.*
    FROM evidence.entity_mention mention
    JOIN active_runs USING (run_id)
    JOIN active_ocr USING (region_id)
)
SELECT run.run_id, run.model_name, run.model_revision,
       active_runs.dataset_id, active_runs.split_id,
       (run.configuration->>'max_regions')::integer AS max_regions,
       (run.configuration->>'input_region_count')::integer AS input_regions,
       count(active_mentions.mention_id) AS candidate_mentions,
       count(DISTINCT active_mentions.region_id) AS cited_regions,
       avg(active_mentions.confidence) AS mean_confidence
FROM active_runs
JOIN evidence.processing_run run USING (run_id)
LEFT JOIN active_mentions USING (run_id)
GROUP BY run.run_id, run.model_name, run.model_revision,
         active_runs.dataset_id, active_runs.split_id,
         run.configuration
ORDER BY candidate_mentions DESC, run.model_name, run.run_id
"""


NER_AGREEMENT_SQL = f"""
WITH {ACTIVE_OCR_CTE},
active_runs AS (
    SELECT DISTINCT input.run_id, run.model_name
    FROM evidence.ner_run_input input
    JOIN evidence.processing_run run USING (run_id)
    WHERE input.source_ocr_run_id IN (SELECT DISTINCT run_id FROM active_ocr)
),
active_mentions AS (
    SELECT mention.*
    FROM evidence.entity_mention mention
    JOIN active_runs USING (run_id)
    JOIN active_ocr USING (region_id)
),
counts AS (
    SELECT active_runs.run_id, active_runs.model_name,
           count(active_mentions.mention_id) AS candidate_count
    FROM active_runs
    LEFT JOIN active_mentions USING (run_id)
    GROUP BY active_runs.run_id, active_runs.model_name
),
pairs AS (
    SELECT left_run.run_id AS left_run_id,
           right_run.run_id AS right_run_id,
           left_run.model_name AS left_model,
           right_run.model_name AS right_model,
           left_run.candidate_count AS left_candidates,
           right_run.candidate_count AS right_candidates
    FROM counts left_run
    JOIN counts right_run ON left_run.run_id < right_run.run_id
),
agreements AS (
    SELECT left_mention.run_id AS left_run_id,
           right_mention.run_id AS right_run_id,
           count(DISTINCT (
               left_mention.region_id, left_mention.entity_type,
               left_mention.text_start, left_mention.text_end
           )) AS exact_agreements
    FROM active_mentions left_mention
    JOIN active_mentions right_mention
      ON left_mention.run_id < right_mention.run_id
     AND left_mention.region_id = right_mention.region_id
     AND left_mention.entity_type = right_mention.entity_type
     AND left_mention.text_start = right_mention.text_start
     AND left_mention.text_end = right_mention.text_end
    GROUP BY left_mention.run_id, right_mention.run_id
)
SELECT pairs.*, COALESCE(agreements.exact_agreements, 0) AS exact_agreements,
       CASE WHEN pairs.left_candidates + pairs.right_candidates
                      - COALESCE(agreements.exact_agreements, 0) = 0
            THEN 1.0
            ELSE COALESCE(agreements.exact_agreements, 0)::double precision
                 / (pairs.left_candidates + pairs.right_candidates
                    - COALESCE(agreements.exact_agreements, 0))
       END AS candidate_jaccard
FROM pairs
LEFT JOIN agreements USING (left_run_id, right_run_id)
ORDER BY candidate_jaccard ASC, left_model, right_model
"""


def theme_rows_to_leads(rows: list[dict[str, Any]]) -> list[ThemeLead]:
    definitions = {
        theme_id: (title, prompt) for theme_id, title, _, prompt in THEMES
    }
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        theme_id = row["theme_id"]
        group = grouped.setdefault(
            theme_id,
            {
                "matched_regions": row["matched_regions"],
                "matched_pages": row["matched_pages"],
                "year_start": row["year_start"],
                "year_end": row["year_end"],
                "examples": [],
            },
        )
        if row.get("region_id") is None:
            continue
        group["examples"].append(
            ThemeEvidence(
                text=row["raw_text"],
                normalized_text=row["normalized_text"],
                confidence=row["confidence"],
                source=SourcePointer(
                    source_uri=row["source_uri"],
                    source_sha256=row["source_sha256"],
                    derivative_id=row["derivative_id"],
                    image_sha256=row["image_sha256"],
                    evidence_tier=row["evidence_tier"],
                    volume_number=row["volume_number"],
                    publication_year=row["publication_year"],
                    page_number=row["page_number"],
                    region_id=row["region_id"],
                    polygon=row["polygon"],
                ),
            )
        )
    return [
        ThemeLead(
            theme_id=theme_id,
            title=definitions[theme_id][0],
            research_prompt=definitions[theme_id][1],
            **grouped.get(
                theme_id,
                {
                    "matched_regions": 0,
                    "matched_pages": 0,
                    "year_start": None,
                    "year_end": None,
                    "examples": [],
                },
            ),
        )
        for theme_id, _, _, _ in THEMES
    ]


def build_exploration_report(
    database_url: str, *, examples_per_theme: int = 3
) -> ExplorationReport:
    if not 1 <= examples_per_theme <= 10:
        raise ValueError("examples_per_theme must be between 1 and 10")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - minimal installations
        raise RuntimeError("Install the data extra: uv sync --extra data") from exc
    theme_ids = [item[0] for item in THEMES]
    patterns = [item[2] for item in THEMES]
    with psycopg.connect(database_url, row_factory=dict_row) as connection:
        counts = ExplorationCounts.model_validate(
            connection.execute(COUNTS_SQL).fetchone()
        )
        theme_rows = connection.execute(
            THEME_SQL, (theme_ids, patterns, examples_per_theme)
        ).fetchall()
        ner_rows = connection.execute(NER_RUNS_SQL).fetchall()
        agreement_rows = connection.execute(NER_AGREEMENT_SQL).fetchall()
    themes = theme_rows_to_leads(theme_rows)
    warnings = [
        "Exploratory themes are regex matches over machine OCR, not reviewed historical findings.",
        "NER counts and agreement scores are candidate-generation diagnostics, not accuracy measurements.",
        "Open each cited scan region and complete historian review before interpretation or publication.",
    ]
    if counts.active_pages < 10:
        warnings.append(
            f"The active searchable corpus currently contains only {counts.active_pages} page(s); "
            "absence and frequency comparisons are not meaningful yet."
        )
    if not any(theme.matched_regions for theme in themes):
        warnings.append("No configured women-centered exploratory theme matched active OCR.")
    return ExplorationReport(
        generated_at=datetime.now(timezone.utc),
        counts=counts,
        themes=themes,
        ner_runs=[NERRunSignal.model_validate(row) for row in ner_rows],
        ner_agreements=[
            NERAgreementSignal.model_validate(row) for row in agreement_rows
        ],
        warnings=warnings,
    )
