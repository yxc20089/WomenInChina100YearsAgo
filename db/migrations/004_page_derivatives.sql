BEGIN;

CREATE TABLE archive.page_derivative (
    derivative_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    page_id uuid NOT NULL REFERENCES archive.page(page_id) ON DELETE CASCADE,
    image_uri text NOT NULL,
    image_sha256 text NOT NULL CHECK (image_sha256 ~ '^[0-9a-f]{64}$'),
    width integer NOT NULL CHECK (width > 0),
    height integer NOT NULL CHECK (height > 0),
    dpi integer CHECK (dpi > 0),
    media_type text NOT NULL CHECK (media_type LIKE 'image/%'),
    evidence_tier text NOT NULL CHECK (
        evidence_tier IN (
            'screening_derivative',
            'unreviewed_input',
            'non_gold_lossless_pilot',
            'historian_selected_gold'
        )
    ),
    preference_rank smallint NOT NULL,
    render_manifest_uri text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (page_id, image_sha256),
    CHECK (
        (evidence_tier = 'screening_derivative' AND preference_rank = 10)
        OR (evidence_tier = 'unreviewed_input' AND preference_rank = 20)
        OR (evidence_tier = 'non_gold_lossless_pilot' AND preference_rank = 30)
        OR (evidence_tier = 'historian_selected_gold' AND preference_rank = 40)
    )
);

CREATE INDEX page_derivative_preference_idx
    ON archive.page_derivative(
        page_id,
        preference_rank DESC,
        width DESC,
        height DESC,
        created_at,
        derivative_id
    );

ALTER TABLE archive.page
    ADD COLUMN preferred_derivative_id uuid;

ALTER TABLE archive.page
    ADD CONSTRAINT page_preferred_derivative_fk
    FOREIGN KEY (preferred_derivative_id)
    REFERENCES archive.page_derivative(derivative_id)
    ON DELETE SET NULL;

INSERT INTO archive.page_derivative (
    page_id, image_uri, image_sha256, width, height, dpi, media_type,
    evidence_tier, preference_rank, metadata
)
SELECT page_id, source_image_uri, source_image_sha256, width, height, dpi,
       CASE
           WHEN source_image_uri ~* '\.png$' THEN 'image/png'
           WHEN source_image_uri ~* '\.(tif|tiff)$' THEN 'image/tiff'
           ELSE 'image/jpeg'
       END,
       CASE
           WHEN metadata::text ILIKE '%screening derivative%'
               THEN 'screening_derivative'
           ELSE 'unreviewed_input'
       END,
       CASE
           WHEN metadata::text ILIKE '%screening derivative%' THEN 10
           ELSE 20
       END,
       jsonb_build_object('backfilled_from_legacy_page_columns', true)
FROM archive.page
WHERE source_image_uri IS NOT NULL
  AND source_image_sha256 IS NOT NULL
  AND width IS NOT NULL
  AND height IS NOT NULL
ON CONFLICT (page_id, image_sha256) DO NOTHING;

UPDATE archive.page AS page
SET preferred_derivative_id = (
    SELECT derivative_id
    FROM archive.page_derivative AS derivative
    WHERE derivative.page_id = page.page_id
    ORDER BY preference_rank DESC, width DESC, height DESC,
             created_at, derivative_id
    LIMIT 1
);

COMMIT;
