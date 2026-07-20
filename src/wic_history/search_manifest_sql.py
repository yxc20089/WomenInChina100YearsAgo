from typing import Final


ACTIVE_ARTICLES_SQL: Final = """
SELECT revision.revision_id, revision.unit_id, revision.title
FROM evidence.coherent_unit_revision revision
JOIN evidence.page_article_segmentation_selection approval
  ON approval.selection_id = revision.approval_selection_id
 AND approval.superseded_at IS NULL
WHERE revision.superseded_at IS NULL AND revision.unit_kind = 'article'
ORDER BY revision.revision_id
"""
EMBEDDING_SQL: Final = """
SELECT embedding.embedding::text AS vector
FROM retrieval.embedding embedding
JOIN evidence.processing_run run USING (run_id)
WHERE embedding.target_kind = 'coherent_unit_revision'
  AND embedding.target_id = %s
  AND embedding.model_name = %s
  AND embedding.model_revision = %s
  AND embedding.input_sha256 = %s
  AND embedding.content_sha256 = %s
  AND embedding.configuration_sha256 = %s
  AND run.status = 'completed'
"""
SOURCES_SQL: Final = """
SELECT region.region_id, source.source_uri, source.sha256 AS source_sha256,
       page.page_id, derivative.derivative_id, derivative.image_uri,
       derivative.image_sha256, derivative.evidence_tier,
       volume.volume_number, volume.publication_year, page.page_number,
       region.polygon, derivative.metadata->'warnings' AS warnings
FROM evidence.ocr_region region
JOIN archive.page page USING (page_id)
JOIN archive.volume volume USING (volume_id)
JOIN archive.source_object source USING (source_object_id)
JOIN evidence.ocr_run_input input
  ON input.run_id = region.run_id AND input.page_id = region.page_id
JOIN archive.page_derivative derivative
  ON derivative.derivative_id = input.derivative_id
 AND derivative.page_id = input.page_id
WHERE region.region_id = ANY(%s)
ORDER BY region.region_id
"""
