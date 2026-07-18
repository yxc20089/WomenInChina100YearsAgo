# Objective status

Status at 2026-07-18. “Implemented” means exercised on a bounded three-page,
three-volume 1924–1926 slice unless the evidence column says otherwise; it does not imply
corpus-scale production readiness or historical validity.

| Objective | Status | Evidence and next gate |
|---|---|---|
| Identify available modalities | Audited | The readable `sb_raw/` prefix contains 395 PDFs, 5 DjVu files and 1 JPEG across 400 volumes/340,511 known pages (~157.4 GiB). No audio, video or born-digital text was found in that authorized prefix. |
| Select OCR/document models | Provisional selection implemented | Source-resolution PP-OCRv6 produced a coordinate-preserving 1,099-region pilot. PP-StructureV3 and PaddleOCR-VL-1.6 remain benchmark arms; historian-selected lossless gold is the gate. |
| Select NER/extraction models | Architecture and benchmark contract implemented | Rules + a MacBERT/GujiRoBERTa/SIKU W2NER tournament, with gated Otter/GLiNER-X recall challengers and NuExtract3 routed difficult cases. Current zero-shot outputs disagree severely and are candidates only. A double-reviewed raw-OCR/corrected-text gold set is the gate. |
| Prepare NER gold work | Candidate workflow and segmentation gate implemented; human gold absent | Active OCR can be sampled into a content-addressed administrative packet, blinded reviewer view and strict two-review/adjudication template. Schema 1.1 bounds context by approved coherent units when present. The current 150-unit packet remains ineligible: 0/150 approved-unit bounds, one decade, no issue IDs and fewer than 500 units. |
| Durable multimodal ingestion | Bounded three-volume vertical slice implemented | PostgreSQL DAG, leases, retries, provenance checks, resumable stage workers, failure propagation, cancellation and replay are live-tested. Fresh four-stage DAGs completed for 1924 and 1926 in addition to 1925. Production fleet controls, capacity planning and corpus-scale execution remain. |
| Relational evidence store | Implemented | PostgreSQL is authoritative for immutable sources/derivatives, OCR/NER runs, exact evidence, review decisions, claims and job control. Machine segmentation proposals are immutable and separate from copied, revisioned historian-approved coherent units and issue assignments. |
| Search/retrieval store | Implemented baseline | Active-selection OpenSearch CJK/BGE-M3/RRF contains 2,498 regions across 1924–1926 and returns exact derivative/region pointers. Three year-specific live queries passed; historian questions, reranking and larger-corpus evaluation remain. |
| Knowledge graph | Safe projection implemented | Neo4j is a rebuildable reviewed-only projection and correctly remains empty. Genuine reviewed entities and cited claims are required before graph interpretation. |
| GraphRAG comparison | Adapters, page smoke export and reviewed-unit gate implemented; experiment pending | The page export contains 2,471 cited text regions and 27 accounted empty regions. A separate reviewed-coherent-unit exporter hard-fails today because 106 machine candidate windows have zero reviews/approved revisions. Historian segmentation, recorded LLM configuration and retrieval judgments remain required. |
| Researcher UI | Implemented bounded workflow | Natural-language lexical/dense/hybrid search, scan links, mention/entity/claim review, reviewed insights and evidence-linked machine exploration are available locally. Authentication is not implemented. |
| LLM research assistant | Bounded multi-turn contract implemented | Each question performs fresh retrieval; at most 12 prior turns are untrusted continuity context, and region citations are validated against current evidence. Browser history is intentionally not server-persisted. No live model is configured, so model-quality evaluation remains. |
| Scene/lived-experience reconstruction | Safety gate implemented; content intentionally blocked | Scene generation hard-abstains without reviewed claims. Historical review, uncertainty policy and model evaluation must precede any reconstructed narrative. |
| New historical insights | Not yet established | The exploratory panel surfaces scan-linked reading priorities and model disagreement, while the reviewed insight plane remains empty. Only historian-reviewed evidence can support a historical finding. |

The next scientific bottleneck is not another unscored model run. It is a
representative, issue-split, double-reviewed OCR/layout/NER benchmark and a historian-authored
retrieval set. The next product bottleneck is configuring and evaluating the
citation-validating assistant over that evidence plane, then adding controlled
conversation persistence only if researcher studies justify it.
