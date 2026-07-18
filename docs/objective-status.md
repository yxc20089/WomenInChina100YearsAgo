# Objective status

Status at 2026-07-18. “Implemented” means exercised on the bounded page-308
pilot unless the evidence column says otherwise; it does not imply
corpus-scale production readiness or historical validity.

| Objective | Status | Evidence and next gate |
|---|---|---|
| Identify available modalities | Audited | The readable `sb_raw/` prefix contains 395 PDFs, 5 DjVu files and 1 JPEG across 400 volumes/340,511 known pages (~157.4 GiB). No audio, video or born-digital text was found in that authorized prefix. |
| Select OCR/document models | Provisional selection implemented | Source-resolution PP-OCRv6 produced a coordinate-preserving 1,099-region pilot. PP-StructureV3 and PaddleOCR-VL-1.6 remain benchmark arms; historian-selected lossless gold is the gate. |
| Select NER/extraction models | Architecture and benchmark contract implemented | Rules + a MacBERT/GujiRoBERTa/SIKU W2NER tournament, with gated Otter/GLiNER-X recall challengers and NuExtract3 routed difficult cases. Current zero-shot outputs disagree severely and are candidates only. A double-reviewed raw-OCR/corrected-text gold set is the gate. |
| Durable multimodal ingestion | Bounded vertical slice implemented | PostgreSQL DAG, leases, retries, provenance checks, resumable stage workers, failure propagation, cancellation and replay are live-tested. Production fleet controls, capacity planning and corpus-scale execution remain. |
| Relational evidence store | Implemented | PostgreSQL is authoritative for immutable sources/derivatives, OCR/NER runs, exact evidence, review decisions, claims and job control. |
| Search/retrieval store | Implemented baseline | Active-selection OpenSearch CJK/BGE-M3/RRF projection contains 1,099 regions and returns exact derivative/region pointers. Historian questions, reranking and larger-corpus evaluation remain. |
| Knowledge graph | Safe projection implemented | Neo4j is a rebuildable reviewed-only projection and correctly remains empty. Genuine reviewed entities and cited claims are required before graph interpretation. |
| GraphRAG comparison | Adapters implemented, experiment pending | One citation-preserving shared export feeds LightRAG and Microsoft GraphRAG. Article segmentation, recorded LLM configuration and historian retrieval judgments are required before indexing/model comparison. |
| Researcher UI | Implemented bounded workflow | Natural-language lexical/dense/hybrid search, scan links, mention/entity/claim review, reviewed insights and evidence-linked machine exploration are available locally. Authentication is not implemented. |
| LLM research assistant | Bounded multi-turn contract implemented | Each question performs fresh retrieval; at most 12 prior turns are untrusted continuity context, and region citations are validated against current evidence. Browser history is intentionally not server-persisted. No live model is configured, so model-quality evaluation remains. |
| Scene/lived-experience reconstruction | Safety gate implemented; content intentionally blocked | Scene generation hard-abstains without reviewed claims. Historical review, uncertainty policy and model evaluation must precede any reconstructed narrative. |
| New historical insights | Not yet established | The exploratory panel surfaces scan-linked reading priorities and model disagreement, while the reviewed insight plane remains empty. Only historian-reviewed evidence can support a historical finding. |

The next scientific bottleneck is not another unscored model run. It is a
representative, double-reviewed OCR/layout/NER benchmark and a historian-authored
retrieval set. The next product bottleneck is configuring and evaluating the
citation-validating assistant over that evidence plane, then adding controlled
conversation persistence only if researcher studies justify it.
