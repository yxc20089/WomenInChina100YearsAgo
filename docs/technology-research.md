# Technology research: Shen Bao women's-history pilot

> Superseded by the more complete [technical design](technical-design.md), researched through 2026-07-18. This file is retained as the preliminary assessment.

Date: 2026-07-18

## 1. Observed data modalities

The accessible scope is `s3://ccaa-us-east-1-504133794192/sb_raw/`.

| Container | Count | Total size | Observed character |
|---|---:|---:|---|
| PDF | 395 | 167,284,981,412 bytes | Large, image-heavy scanned volumes |
| DjVu | 5 | 1,772,878,220 bytes | Scanned-document containers |
| JPEG | 1 | 708,760 bytes | Year-to-volume index |

Total: 401 objects and 169,058,568,392 bytes (about 157.4 GiB). The index says the 400 numbered volumes cover 1872–1949. PDF sizes range from 160 MB to 642 MB, with a median of about 423 MB.

This is **visual document data**, not clean text. Binary probes of six PDFs found image objects and no font or `ToUnicode` markers in their first MiB. That is evidence of image-based pages, but a full-page extraction test is still required before concluding that every volume lacks a hidden text layer. The semantic modalities to preserve are:

- page image;
- detected regions and reading order;
- OCR text and token/line coordinates;
- newspaper structure (date, page, column, headline, article, advertisement, caption);
- typography and visual evidence, including photographs, illustrations, and advertisements;
- subsequently derived entities, relations, events, claims, and embeddings.

There is no audio or video in the accessible prefix.

## 2. Architectural principle

Maintain two separate graphs:

1. **Evidence graph**: durable, versioned scholarly records. Every assertion points to a source page/region and records extraction method, confidence, review state, and temporal validity.
2. **Retrieval graph**: machine-generated entities, relations, chunks, embeddings, communities, and summaries. It may be regenerated when OCR, prompts, or models change.

Never promote a retrieval edge directly into the evidence graph. Promotion requires validation rules or human review. This separation is more important than the choice of graph database.

## 3. Graph database options

| Option | Strengths for this project | Weaknesses | Recommendation |
|---|---|---|---|
| Neo4j | Mature property graph, Cypher, strong visualization/tooling, supported by LightRAG | Operational component and licensing/edition decisions; provenance must be modeled explicitly | Best pilot graph and researcher exploration layer |
| Amazon Neptune Database | Managed in the existing AWS account; supports RDF/SPARQL and property-graph APIs; backups and scaling are managed | Higher operating cost and AWS coupling; development loop is slower | Re-evaluate for production after pilot and security review |
| PostgreSQL + relational claim model + pgvector | Excellent constraints, migrations, provenance, full auditability, and one operational store | Multi-hop exploration is less ergonomic than a graph-native database | Recommended system of record, even if Neo4j is added |
| RDF store (Apache Jena/Fuseki or managed equivalent) | Standards-based interoperability; natural fit for PROV-O, controlled vocabularies, and linked historical data | Higher ontology/SPARQL learning cost; researcher-facing graph exploration needs more work | Good export/interchange format; consider if linked-open-data publication is a core goal |
| LightRAG-supported all-in-one stores | PostgreSQL, MongoDB, or OpenSearch can supply its four storage roles; Neo4j/Memgraph can supply graph storage | LightRAG's generated graph is optimized for retrieval, not archival truth | Use only for the disposable retrieval graph |

### Initial database decision

Use PostgreSQL as the authoritative store and Neo4j Community for graph exploration in the pilot. Treat Neo4j as a derived projection that can be rebuilt. Export a small RDF/PROV-O sample to test future interoperability. Do not provision Neptune until query load, collaboration requirements, and AWS budget justify it.

Core records should include `SourceObject`, `Volume`, `Issue`, `Page`, `Region`, `TextSpan`, `EntityMention`, `Entity`, `Claim`, `Event`, `Place`, `Organization`, `Person`, `ModelRun`, and `ReviewDecision`. A claim must retain exact page coordinates and OCR/model versions.

## 4. OCR and layout analysis

### Recommended candidates

| Candidate | Role | Assessment |
|---|---|---|
| PaddleOCR PP-StructureV3 + PP-OCRv6 | Baseline layout, coordinates, detection, and Chinese recognition | Primary baseline. Fine-grained coordinates and structured JSON/Markdown are useful for provenance. Current general models still require testing on vertical, dense, degraded historical newspaper print. |
| PaddleOCR-VL 1.6 | Structure-aware document VLM challenger | Test on difficult pages, rare characters, seals, tables, and complex layouts; potentially slower and less deterministic than the classic pipeline. |
| MinerU | PDF parsing/layout challenger and possible LightRAG bridge | Useful comparative parser, but do not let Markdown become the canonical representation because page geometry matters. |
| Docling | Additional document conversion baseline | Valuable for pipeline comparison; likely not the strongest Chinese historical-newspaper recognizer by itself. |
| Cloud OCR (AWS Textract or comparable) | Managed benchmark | Benchmark cost and quality on the same pages, but do not assume strong traditional-Chinese historical accuracy. |

The canonical OCR result should be ALTO XML, PAGE XML, or an equivalent JSON schema retaining polygons, reading order, confidence, page identity, and model version. Markdown is a derivative for retrieval.

### Required OCR benchmark

Create a stratified gold set of 100–200 pages across 1870s, 1900s, 1920s, and 1940s, sampling front pages, inner pages, advertisements, photographs/captions, degradation levels, and both horizontal and vertical layouts. Human-transcribe selected regions. Measure:

- character error rate (CER), including traditional-character normalization policy;
- layout detection F1/IoU by region type;
- reading-order accuracy;
- headline/article segmentation;
- date and page-number accuracy;
- downstream entity recall, because a small CER difference may cause a large name-recall difference;
- throughput, GPU memory, and cost per 1,000 pages.

Do not select OCR by modern-document leaderboards alone.

## 5. NER, relation extraction, and entity linking

NER and entity linking must be separate stages. Recognizing `宋庆龄` as a person mention is different from linking it to the correct historical person; OCR variants, courtesy names, married names, transliterations, and traditional/simplified forms complicate both.

### Candidate stack

1. **Rules and gazetteers** for dates, newspaper issue structure, addresses, schools, organizations, titles, and historical place names.
2. **Supervised fixed-ontology tournament** using identical W2NER heads and training budgets on MacBERT, GujiRoBERTa-jian-fan (license-gated), and SIKU-BERT. MacBERT is the first arm because a directly relevant retyped *Shen Bao* study reports 58.26 F1 versus SIKU 56.83; neither score demonstrates OCR robustness.
3. **Open-type recall challengers**: Otter CE mmBERT and GLiNER-X, with the current GLiNER-multi run retained as a baseline. Add one to the production union only if paired gold evaluation shows meaningful raw-OCR recall gain at bounded precision loss.
4. **Schema-constrained multimodal extraction**: NuExtract3 on routed difficult cases and Qwen3.6-27B only as a high-compute ceiling. Require verbatim surfaces/exact offsets; general multimodal models are not assumed robust on historical Chinese scans.
5. **Entity linking** using aliases/gazetteers, temporal and geographic compatibility, graph context, and a human-review queue.

Start with entity types that support real research questions: `PERSON`, `ALIAS`, `PLACE`, `ADDRESS`, `ORGANIZATION`, `SCHOOL`, `OCCUPATION`, `ROLE_TITLE`, `PUBLICATION`, `EVENT`, `DATE`, `KINSHIP_TERM`, `PRODUCT`, and `ADVERTISEMENT`. Add relation/event schemas only after historians review examples.

The evaluation set should contain at least 500 carefully reviewed snippets and report strict span F1, relaxed span F1, type F1, entity-linking accuracy, calibration, and recall by entity type. Report performance separately by OCR quality and decade. Compare corrected text, raw OCR and observed-confusion augmentation; select with paired issue-cluster bootstrap intervals rather than a generic leaderboard or unsupported absolute initial F1 threshold.

## 6. GraphRAG framework assessment

| System | What it is | Fit |
|---|---|---|
| Microsoft GraphRAG | LLM extraction, community detection/summaries, and local/global/DRIFT-style retrieval over unstructured text | Strong research baseline for corpus-level thematic questions, but indexing is expensive and its own repository calls the implementation a demonstration rather than an officially supported product. Not the evidence database. |
| LazyGraphRAG | Microsoft Research strategy that defers expensive LLM work and combines best-first and breadth-first retrieval | Most promising Microsoft variant for a large OCR corpus. Published results report vector-RAG-like indexing cost and 0.1% of full GraphRAG indexing cost, but we must reproduce quality on Chinese OCR. |
| LightRAG | Incremental entity/relation graph plus vector retrieval, multiple production storage backends, citations, reranking, and recent multimodal parsing integration | Best framework for an early engineering prototype and incremental updates. Its generated descriptions/edges remain retrieval artifacts. Benchmark extraction quality and citation fidelity. |
| MiroFish | AGPL-licensed multi-agent simulation/prediction application that builds a world from seed material | Not an OCR, NER, graph database, or core RAG solution. Potentially interesting much later for clearly labeled counterfactual or experiential simulations; unsuitable for factual historical reconstruction without major safeguards. |
| Vector RAG baseline | Hybrid lexical/vector search over page/article chunks with reranking | Mandatory control. It may answer local evidence questions better and more cheaply than a graph pipeline. |

The 2026 RAG landscape article is useful for generating candidates, but framework claims should be verified against primary papers, repositories, documentation, and our own historical-Chinese evaluation.

## 7. Recommended pilot stack

1. Stream selected volumes from S3; do not download the entire 157.4 GiB archive.
2. Extract pages with stable identifiers and checksums; retain the original S3 URI.
3. Benchmark PaddleOCR PP-StructureV3/PP-OCRv6, PaddleOCR-VL, and MinerU on the gold pages.
4. Store page geometry and OCR versions in PostgreSQL/object storage.
5. Build hybrid retrieval with Chinese lexical search, multilingual embeddings, and reranking.
6. Compare GLiNER v2.5, a multilingual GLiNER model, rules, and one schema-constrained LLM on the same gold snippets.
7. Build the reviewed evidence graph in PostgreSQL and project approved entities/claims into Neo4j.
8. Run an answer-quality experiment comparing vector RAG, LightRAG, Microsoft GraphRAG, and LazyGraphRAG on 30–50 historian-authored questions.
9. Require every answer to cite page-level evidence; score citation correctness separately from prose quality.

### Suggested first corpus

Use 1924–1926 (volumes 199–230) only after a smaller technical sample. Begin with 3 volumes—one each from 1924, 1925, and 1926—and 100–200 benchmark pages. Scale only after measuring pages per volume and compute/cost.

## 8. Decision gates

- OCR gate: choose the system with the best downstream entity recall at acceptable throughput, not simply the best visual demo.
- NER gate: no model enters batch processing without a reviewed Chinese historical-news benchmark.
- graph gate: retain Neo4j only if graph queries produce researcher value beyond PostgreSQL plus hybrid search.
- GraphRAG gate: adopt only if it improves grounded answers over vector RAG enough to justify indexing and operational cost.
- simulation gate: no generated scene or agent simulation may be presented without separating sourced facts, plausible inference, and speculation.

## Sources

- [Microsoft GraphRAG repository](https://github.com/microsoft/graphrag)
- [Microsoft GraphRAG paper](https://arxiv.org/abs/2404.16130)
- [Microsoft Research: LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [LightRAG repository](https://github.com/HKUDS/LightRAG)
- [LightRAG paper](https://arxiv.org/abs/2410.05779)
- [MiroFish repository](https://github.com/666ghj/MiroFish)
- [PaddleOCR repository](https://github.com/PaddlePaddle/PaddleOCR)
- [GLiNER repository and paper links](https://github.com/urchade/GLiNER)
- [GLiNER large v2.5 model card](https://huggingface.co/gliner-community/gliner_large-v2.5)
- [RAG state-of-the-art 2026 landscape supplied for review](https://techwithcolonel.com/artifact/rag-state-of-the-art-2026.html)
