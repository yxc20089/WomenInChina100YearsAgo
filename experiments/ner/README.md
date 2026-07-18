# Historical-Chinese NER benchmark

The pinned candidate registry is `candidates.json`. It is a shortlist, not a
production selection.

Evaluate each applicable model on two paired inputs: double-corrected text and
the corresponding raw OCR. Split by issue/date rather than random snippets so
near-duplicate newspaper language cannot leak across sets. Report exact and
relaxed span F1 by type, hallucinated-span rate, evidence/offset validity,
throughput, peak memory, and degradation as OCR CER rises.

The production hypothesis is a cascade: gazetteers and rules; a SIKU-BERT-based
project-specific span/token model plus GLiNER-X candidates; then NuExtract3 only
for disagreements, rare types, implicit relations, or difficult page crops.
Every stage must preserve exact source offsets and may abstain. Entity linking
and claim review remain separate gates.
