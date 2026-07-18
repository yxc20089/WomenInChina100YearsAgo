# Grounded model handoff operations

The research API can call a local or explicitly approved remote
OpenAI-compatible `/chat/completions` endpoint. This is a model handoff, not a
historical-fact promotion path. Retrieval remains fresh for every question,
and reconstructed scenes remain blocked until cited, reviewed claims exist.

## Configuration

All three identity fields are required together:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL=local-model-name
export LLM_MODEL_REVISION=immutable-weight-or-deployment-id
export LLM_MAX_OUTPUT_TOKENS=2048
export LLM_SEED=17
export LLM_INPUT_COST_PER_MILLION_TOKENS_USD=0  # set only from the provider contract
export LLM_OUTPUT_COST_PER_MILLION_TOKENS_USD=0 # zero is appropriate only for a free local model
```

`LLM_MODEL_REVISION` may be an immutable weights hash, release, or deployment
identifier. Moving identifiers such as `main`, `master`, `latest`, `nightly`
and `dev` are rejected. `LLM_TIMEOUT_SECONDS` defaults to 120 and must be at
most 300 seconds.

For a remote endpoint, HTTPS and explicit data-egress consent are mandatory:

```bash
export LLM_BASE_URL=https://approved-model-service.example/v1
export LLM_ALLOW_REMOTE=true
export LLM_API_KEY=secret-from-an-approved-secret-store
```

The request contains the research query, current retrieved OCR, any cited
reviewed claims, and bounded chat history. Do not enable remote egress until
the archive/data-use agreement and provider retention/training policy permit
those bytes to leave the workstation. HTTP redirects are never followed, so a
provider cannot redirect the archive context or bearer token to another URL.
The API key is neither returned nor included in configuration hashes.
Token prices are optional nonnegative USD values and become part of the
configuration hash. Cost is calculated only from provider-reported prompt and
completion token counts; the system never guesses token usage from character
length. Missing usage remains explicit in responses and benchmark reports.

Restart `wic-api` after configuration changes. `/api/health` reports
`generation_configured=false` and a sanitized configuration error when the
contract is incomplete or invalid; it does not contact the model endpoint.

## Output gate

Every usable output must contain at least one exact citation of the form
`[region:UUID]`. The server resolves each UUID against the request's allowed
context. Malformed or foreign citations reject the entire model text. The raw
rejected text is withheld; only its SHA-256 remains for audit/debugging.

A reconstructed scene has stricter rules:

1. Reviewed claims must exist before the provider is loaded.
2. `Direct evidence`, `Plausible reconstruction`, and `Speculative details`
   must appear in that order.
3. `Direct evidence` must contain at least one citation to a reviewed claim.
4. OCR-only retrieval leads are never allowed as scene citations.

Responses expose status, model/revision, provider, configuration hash, prompt
hash, exact context hash, raw-output hash, resolved scan citations, validation
errors, token usage, estimated cost and warnings. The browser renders these
fields with `textContent`; model output is never interpreted as HTML.

Statuses are:

- `completed`: provider output passed the structural evidence gate;
- `rejected`: a provider responded, but its text failed citation/scene checks
  and was withheld;
- `abstained`: retrieval or reviewed evidence was insufficient, so no provider
  call was made;
- `unavailable`: no provider is configured.

Passing this gate proves citation structure, not historical truth or model
quality. Historians must still assess the cited evidence and the boundary
between evidence, plausible reconstruction and speculation.

The selection protocol and executable commands are documented in
[`experiments/generation/README.md`](../experiments/generation/README.md).
Do not infer a model winner from transport success, citation acceptance, token
cost, or an unpaired mean score.
