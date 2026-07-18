const form = document.querySelector('#search-form');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const warnings = document.querySelector('#warnings');
const contextButton = document.querySelector('#context-button');
const contextPanel = document.querySelector('#context-panel');
const contextJson = document.querySelector('#context-json');
const briefButton = document.querySelector('#brief-button');
const sceneButton = document.querySelector('#scene-button');
const generationPanel = document.querySelector('#generation-panel');
const generationLabel = document.querySelector('#generation-label');
const generationOutput = document.querySelector('#generation-output');
const reviewButton = document.querySelector('#review-button');
const reviewPanel = document.querySelector('#review-panel');
const reviewItems = document.querySelector('#review-items');
const reviewSummary = document.querySelector('#review-summary');
const reviewerInput = document.querySelector('#reviewer');
const moreReview = document.querySelector('#more-review');
const insightsButton = document.querySelector('#insights-button');
const insightsPanel = document.querySelector('#insights-panel');
const claimReviewItems = document.querySelector('#claim-review-items');
const claimReviewSummary = document.querySelector('#claim-review-summary');
const moreClaims = document.querySelector('#more-claims');
let lastRequest = null;
let reviewOffset = 0;
const reviewLimit = 20;
let reviewTotal = 0;
let claimOffset = 0;
const claimLimit = 20;
let claimTotal = 0;

reviewerInput.value = localStorage.getItem('wic-reviewer') || '';
reviewerInput.addEventListener('change', () => {
  localStorage.setItem('wic-reviewer', reviewerInput.value.trim());
});

function reviewer() {
  const value = reviewerInput.value.trim();
  if (!value) {
    reviewerInput.focus();
    throw new Error('Enter a reviewer name before making decisions.');
  }
  return value;
}

function pageImageUrl(volumeNumber, pageNumber, derivativeId = null) {
  const base = `/api/page-image/${volumeNumber}/${pageNumber}`;
  return derivativeId ? `${base}?derivative_id=${encodeURIComponent(derivativeId)}` : base;
}

function appendHighlightedText(node, text, start, end) {
  node.replaceChildren();
  node.append(document.createTextNode(text.slice(0, start)));
  const mark = document.createElement('mark');
  mark.textContent = text.slice(start, end);
  node.append(mark, document.createTextNode(text.slice(end)));
}

async function postReview(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({...body, review_id: crypto.randomUUID(), reviewer: reviewer()}),
  });
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}

function renderResolutionActions(container, item) {
  container.replaceChildren();
  container.hidden = false;
  const nil = item.link_candidates.find(candidate => candidate.is_nil);
  const existing = item.link_candidates.filter(candidate => !candidate.is_nil);
  if (existing.length) {
    const select = document.createElement('select');
    existing.forEach(candidate => {
      const option = document.createElement('option');
      option.value = candidate.link_candidate_id;
      option.textContent = `${candidate.proposed_canonical_name} · ${candidate.score.toFixed(3)}`;
      select.append(option);
    });
    const link = document.createElement('button');
    link.type = 'button';
    link.textContent = 'Link reviewed entity';
    link.addEventListener('click', async () => {
      try {
        await postReview(`/api/review/mentions/${item.mention_id}/entity-resolution`, {
          selected_link_candidate_id: select.value, action: 'link_existing',
        });
        container.textContent = 'Linked to reviewed entity.';
      } catch (error) { container.textContent = error.message; }
    });
    container.append(select, link);
  }
  if (nil) {
    const create = document.createElement('button');
    create.type = 'button';
    create.textContent = 'Create reviewed entity';
    create.addEventListener('click', async () => {
      const name = window.prompt('Canonical name for the new reviewed entity:', item.mention_text);
      if (!name || !name.trim()) return;
      if (!window.confirm(`Create a reviewed ${item.entity_type} entity named “${name.trim()}”?`)) return;
      try {
        const result = await postReview(`/api/review/mentions/${item.mention_id}/entity-resolution`, {
          selected_link_candidate_id: nil.link_candidate_id,
          action: 'create_new',
          canonical_name: name.trim(),
        });
        container.textContent = `Created reviewed entity ${result.entity_id}.`;
      } catch (error) { container.textContent = error.message; }
    });
    const keepNil = document.createElement('button');
    keepNil.type = 'button';
    keepNil.className = 'quiet';
    keepNil.textContent = 'Keep unresolved / NIL';
    keepNil.addEventListener('click', async () => {
      try {
        await postReview(`/api/review/mentions/${item.mention_id}/entity-resolution`, {
          selected_link_candidate_id: nil.link_candidate_id, action: 'keep_nil',
        });
        container.textContent = 'Accepted span remains unresolved (NIL).';
      } catch (error) { container.textContent = error.message; }
    });
    container.append(create, keepNil);
  }
  if (!item.link_candidates.length) {
    container.textContent = 'No entity-link candidates exist for this model run yet.';
  }
}

function renderMention(item) {
  const fragment = document.querySelector('#mention-template').content.cloneNode(true);
  const card = fragment.querySelector('.mention-card');
  fragment.querySelector('.mention-meta').textContent = `${item.entity_type} · “${item.mention_text}” · ${(item.confidence || 0).toFixed(3)}`;
  appendHighlightedText(fragment.querySelector('.mention-context'), item.region_text, item.text_start, item.text_end);
  const inputProvenance = item.input_variant
    ? ` · ${item.input_variant} · ${item.dataset_id || 'unassigned'} / ${item.split_id || 'unassigned'}`
    : '';
  fragment.querySelector('.mention-provenance').textContent = `Volume ${item.volume_number} · ${item.publication_year} · page ${item.page_number} · ${item.model_name} @ ${item.model_revision}${inputProvenance}`;
  fragment.querySelector('.mention-scan').href = pageImageUrl(
    item.volume_number, item.page_number, item.derivative_id
  );
  const statusNode = fragment.querySelector('.decision-status');
  const resolution = fragment.querySelector('.resolution-actions');
  const buttons = fragment.querySelectorAll('.accept-mention, .reject-mention, .defer-mention');
  async function decide(decision) {
    buttons.forEach(button => { button.disabled = true; });
    try {
      const result = await postReview(`/api/review/mentions/${item.mention_id}`, {decision});
      statusNode.textContent = `Recorded: ${result.action} · ${result.review_id}`;
      if (decision === 'accept') renderResolutionActions(resolution, item);
      else card.classList.add('decided');
    } catch (error) {
      statusNode.textContent = error.message;
      buttons.forEach(button => { button.disabled = false; });
    }
  }
  fragment.querySelector('.accept-mention').addEventListener('click', () => decide('accept'));
  fragment.querySelector('.reject-mention').addEventListener('click', () => decide('reject'));
  fragment.querySelector('.defer-mention').addEventListener('click', () => decide('needs_review'));
  reviewItems.append(fragment);
}

async function loadReview(reset = false) {
  if (reset) {
    reviewOffset = 0;
    reviewItems.replaceChildren();
  }
  reviewSummary.textContent = 'Loading candidates…';
  const response = await fetch(`/api/review/mentions?status=candidate&limit=${reviewLimit}&offset=${reviewOffset}`);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  const data = await response.json();
  reviewTotal = data.total;
  data.items.forEach(renderMention);
  reviewOffset += data.items.length;
  reviewSummary.textContent = `${reviewOffset} of ${reviewTotal} candidate spans loaded`;
  moreReview.hidden = reviewOffset >= reviewTotal;
}

reviewButton.addEventListener('click', async () => {
  reviewPanel.hidden = false;
  reviewPanel.scrollIntoView({behavior: 'smooth'});
  try {
    await Promise.all([loadReview(true), loadClaims(true)]);
  } catch (error) {
    reviewSummary.textContent = error.message;
    claimReviewSummary.textContent = error.message;
  }
});
document.querySelector('#close-review').addEventListener('click', () => {
  reviewPanel.hidden = true;
  document.querySelector('#search-form').scrollIntoView({behavior: 'smooth'});
});
document.querySelector('#refresh-review').addEventListener('click', () => loadReview(true));
moreReview.addEventListener('click', () => loadReview(false));

function claimObjectLabel(item) {
  if (item.object_canonical_name) return item.object_canonical_name;
  return JSON.stringify(item.object_literal);
}

function renderClaim(item) {
  const fragment = document.querySelector('#claim-template').content.cloneNode(true);
  const card = fragment.querySelector('.claim-card');
  fragment.querySelector('.claim-statement').textContent = `${item.subject_canonical_name} — ${item.predicate} → ${claimObjectLabel(item)}`;
  fragment.querySelector('.claim-provenance').textContent = `${item.model_name} @ ${item.model_revision} · confidence ${(item.confidence || 0).toFixed(3)} · ${item.claim_id}`;
  const evidenceNode = fragment.querySelector('.claim-evidence');
  item.evidence.forEach(evidence => {
    const block = document.createElement('blockquote');
    block.textContent = evidence.evidence_quote;
    const citation = document.createElement('a');
    citation.className = 'scan-link';
    citation.target = '_blank';
    citation.rel = 'noopener';
    citation.href = pageImageUrl(
      evidence.volume_number, evidence.page_number, evidence.derivative_id
    );
    citation.textContent = `Volume ${evidence.volume_number} · ${evidence.publication_year} · page ${evidence.page_number} ↗`;
    evidenceNode.append(block, citation);
  });
  if (!item.evidence.length) {
    const warning = document.createElement('p');
    warning.className = 'warning';
    warning.textContent = 'No cited evidence is attached. This claim cannot be accepted.';
    evidenceNode.append(warning);
  }
  const statusNode = fragment.querySelector('.decision-status');
  const buttons = fragment.querySelectorAll('.accept-claim, .reject-claim, .dispute-claim, .defer-claim');
  async function decide(decision) {
    buttons.forEach(button => { button.disabled = true; });
    try {
      const result = await postReview(`/api/review/claims/${item.claim_id}`, {decision});
      const projectionNotice = decision === 'accept'
        ? ' Rebuild the graph before using graph analysis.'
        : ' The reviewed graph is unchanged.';
      statusNode.textContent = `Recorded: ${result.action} · ${result.review_id}.${projectionNotice}`;
      card.classList.add('decided');
    } catch (error) {
      statusNode.textContent = error.message;
      buttons.forEach(button => { button.disabled = false; });
    }
  }
  fragment.querySelector('.accept-claim').addEventListener('click', () => decide('accept'));
  fragment.querySelector('.reject-claim').addEventListener('click', () => decide('reject'));
  fragment.querySelector('.dispute-claim').addEventListener('click', () => decide('dispute'));
  fragment.querySelector('.defer-claim').addEventListener('click', () => decide('needs_review'));
  claimReviewItems.append(fragment);
}

async function loadClaims(reset = false) {
  if (reset) {
    claimOffset = 0;
    claimReviewItems.replaceChildren();
  }
  claimReviewSummary.textContent = 'Loading candidate claims…';
  const response = await fetch(`/api/review/claims?status=candidate&limit=${claimLimit}&offset=${claimOffset}`);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  const data = await response.json();
  claimTotal = data.total;
  data.items.forEach(renderClaim);
  claimOffset += data.items.length;
  claimReviewSummary.textContent = `${claimOffset} of ${claimTotal} candidate claims loaded`;
  moreClaims.hidden = claimOffset >= claimTotal;
}

document.querySelector('#refresh-claims').addEventListener('click', () => loadClaims(true));
moreClaims.addEventListener('click', () => loadClaims(false));

async function loadInsights() {
  const counts = document.querySelector('#insight-counts');
  const warningsNode = document.querySelector('#insight-warnings');
  const itemsNode = document.querySelector('#insight-items');
  counts.textContent = 'Loading reviewed graph…';
  warningsNode.replaceChildren();
  itemsNode.replaceChildren();
  const response = await fetch('/api/insights');
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  const data = await response.json();
  const values = data.evidence_counts;
  const graph = data.graph_projection;
  counts.textContent = `${values.reviewed_entities} entities · ${values.reviewed_mentions} mentions · ${values.reviewed_claims} claims · ${values.reviewed_claim_evidence} evidence links · graph ${graph.stale ? 'STALE' : 'current/empty'}`;
  const graphState = document.createElement('p');
  graphState.className = graph.stale ? 'warning' : 'projection-state';
  graphState.textContent = graph.reason;
  warningsNode.append(graphState);
  data.warnings.forEach(message => {
    const warning = document.createElement('p');
    warning.className = 'warning';
    warning.textContent = message;
    warningsNode.append(warning);
  });
  data.items.forEach(item => {
    const card = document.createElement('article');
    card.className = 'mention-card';
    const label = document.createElement('p');
    label.className = 'eyebrow';
    label.textContent = item.kind.replaceAll('_', ' ');
    const heading = document.createElement('h3');
    heading.textContent = item.title;
    const summary = document.createElement('p');
    summary.textContent = item.summary;
    const details = document.createElement('pre');
    details.textContent = JSON.stringify({metrics: item.metrics, entity_ids: item.entity_ids, claim_ids: item.claim_ids, supporting_page_ids: item.supporting_page_ids}, null, 2);
    card.append(label, heading, summary, details);
    itemsNode.append(card);
  });
}

insightsButton.addEventListener('click', async () => {
  insightsPanel.hidden = false;
  insightsPanel.scrollIntoView({behavior: 'smooth'});
  try { await loadInsights(); } catch (error) {
    document.querySelector('#insight-counts').textContent = error.message;
  }
});
document.querySelector('#close-insights').addEventListener('click', () => {
  insightsPanel.hidden = true;
});

function requestBody() {
  const body = {
    query: document.querySelector('#query').value.trim(),
    mode: document.querySelector('#mode').value,
    limit: 10,
  };
  const yearStart = document.querySelector('#year-start').value;
  const yearEnd = document.querySelector('#year-end').value;
  if (yearStart) body.year_start = Number(yearStart);
  if (yearEnd) body.year_end = Number(yearEnd);
  return body;
}

function render(data) {
  results.replaceChildren();
  warnings.replaceChildren();
  for (const message of data.warnings) {
    const node = document.createElement('p');
    node.className = 'warning';
    node.textContent = message;
    warnings.append(node);
  }
  const template = document.querySelector('#result-template');
  data.hits.forEach(hit => {
    const card = template.content.cloneNode(true);
    card.querySelector('.rank').textContent = String(hit.rank).padStart(2, '0');
    card.querySelector('.citation').textContent = `Volume ${hit.source.volume_number} · ${hit.source.publication_year} · page ${hit.source.page_number}`;
    card.querySelector('blockquote').textContent = hit.text || '〔empty OCR region〕';
    card.querySelector('.score').textContent = `${hit.explanation.retriever} · score ${hit.score.toFixed(5)}`;
    card.querySelector('.pointer').textContent = JSON.stringify(hit.source, null, 2);
    card.querySelector('.scan-link').href = pageImageUrl(
      hit.source.volume_number, hit.source.page_number, hit.source.derivative_id
    );
    results.append(card);
  });
  status.textContent = `${data.hits.length} cited region${data.hits.length === 1 ? '' : 's'} · ${data.mode}`;
  contextButton.disabled = false;
  briefButton.disabled = false;
  sceneButton.disabled = false;
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  lastRequest = requestBody();
  status.textContent = lastRequest.mode === 'lexical' ? 'Searching…' : 'Loading multilingual retrieval model…';
  contextButton.disabled = true;
  contextPanel.hidden = true;
  try {
    const response = await fetch('/api/search', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(lastRequest),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    render(await response.json());
  } catch (error) {
    status.textContent = `Search failed: ${error.message}`;
  }
});

async function generate(task, button) {
  if (!lastRequest) return;
  const original = button.textContent;
  button.textContent = 'Preparing…';
  briefButton.disabled = true;
  sceneButton.disabled = true;
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...lastRequest, task}),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const data = await response.json();
    generationLabel.textContent = `${data.task.replaceAll('_', ' ')} · ${data.status}`;
    generationOutput.textContent = data.output;
    generationPanel.hidden = false;
    generationPanel.scrollIntoView({behavior: 'smooth'});
  } catch (error) {
    status.textContent = `Generation failed: ${error.message}`;
  } finally {
    button.textContent = original;
    briefButton.disabled = false;
    sceneButton.disabled = false;
  }
}

briefButton.addEventListener('click', () => generate('research_brief', briefButton));
sceneButton.addEventListener('click', () => generate('reconstructed_scene', sceneButton));

contextButton.addEventListener('click', async () => {
  if (!lastRequest) return;
  contextButton.textContent = 'Preparing…';
  try {
    const response = await fetch('/api/context', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(lastRequest),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    contextJson.textContent = JSON.stringify(await response.json(), null, 2);
    contextPanel.hidden = false;
    contextPanel.scrollIntoView({behavior: 'smooth'});
  } catch (error) {
    status.textContent = `Context export failed: ${error.message}`;
  } finally {
    contextButton.textContent = 'Prepare model context';
  }
});

document.querySelector('#copy-context').addEventListener('click', async event => {
  await navigator.clipboard.writeText(contextJson.textContent);
  event.currentTarget.textContent = 'Copied';
  setTimeout(() => { event.currentTarget.textContent = 'Copy JSON'; }, 1200);
});
