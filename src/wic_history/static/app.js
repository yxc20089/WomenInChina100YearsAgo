const form = document.querySelector('#search-form');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const warnings = document.querySelector('#warnings');
const contextButton = document.querySelector('#context-button');
const contextPanel = document.querySelector('#context-panel');
const contextJson = document.querySelector('#context-json');
const briefButton = document.querySelector('#brief-button');
const sceneButton = document.querySelector('#scene-button');
const chatButton = document.querySelector('#chat-button');
const chatPanel = document.querySelector('#chat-panel');
const chatForm = document.querySelector('#chat-form');
const chatQuestion = document.querySelector('#chat-question');
const chatSend = document.querySelector('#chat-send');
const chatTranscript = document.querySelector('#chat-transcript');
const generationPanel = document.querySelector('#generation-panel');
const generationLabel = document.querySelector('#generation-label');
const generationOutput = document.querySelector('#generation-output');
const generationProvenance = document.querySelector('#generation-provenance');
const generationCitations = document.querySelector('#generation-citations');
const generationWarnings = document.querySelector('#generation-warnings');
const reviewButton = document.querySelector('#review-button');
const reviewPanel = document.querySelector('#review-panel');
const reviewItems = document.querySelector('#review-items');
const reviewSummary = document.querySelector('#review-summary');
const reviewerInput = document.querySelector('#reviewer');
const moreReview = document.querySelector('#more-review');
const insightsButton = document.querySelector('#insights-button');
const insightsPanel = document.querySelector('#insights-panel');
const explorationButton = document.querySelector('#exploration-button');
const explorationPanel = document.querySelector('#exploration-panel');
const claimReviewItems = document.querySelector('#claim-review-items');
const claimReviewSummary = document.querySelector('#claim-review-summary');
const moreClaims = document.querySelector('#more-claims');
const segmentationItems = document.querySelector('#segmentation-items');
const segmentationSummary = document.querySelector('#segmentation-summary');
const moreSegmentations = document.querySelector('#more-segmentations');
const segmentationDetail = document.querySelector('#segmentation-detail');
const segmentationEditor = document.querySelector('#segmentation-editor');
const segmentationStatus = document.querySelector('#segmentation-decision-status');
let lastRequest = null;
let reviewOffset = 0;
const reviewLimit = 20;
let reviewTotal = 0;
let claimOffset = 0;
const claimLimit = 20;
let claimTotal = 0;
let segmentationOffset = 0;
const segmentationLimit = 20;
let segmentationTotal = 0;
let currentSegmentation = null;
let chatHistory = [];

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

async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}

function renderSegmentationSummary(item) {
  const card = document.createElement('article');
  card.className = 'mention-card segmentation-card';
  const heading = document.createElement('h3');
  heading.textContent = `Volume ${item.volume_number} · ${item.publication_year} · page ${item.page_number}`;
  const kind = document.createElement('p');
  kind.className = 'mention-meta';
  kind.textContent = `${item.proposal_kind.replaceAll('_', ' ')} · ${item.method} v${item.method_version}`;
  const counts = document.createElement('p');
  counts.textContent = `${item.units} proposed unit(s) · ${item.member_spans} source span(s) · ${item.review_count} review(s) · ${item.approved_units} active approved unit(s)`;
  const provenance = document.createElement('p');
  provenance.className = 'mention-provenance';
  provenance.textContent = `Proposed by ${item.proposed_by} · ${item.run_id} · derivative ${item.derivative_id} · ${item.evidence_tier}`;
  const state = document.createElement('p');
  state.className = item.source_selection_active ? 'decision-status' : 'warning';
  if (item.active_selection_id) {
    state.textContent = `ACTIVE reviewed selection ${item.active_selection_id}`;
  } else if (!item.source_selection_active) {
    state.textContent = 'STALE: the source OCR selection is no longer active.';
  } else if (item.latest_decision) {
    state.textContent = `Latest review: ${item.latest_decision} by ${item.latest_reviewer}`;
  } else {
    state.textContent = 'Unreviewed proposal.';
  }
  const inspect = document.createElement('button');
  inspect.type = 'button';
  inspect.textContent = 'Inspect every unit';
  inspect.addEventListener('click', () => openSegmentation(item.run_id));
  const scan = document.createElement('a');
  scan.className = 'mention-scan';
  scan.target = '_blank';
  scan.rel = 'noopener';
  scan.href = pageImageUrl(item.volume_number, item.page_number, item.derivative_id);
  scan.textContent = 'Open exact scan ↗';
  const actions = document.createElement('div');
  actions.className = 'mention-actions';
  actions.append(inspect, scan);
  card.append(kind, heading, counts, provenance, state, actions);
  segmentationItems.append(card);
}

async function loadSegmentations(reset = false) {
  if (reset) {
    segmentationOffset = 0;
    segmentationItems.replaceChildren();
  }
  segmentationSummary.textContent = 'Loading segmentation proposals…';
  const response = await fetch(`/api/review/segmentations?limit=${segmentationLimit}&offset=${segmentationOffset}`);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  const data = await response.json();
  segmentationTotal = data.total;
  data.items.forEach(renderSegmentationSummary);
  segmentationOffset += data.items.length;
  segmentationSummary.textContent = `${segmentationOffset} of ${segmentationTotal} immutable proposals loaded. ${data.warnings.join(' ')}`;
  moreSegmentations.hidden = segmentationOffset >= segmentationTotal;
}

function renderSegmentationUnits(units) {
  const container = document.querySelector('#segmentation-unit-previews');
  container.replaceChildren();
  units.forEach(unit => {
    const details = document.createElement('details');
    details.className = 'segmentation-unit';
    const summary = document.createElement('summary');
    summary.textContent = `Unit ${unit.ordinal} · ${unit.unit_kind} · ${unit.region_spans} exact span(s)${unit.title ? ` · ${unit.title}` : ''}`;
    const text = document.createElement('pre');
    text.textContent = unit.text || '〔empty OCR spans〕';
    const spans = document.createElement('div');
    spans.className = 'segmentation-spans';
    unit.spans.forEach(span => {
      const spanDetails = document.createElement('details');
      const spanSummary = document.createElement('summary');
      spanSummary.textContent = `OCR ${span.reading_order} · ${span.region_id} · [${span.text_start}, ${span.text_end}) · “${span.text}”`;
      const geometry = document.createElement('pre');
      geometry.textContent = JSON.stringify(span.polygon, null, 2);
      spanDetails.append(spanSummary, geometry);
      spans.append(spanDetails);
    });
    details.append(summary, text, spans);
    container.append(details);
  });
}

function renderSegmentationReviews(data) {
  const container = document.querySelector('#segmentation-review-history');
  container.replaceChildren();
  const heading = document.createElement('h4');
  heading.textContent = 'Immutable review history';
  container.append(heading);
  if (!data.reviews.length) {
    const empty = document.createElement('p');
    empty.textContent = 'No reviews recorded.';
    container.append(empty);
    return;
  }
  data.reviews.forEach(review => {
    const row = document.createElement('div');
    row.className = 'segmentation-review-row';
    const text = document.createElement('p');
    text.textContent = `${review.decision} · ${review.reviewer} · ${review.reviewed_at}${review.note ? ` · ${review.note}` : ''}`;
    row.append(text);
    if (review.activated_selection_id) {
      const state = document.createElement('p');
      state.className = 'decision-status';
      state.textContent = `${review.selection_active ? 'Active' : 'Superseded'} selection ${review.activated_selection_id}`;
      row.append(state);
    } else if (review.decision === 'accept') {
      const activate = document.createElement('button');
      activate.type = 'button';
      activate.className = 'activation-button';
      activate.textContent = 'Activate this accepted review';
      activate.disabled = !data.reviewable;
      activate.addEventListener('click', async () => {
        try {
          const previous = data.summary.current_page_selection_id || 'none';
          if (!window.confirm(`Activate ${data.units.length} reviewed units and supersede current page selection ${previous}? This is separate from acceptance.`)) return;
          activate.disabled = true;
          const result = await postJson(`/api/review/segmentation-reviews/${review.review_id}/activate`, {
            selected_by: reviewer(),
            expected_previous_selection_id: data.summary.current_page_selection_id,
            expected_proposal_sha256: data.summary.proposal_sha256,
            confirmation: 'ACTIVATE_ACCEPTED_SEGMENTATION',
          });
          segmentationStatus.textContent = `Activated selection ${result.selection_id}; ${result.approved_units} approved units copied.${result.reused ? ' Existing activation reused.' : ''}`;
          await Promise.all([openSegmentation(data.summary.run_id), loadSegmentations(true)]);
        } catch (error) {
          segmentationStatus.textContent = error.message;
          activate.disabled = false;
        }
      });
      row.append(activate);
    }
    container.append(row);
  });
}

async function openSegmentation(runId) {
  segmentationDetail.hidden = false;
  segmentationStatus.textContent = 'Loading exact spans and scan provenance…';
  const response = await fetch(`/api/review/segmentations/${runId}`);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  const data = await response.json();
  currentSegmentation = data;
  document.querySelector('#segmentation-detail-title').textContent = `Volume ${data.summary.volume_number} · page ${data.summary.page_number} · ${data.units.length} proposed units`;
  document.querySelector('#segmentation-detail-meta').textContent = `Run ${data.summary.run_id} · proposal ${data.summary.proposal_sha256} · input ${data.summary.input_sha256} · derivative ${data.summary.derivative_id} / ${data.summary.image_sha256} · ${data.summary.image_width}×${data.summary.image_height} · ${data.summary.evidence_tier}`;
  const scan = document.querySelector('#segmentation-scan');
  scan.href = pageImageUrl(data.summary.volume_number, data.summary.page_number, data.summary.derivative_id);
  const warningContainer = document.querySelector('#segmentation-detail-warnings');
  warningContainer.replaceChildren();
  [...data.warnings, ...data.review_blockers].forEach(message => {
    const warning = document.createElement('p');
    warning.className = 'warning';
    warning.textContent = message;
    warningContainer.append(warning);
  });
  const coverage = document.createElement('p');
  coverage.className = data.coverage_complete ? 'decision-status' : 'warning';
  coverage.textContent = `${data.covered_regions}/${data.source_regions} OCR regions · ${data.member_spans} exact spans · coverage ${data.coverage_complete ? 'complete' : 'BLOCKED'} · scan hash ${data.scan_available ? 'verified' : 'BLOCKED'}`;
  warningContainer.prepend(coverage);
  renderSegmentationUnits(data.units);
  segmentationEditor.value = JSON.stringify(data.editable_artifact, null, 2);
  document.querySelector('#segmentation-review-note').value = '';
  const decisionButtons = [
    document.querySelector('#accept-segmentation'),
    document.querySelector('#reject-segmentation'),
    document.querySelector('#defer-segmentation'),
  ];
  decisionButtons.forEach(button => { button.disabled = !data.reviewable; });
  renderSegmentationReviews(data);
  segmentationStatus.textContent = data.reviewable
    ? 'Loaded. Acceptance will record a review only; it will not activate units.'
    : `Review blocked: ${data.review_blockers.join(' ')}`;
  segmentationDetail.scrollIntoView({behavior: 'smooth', block: 'start'});
  return data;
}

async function recordSegmentationDecision(decision) {
  if (!currentSegmentation) return;
  try {
    const note = document.querySelector('#segmentation-review-note').value.trim() || null;
    if (decision === 'accept' && !window.confirm('Confirm that you inspected every unit, every exact span, and the registered scan. This records acceptance but does NOT activate it.')) return;
    const result = await postJson(`/api/review/segmentations/${currentSegmentation.summary.run_id}/reviews`, {
      review_id: crypto.randomUUID(),
      decision,
      reviewer: reviewer(),
      note,
      expected_proposal_sha256: currentSegmentation.summary.proposal_sha256,
      expected_input_sha256: currentSegmentation.summary.input_sha256,
      checked_all_units: decision === 'accept',
      confirmation: 'RECORD_REVIEW_WITHOUT_ACTIVATION',
    });
    segmentationStatus.textContent = `Recorded ${result.decision} review ${result.review_id}. No activation occurred.`;
    await Promise.all([openSegmentation(currentSegmentation.summary.run_id), loadSegmentations(true)]);
  } catch (error) {
    segmentationStatus.textContent = error.message;
  }
}

document.querySelector('#import-segmentation').addEventListener('click', async () => {
  try {
    const artifact = JSON.parse(segmentationEditor.value);
    if (!window.confirm('Create a NEW immutable, unapproved proposal from this JSON? This does not alter or approve the current proposal.')) return;
    const result = await postJson('/api/review/segmentation-imports', {
      artifact,
      proposed_by: reviewer(),
      confirmation: 'CREATE_UNAPPROVED_PROPOSAL',
    });
    segmentationStatus.textContent = `${result.reused ? 'Reused' : 'Created'} unapproved proposal ${result.run_id}.`;
    await loadSegmentations(true);
    await openSegmentation(result.run_id);
  } catch (error) {
    segmentationStatus.textContent = `Import blocked: ${error.message}`;
  }
});

document.querySelector('#accept-segmentation').addEventListener('click', () => recordSegmentationDecision('accept'));
document.querySelector('#reject-segmentation').addEventListener('click', () => recordSegmentationDecision('reject'));
document.querySelector('#defer-segmentation').addEventListener('click', () => recordSegmentationDecision('needs_revision'));
document.querySelector('#close-segmentation-detail').addEventListener('click', () => {
  segmentationDetail.hidden = true;
  currentSegmentation = null;
});
document.querySelector('#refresh-segmentations').addEventListener('click', () => loadSegmentations(true));
moreSegmentations.addEventListener('click', () => loadSegmentations(false));

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
    await Promise.all([loadSegmentations(true), loadReview(true), loadClaims(true)]);
  } catch (error) {
    segmentationSummary.textContent = error.message;
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

function shortModelName(value) {
  return value.split('/').at(-1).replace('+historical-women-zh-rules', '+rules');
}

async function loadExploration() {
  const countsNode = document.querySelector('#exploration-counts');
  const warningsNode = document.querySelector('#exploration-warnings');
  const themesNode = document.querySelector('#exploration-themes');
  const runsNode = document.querySelector('#exploration-ner-runs');
  const agreementsNode = document.querySelector('#exploration-ner-agreements');
  countsNode.textContent = 'Loading active OCR and candidate diagnostics…';
  warningsNode.replaceChildren();
  themesNode.replaceChildren();
  runsNode.replaceChildren();
  agreementsNode.replaceChildren();
  const response = await fetch('/api/exploration');
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  const data = await response.json();
  const counts = data.counts;
  const meanConfidence = counts.mean_ocr_confidence == null ? 'n/a' : counts.mean_ocr_confidence.toFixed(3);
  countsNode.textContent = `${counts.active_pages} active page(s) · ${counts.active_regions} OCR regions · ${counts.text_characters} characters · mean OCR confidence ${meanConfidence} · ${counts.low_confidence_regions} below 0.5 · ${counts.candidate_mentions} NER candidates`;
  data.warnings.forEach(message => {
    const warning = document.createElement('p');
    warning.className = 'warning';
    warning.textContent = message;
    warningsNode.append(warning);
  });
  data.themes.forEach(theme => {
    const card = document.createElement('article');
    card.className = `exploration-card${theme.matched_regions ? '' : ' no-matches'}`;
    const title = document.createElement('h3');
    title.textContent = theme.title;
    const meta = document.createElement('p');
    meta.className = 'mention-provenance';
    meta.textContent = theme.matched_regions
      ? `${theme.matched_regions} region(s) on ${theme.matched_pages} page(s) · ${theme.year_start}${theme.year_end !== theme.year_start ? `–${theme.year_end}` : ''}`
      : 'No matches in the current active OCR scope';
    const prompt = document.createElement('p');
    prompt.textContent = theme.research_prompt;
    card.append(title, meta, prompt);
    theme.examples.forEach(example => {
      const evidence = document.createElement('div');
      evidence.className = 'exploration-evidence';
      const quote = document.createElement('blockquote');
      quote.textContent = example.text;
      const details = document.createElement('span');
      details.className = 'mention-provenance';
      const confidence = example.confidence == null ? 'n/a' : example.confidence.toFixed(3);
      details.textContent = `Volume ${example.source.volume_number} · ${example.source.publication_year} · page ${example.source.page_number} · OCR ${confidence}`;
      const scan = document.createElement('a');
      scan.className = 'scan-link';
      scan.target = '_blank';
      scan.rel = 'noopener';
      scan.href = pageImageUrl(
        example.source.volume_number, example.source.page_number, example.source.derivative_id
      );
      scan.textContent = 'Open cited scan ↗';
      evidence.append(quote, details, scan);
      card.append(evidence);
    });
    themesNode.append(card);
  });
  data.ner_runs.forEach(run => {
    const item = document.createElement('p');
    item.className = 'diagnostic-row';
    const confidence = run.mean_confidence == null ? 'n/a' : run.mean_confidence.toFixed(3);
    item.textContent = `${shortModelName(run.model_name)} · ${run.candidate_mentions} candidates across ${run.cited_regions} regions · mean uncalibrated score ${confidence} · input ${run.input_regions ?? 'n/a'} regions`;
    runsNode.append(item);
  });
  data.ner_agreements.forEach(signal => {
    const item = document.createElement('p');
    item.className = 'diagnostic-row';
    item.textContent = `${shortModelName(signal.left_model)} ↔ ${shortModelName(signal.right_model)} · ${signal.exact_agreements} exact agreements · candidate Jaccard ${signal.candidate_jaccard.toFixed(4)}`;
    agreementsNode.append(item);
  });
}

explorationButton.addEventListener('click', async () => {
  explorationPanel.hidden = false;
  explorationPanel.scrollIntoView({behavior: 'smooth'});
  try { await loadExploration(); } catch (error) {
    document.querySelector('#exploration-counts').textContent = error.message;
  }
});
document.querySelector('#close-exploration').addEventListener('click', () => {
  explorationPanel.hidden = true;
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
  chatButton.disabled = false;
  briefButton.disabled = false;
  sceneButton.disabled = false;
}

const submitButton = form.querySelector('button[type="submit"]');
const modeSelect = document.querySelector('#mode');

async function runSearch() {
  if (!document.querySelector('#query').value.trim()) return;
  lastRequest = requestBody();
  status.textContent = lastRequest.mode === 'lexical' ? 'Searching…' : 'Loading multilingual retrieval model…';
  submitButton.disabled = true;
  submitButton.textContent = 'Searching…';
  modeSelect.disabled = true;
  contextButton.disabled = true;
  chatButton.disabled = true;
  contextPanel.hidden = true;
  try {
    const response = await fetch('/api/search', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(lastRequest),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    render(await response.json());
  } catch (error) {
    status.textContent = `Search failed: ${error.message}`;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Search';
    modeSelect.disabled = false;
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  runSearch();
});

// Re-run automatically when the mode changes, so the toggle feels live.
document.querySelector('#mode').addEventListener('change', () => {
  if (lastRequest) runSearch();
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
    generationProvenance.textContent = [
      data.model ? `model ${data.model}` : null,
      data.prompt_sha256 ? `prompt ${data.prompt_sha256.slice(0, 12)}…` : null,
      data.context_sha256 ? `context ${data.context_sha256.slice(0, 12)}…` : null,
      data.raw_output_sha256 ? `output ${data.raw_output_sha256.slice(0, 12)}…` : null,
      data.total_tokens !== null ? `${data.total_tokens} tokens` : null,
      data.estimated_cost_usd !== null ? `$${data.estimated_cost_usd.toFixed(6)} estimated` : null,
    ].filter(Boolean).join(' · ');
    generationCitations.replaceChildren();
    data.citations.forEach(source => {
      const link = document.createElement('a');
      link.className = 'scan-link';
      link.target = '_blank';
      link.rel = 'noopener';
      link.href = pageImageUrl(source.volume_number, source.page_number, source.derivative_id);
      link.textContent = `Volume ${source.volume_number} · ${source.publication_year} · page ${source.page_number} ↗`;
      generationCitations.append(link);
    });
    generationWarnings.replaceChildren();
    [...data.validation_errors, ...data.warnings].forEach(message => {
      const warning = document.createElement('p');
      warning.className = 'chat-warning';
      warning.textContent = message;
      generationWarnings.append(warning);
    });
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

function appendChatTurn(role, content, citations = [], messages = [], provenance = '') {
  const turn = document.createElement('article');
  turn.className = `chat-turn ${role}`;
  const label = document.createElement('p');
  label.className = 'eyebrow';
  label.textContent = role === 'user' ? 'Researcher' : 'Assistant';
  const output = document.createElement('div');
  output.className = 'chat-content';
  output.textContent = content;
  turn.append(label, output);
  if (provenance) {
    const metadata = document.createElement('p');
    metadata.className = 'generation-provenance';
    metadata.textContent = provenance;
    turn.append(metadata);
  }
  if (citations.length) {
    const citationList = document.createElement('div');
    citationList.className = 'chat-citations';
    citations.forEach(source => {
      const link = document.createElement('a');
      link.className = 'scan-link';
      link.target = '_blank';
      link.rel = 'noopener';
      link.href = pageImageUrl(source.volume_number, source.page_number, source.derivative_id);
      link.textContent = `Volume ${source.volume_number} · ${source.publication_year} · page ${source.page_number} ↗`;
      citationList.append(link);
    });
    turn.append(citationList);
  }
  messages.forEach(message => {
    const warning = document.createElement('p');
    warning.className = 'chat-warning';
    warning.textContent = message;
    turn.append(warning);
  });
  chatTranscript.append(turn);
  turn.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

chatButton.addEventListener('click', () => {
  chatPanel.hidden = false;
  if (!chatQuestion.value && lastRequest) chatQuestion.value = lastRequest.query;
  chatPanel.scrollIntoView({behavior: 'smooth'});
  chatQuestion.focus();
});

document.querySelector('#clear-chat').addEventListener('click', () => {
  chatHistory = [];
  chatTranscript.replaceChildren();
  chatQuestion.value = '';
  chatQuestion.focus();
});

chatForm.addEventListener('submit', async event => {
  event.preventDefault();
  if (!lastRequest) return;
  const question = chatQuestion.value.trim();
  if (!question) return;
  const priorHistory = chatHistory.slice(-12);
  appendChatTurn('user', question);
  chatQuestion.value = '';
  chatSend.disabled = true;
  chatSend.textContent = 'Searching evidence…';
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...requestBody(), query: question, history: priorHistory}),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    const data = await response.json();
    const provenance = [
      data.status,
      data.model,
      data.prompt_sha256 ? `prompt ${data.prompt_sha256.slice(0, 12)}…` : null,
      data.context_sha256 ? `context ${data.context_sha256.slice(0, 12)}…` : null,
      data.total_tokens !== null ? `${data.total_tokens} tokens` : null,
      data.estimated_cost_usd !== null ? `$${data.estimated_cost_usd.toFixed(6)} estimated` : null,
    ].filter(Boolean).join(' · ');
    appendChatTurn(
      'assistant', data.output, data.citations,
      [...data.validation_errors, ...data.warnings], provenance,
    );
    chatHistory.push(
      {role: 'user', content: question},
      {role: 'assistant', content: data.output},
    );
    chatHistory = chatHistory.slice(-12);
  } catch (error) {
    appendChatTurn('assistant', `Chat unavailable: ${error.message}`);
  } finally {
    chatSend.disabled = false;
    chatSend.textContent = 'Ask';
    chatQuestion.focus();
  }
});

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
