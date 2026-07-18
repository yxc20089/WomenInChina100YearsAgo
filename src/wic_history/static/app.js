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
let lastRequest = null;

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
    card.querySelector('.scan-link').href = `/api/page-image/${hit.source.volume_number}/${hit.source.page_number}`;
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
