"""Serve a localhost visual-review UI for benchmark candidates."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse


ANNOTATION_SCHEMA_VERSION = "1.0"
ALLOWED_VALUES = {
    "page_genre": {
        "unreviewed",
        "news_editorial",
        "advertisement_classified",
        "mixed",
        "photograph_caption",
        "table_market_schedule",
        "front_matter_index",
        "blank_other",
    },
    "layout": {"unreviewed", "vertical", "horizontal", "mixed"},
    "scan_quality": {"unreviewed", "clean", "moderate", "poor", "unusable"},
    "women_relevance": {"unreviewed", "none", "possible", "explicit"},
    "gold_status": {"undecided", "include", "exclude"},
}


INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shen Bao benchmark review</title>
<style>
body{margin:0;font:14px system-ui;background:#eee;color:#222}
header{position:sticky;top:0;z-index:2;background:#17202a;color:white;padding:9px 14px;display:flex;gap:10px;align-items:center}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;height:calc(100vh - 46px)}
.image{overflow:auto;text-align:center;background:#555;padding:12px}.image img{max-width:100%;height:auto;background:white}
.panel{overflow:auto;padding:15px;background:white;border-left:1px solid #bbb}label{display:block;margin:10px 0 4px;font-weight:600}
select,textarea,input{width:100%;box-sizing:border-box;padding:7px}textarea{height:85px}.buttons{display:flex;gap:8px;margin-top:14px}
button{padding:7px 12px;cursor:pointer}.meta{font:12px ui-monospace,monospace;white-space:pre-wrap;background:#f4f4f4;padding:8px}
.error{padding:30px;color:#8b0000;font-size:18px}.saved{color:#8fda9b}.filter{margin-left:auto;width:auto;padding:4px}
</style></head><body>
<header><button id="prev">←</button><button id="next">→</button><span id="position"></span><span id="saveState"></span>
<select id="filter" class="filter"><option value="all">All</option><option value="unreviewed">Unreviewed</option><option value="include">Gold: include</option><option value="render_error">Render errors</option></select></header>
<div class="wrap"><main class="image" id="image"></main><aside class="panel"><div class="meta" id="meta"></div>
<label>Page genre</label><select id="page_genre"><option>unreviewed</option><option>news_editorial</option><option>advertisement_classified</option><option>mixed</option><option>photograph_caption</option><option>table_market_schedule</option><option>front_matter_index</option><option>blank_other</option></select>
<label>Layout</label><select id="layout"><option>unreviewed</option><option>vertical</option><option>horizontal</option><option>mixed</option></select>
<label>Scan quality</label><select id="scan_quality"><option>unreviewed</option><option>clean</option><option>moderate</option><option>poor</option><option>unusable</option></select>
<label>Women relevance</label><select id="women_relevance"><option>unreviewed</option><option>none</option><option>possible</option><option>explicit</option></select>
<label>Gold status</label><select id="gold_status"><option>undecided</option><option>include</option><option>exclude</option></select>
<label>Reviewer</label><input id="reviewer" placeholder="initials or name"><label>Notes</label><textarea id="notes"></textarea>
<div class="buttons"><button id="save">Save (S)</button><button id="saveNext">Save + next (Enter)</button></div></aside></div>
<script>
let samples=[],visible=[],index=0;
const fields=['page_genre','layout','scan_quality','women_relevance','gold_status','reviewer','notes'];
function applyFilter(){const f=document.getElementById('filter').value;return samples.filter(s=>f==='all'||(f==='unreviewed'&&!s.annotation)||(f==='include'&&s.annotation?.gold_status==='include')||(f==='render_error'&&s.status!=='rendered'));}
function show(){visible=applyFilter();if(!visible.length){document.getElementById('image').innerHTML='<div class="error">No matching samples</div>';return;}index=Math.max(0,Math.min(index,visible.length-1));const s=visible[index],a=s.annotation||{};document.getElementById('position').textContent=`${index+1}/${visible.length} · ${s.sample_id}`;document.getElementById('meta').textContent=JSON.stringify({year:s.publication_year,volume:s.volume_number,page:s.page_number,status:s.status,renderer:s.renderer,issue:s.issue},null,2);document.getElementById('image').innerHTML=s.status==='rendered'?`<img src="/image/${encodeURIComponent(s.sample_id)}" alt="${s.sample_id}">`:`<div class="error">${s.issue||s.status}</div>`;for(const f of fields){const e=document.getElementById(f);e.value=a[f]??(f==='gold_status'?'undecided':(f==='reviewer'||f==='notes'?'':'unreviewed'));}document.getElementById('saveState').textContent='';}
async function save(next){const s=visible[index],body={sample_id:s.sample_id};for(const f of fields)body[f]=document.getElementById(f).value;const r=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!r.ok){alert(await r.text());return;}const saved=await r.json();s.annotation=saved.annotation;samples.find(x=>x.sample_id===s.sample_id).annotation=saved.annotation;document.getElementById('saveState').textContent='saved';document.getElementById('saveState').className='saved';if(next){index=Math.min(index+1,applyFilter().length-1);show();}}
function move(d){index=Math.max(0,Math.min(index+d,visible.length-1));show();}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('save').onclick=()=>save(false);document.getElementById('saveNext').onclick=()=>save(true);document.getElementById('filter').onchange=()=>{index=0;show();};
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1);if(e.key.toLowerCase()==='s')save(false);if(e.key==='Enter')save(true);});
fetch('/api/samples').then(r=>r.json()).then(x=>{samples=x.samples;show();});
</script></body></html>"""


class AnnotationStore:
    """Small atomic JSON store suitable for a single local reviewer."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": ANNOTATION_SCHEMA_VERSION, "annotations": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, sample_id: str, values: dict[str, str]) -> dict[str, Any]:
        with self.lock:
            data = self.load()
            annotation: dict[str, Any] = {"sample_id": sample_id}
            for field, allowed in ALLOWED_VALUES.items():
                default = "undecided" if field == "gold_status" else "unreviewed"
                value = values.get(field, default)
                if value not in allowed:
                    raise ValueError(f"Invalid {field}: {value}")
                annotation[field] = value
            annotation["reviewer"] = str(values.get("reviewer", "")).strip()[:200]
            annotation["notes"] = str(values.get("notes", "")).strip()[:5000]
            annotation["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            data["schema_version"] = ANNOTATION_SCHEMA_VERSION
            data.setdefault("annotations", {})[sample_id] = annotation
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(self.path)
            return annotation


def read_render_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def make_handler(samples: list[dict[str, Any]], store: AnnotationStore):
    sample_by_id = {sample["sample_id"]: sample for sample in samples}

    class ReviewHandler(BaseHTTPRequestHandler):
        def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                body = INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/samples":
                annotations = store.load().get("annotations", {})
                merged = [
                    {**sample, "annotation": annotations.get(sample["sample_id"])}
                    for sample in samples
                ]
                self.send_json({"samples": merged})
                return
            if path.startswith("/image/"):
                sample_id = unquote(path.removeprefix("/image/"))
                sample = sample_by_id.get(sample_id)
                if not sample or sample.get("status") != "rendered":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                image_path = Path(sample["render_path"]).resolve()
                if not image_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                body = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    mimetypes.guess_type(image_path.name)[0] or "application/octet-stream",
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/annotation":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 20_000:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(length))
                sample_id = str(payload.get("sample_id", ""))
                if sample_id not in sample_by_id:
                    raise ValueError("Unknown sample_id")
                annotation = store.update(sample_id, payload)
                self.send_json({"annotation": annotation})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ReviewHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/benchmark-pages/render_manifest.jsonl"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("artifacts/benchmark-review/annotations.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = read_render_manifest(args.manifest)
    store = AnnotationStore(args.annotations)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(samples, store))
    print(f"Review {len(samples)} candidates at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
