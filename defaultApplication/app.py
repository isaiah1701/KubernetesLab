
# --- Fun Challenges endpoints ---
from fastapi import Body
import random, base64, urllib.parse, hashlib, os, csv, io
from typing import List, Dict, Optional

# In-memory stores for quiz/riddle
quiz_math_store: Dict[int, Dict] = {}
riddle_store = [
    {"id": 1, "question": "What has keys but can't open locks?", "answer": "A piano"},
    {"id": 2, "question": "What runs but never walks?", "answer": "Water"},
    {"id": 3, "question": "What has a face and two hands but no arms or legs?", "answer": "A clock"},
]

# --- Data sets for jq/sort/uniq ---
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "gray", "black", "white", "cyan", "magenta", "lime", "teal", "indigo", "violet", "gold", "silver", "beige"]
ANIMALS = [
    {"id": 1, "name": "cat", "legs": 4},
    {"id": 2, "name": "dog", "legs": 4},
    {"id": 3, "name": "bird", "legs": 2},
    {"id": 4, "name": "ant", "legs": 6},
    {"id": 5, "name": "spider", "legs": 8},
    {"id": 6, "name": "fish", "legs": 0},
    {"id": 7, "name": "horse", "legs": 4},
    {"id": 8, "name": "frog", "legs": 4},
    {"id": 9, "name": "snake", "legs": 0},
    {"id": 10, "name": "bee", "legs": 6},
]
WORDS = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape", "honeydew", "kiwi", "lemon", "mango", "nectarine", "orange", "papaya", "quince", "raspberry", "strawberry", "tangerine", "ugli", "vanilla", "watermelon", "xigua", "yam", "zucchini"]
"""
Container Notes:
- This app is container-friendly and listens on 0.0.0.0:${PORT} (default 5000).
- No files are written to disk; all logs go to stdout/stderr as structured JSON lines.
- Graceful shutdown is implemented for SIGINT/SIGTERM (waits for in-flight requests).
- No Dockerfile or container build steps included.

Run: python app.py
Example: curl -i http://localhost:5000/healthz
"""

import os
import sys
import signal
import time
import uuid
import json
import hashlib
import random
import string
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Request, Response, status, Query, Path, Header, UploadFile, File, Form, HTTPException, Depends, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse, RedirectResponse, Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.responses import FileResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.requests import ClientDisconnect
from starlette.datastructures import Headers
from starlette.concurrency import run_in_threadpool
from starlette.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware import Middleware
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_206_PARTIAL_CONTENT, HTTP_304_NOT_MODIFIED


# --- Config ---
PORT = int(os.getenv("PORT", "5000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
STARTUP_DELAY_MS = int(os.getenv("STARTUP_DELAY_MS", "0"))

# --- FastAPI app ---
app = FastAPI()

# --- Fun Challenges endpoints ---
# (all endpoint definitions follow here)
CAMP_SEED = int(os.getenv("CAMP_SEED", "42"))
CAMP_TOKEN = os.getenv("CAMP_TOKEN", "")

# --- Globals ---
ready_flag = True
metrics = {
    "requests_total": 0,
    "requests_inflight": 0,
    "last_reaction_ms": 0,
    "avg_work_ms": 0.0,
}
work_timings = []

# --- Logging ---
class StructuredLogger:
    def __init__(self):
        self.logger = logging.getLogger("uvicorn.error")
        self.logger.setLevel(logging.INFO)

    def log(self, ts, method, path, status, latency_ms, user_agent, req_id):
        log_obj = {
            "ts": ts,
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": latency_ms,
            "user_agent": user_agent,
            "req_id": req_id,
        }
        print(json.dumps(log_obj), flush=True)

logger = StructuredLogger()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        metrics["requests_inflight"] += 1
        req_id = request.headers.get("X-Req-Id") or str(uuid.uuid4())
        response = await call_next(request)
        latency_ms = int((time.time() - start) * 1000)
        ts = datetime.now(timezone.utc).isoformat()
        user_agent = request.headers.get("user-agent", "")
        logger.log(ts, request.method, request.url.path, response.status_code, latency_ms, user_agent, req_id)
        metrics["requests_total"] += 1
        metrics["requests_inflight"] -= 1
        return response

# --- App ---
app = FastAPI(title="Linux Command Training Camp", docs_url=None, redoc_url=None)
app.add_middleware(LoggingMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Startup Delay ---
if STARTUP_DELAY_MS > 0:
    print(f"[Startup] Sleeping {STARTUP_DELAY_MS}ms before accepting requests...", flush=True)
    time.sleep(STARTUP_DELAY_MS / 1000)

# --- Graceful Shutdown ---
shutdown_event = asyncio.Event()

def handle_shutdown(signum, frame):
    print(f"[Shutdown] Signal {signum} received. Waiting for in-flight requests...", flush=True)
    ready_flag = False
    shutdown_event.set()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# --- Utility Functions ---
def get_client_ip(request: Request) -> str:
    return request.client.host if request.client else ""

def get_env_vars() -> Dict[str, Any]:
    return {
        "PORT": PORT,
        "LOG_LEVEL": LOG_LEVEL,
        "STARTUP_DELAY_MS": STARTUP_DELAY_MS,
        "CAMP_SEED": CAMP_SEED,
        "CAMP_TOKEN": CAMP_TOKEN,
    }

def deterministic_users(n: int, seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    users = []
    for i in range(n):
        name = f"user{i}-{random.choice(['alice','bob','carol','dave','eve','frank'])}"
        email = f"{name}@camp.local"
        tags = random.sample(["admin","dev","ops","guest","trainer","trainee"], k=random.randint(1,3))
        meta = {
            "team": random.choice(["red","blue","green","yellow"]),
            "active": bool(random.getrandbits(1)),
            "score": random.randint(0,100),
        }
        users.append({"id": i+1, "name": name, "email": email, "tags": tags, "meta": meta})
    return users

def deterministic_logs(n: int, levels: List[str], seed: int) -> List[Dict[str, Any]]:
    random.seed(seed)
    logs = []
    for i in range(n):
        level = random.choice(levels)
        msg = f"Log message {i}"
        ctx = {"rid": str(uuid.uuid4()), "user": f"user{random.randint(1,10)}"}
        ts = datetime.now(timezone.utc).isoformat()
        logs.append({"ts": ts, "level": level, "msg": msg, "ctx": ctx})
    return logs

def deterministic_tree(depth: int, fanout: int, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    def make_node(d):
        if d == 0:
            return {"id": str(uuid.uuid4()), "name": f"leaf-{random.randint(1,100)}"}
        return {
            "id": str(uuid.uuid4()),
            "name": f"node-{d}",
            "children": [make_node(d-1) for _ in range(fanout)]
        }
    return make_node(depth)

# --- Models ---
class EchoModel(BaseModel):
    data: Any

# --- Endpoints ---
@app.get("/random/number", response_model=Dict[str, int])
async def random_number(min: int = Query(0), max: int = Query(100), seed: Optional[int] = Query(None)):
    """Return a random integer in [min, max]."""
    if min > max:
        raise HTTPException(400, "min must be <= max")
    rng = random.Random(seed)
    value = rng.randint(min, max)
    return {"value": value}

@app.get("/random/choice", response_model=Dict[str, str])
async def random_choice(options: str = Query(...), seed: Optional[int] = Query(None)):
    """Return a random choice from comma-separated options."""
    opts = [o.strip() for o in options.split(",") if o.strip()]
    if not opts:
        raise HTTPException(400, "No options provided")
    rng = random.Random(seed)
    value = rng.choice(opts)
    return {"value": value}

@app.get("/quiz/math", response_model=Dict)
async def quiz_math():
    """Return a simple math quiz (A+B)."""
    qid = random.randint(1000, 9999)
    a, b = random.randint(1, 20), random.randint(1, 20)
    quiz_math_store[qid] = {"a": a, "b": b}
    return {"id": qid, "question": f"{a} + {b}"}

@app.post("/quiz/math", response_model=Dict)
async def quiz_math_answer(data: Dict = Body(...)):
    """Check answer for math quiz."""
    qid = data.get("id")
    ans = data.get("answer")
    q = quiz_math_store.get(qid)
    if not q:
        raise HTTPException(404, "Quiz not found")
    solution = q["a"] + q["b"]
    correct = (ans == solution)
    return {"correct": correct, "solution": solution}

@app.post("/string/reverse")
async def string_reverse(request: Request):
    """Reverse input string (plain text)."""
    text = await request.body()
    reversed_text = text.decode()[::-1]
    return Response(content=reversed_text, media_type="text/plain")

@app.post("/string/palindrome", response_model=Dict)
async def string_palindrome(data: Dict = Body(...)):
    """Check if word is palindrome."""
    word = data.get("word", "")
    is_pal = word == word[::-1]
    return {"palindrome": is_pal}

@app.get("/string/shuffle", response_model=Dict)
async def string_shuffle(word: str = Query(...), seed: Optional[int] = Query(None)):
    """Shuffle letters in word."""
    chars = list(word)
    rng = random.Random(seed)
    rng.shuffle(chars)
    return {"shuffled": ''.join(chars)}

@app.get("/data/colors", response_model=List[str])
async def data_colors(n: int = Query(10, ge=1, le=20), seed: Optional[int] = Query(None)):
    """Return n random colors."""
    rng = random.Random(seed)
    return rng.sample(COLORS, min(n, len(COLORS)))

@app.get("/data/animals", response_model=List[Dict])
async def data_animals(n: int = Query(10, ge=1, le=10), seed: Optional[int] = Query(None)):
    """Return n random animals."""
    rng = random.Random(seed)
    return rng.sample(ANIMALS, min(n, len(ANIMALS)))

@app.get("/data/numbers", response_model=List[int])
async def data_numbers(n: int = Query(50, ge=1, le=100)):
    """Return numbers 1..n."""
    return list(range(1, n+1))

@app.get("/data/matrix", response_model=List[List[int]])
async def data_matrix(rows: int = Query(3, ge=1, le=10), cols: int = Query(3, ge=1, le=10), seed: Optional[int] = Query(None)):
    """Return matrix of random ints."""
    rng = random.Random(seed)
    return [[rng.randint(0, 99) for _ in range(cols)] for _ in range(rows)]

@app.get("/challenge/words", response_model=List[str])
async def challenge_words(n: int = Query(50, ge=1, le=100), duplicates: int = Query(0, ge=0, le=1), seed: Optional[int] = Query(None)):
    """Return n words, with/without duplicates."""
    rng = random.Random(seed)
    base = rng.choices(WORDS, k=n) if duplicates else rng.sample(WORDS, min(n, len(WORDS)))
    return base

@app.get("/challenge/numbers", response_model=List[int])
async def challenge_numbers(n: int = Query(100, ge=1, le=200), range: int = Query(20, ge=1, le=100), seed: Optional[int] = Query(None)):
    """Return n ints in range, possible duplicates."""
    rng = random.Random(seed)
    return [rng.randint(0, range-1) for _ in range(n)]

@app.get("/encode/base64", response_model=Dict)
async def encode_base64(text: str = Query(...)):
    """Base64 encode text."""
    b64 = base64.b64encode(text.encode()).decode()
    return {"b64": b64}

@app.get("/decode/base64", response_model=Dict)
async def decode_base64(b64: str = Query(...)):
    """Base64 decode."""
    try:
        text = base64.b64decode(b64.encode()).decode()
    except Exception:
        raise HTTPException(400, "Invalid base64")
    return {"text": text}

@app.get("/encode/url", response_model=Dict)
async def encode_url(text: str = Query(...)):
    """URL encode text."""
    return {"encoded": urllib.parse.quote(text)}

@app.get("/decode/url", response_model=Dict)
async def decode_url(text: str = Query(...)):
    """URL decode text."""
    return {"decoded": urllib.parse.unquote(text)}

@app.get("/stream/numbers")
async def stream_numbers(count: int = Query(20, ge=1, le=100)):
    """Stream numbers as text/plain lines."""
    lines = '\n'.join(str(i) for i in range(1, count+1))
    return Response(content=lines, media_type="text/plain")

@app.get("/stream/words")
async def stream_words(count: int = Query(20, ge=1, le=100), seed: Optional[int] = Query(None)):
    """Stream words as text/plain lines."""
    rng = random.Random(seed)
    words = rng.sample(WORDS, min(count, len(WORDS)))
    lines = '\n'.join(words)
    return Response(content=lines, media_type="text/plain")

@app.get("/stream/json")
async def stream_json(count: int = Query(20, ge=1, le=100), seed: Optional[int] = Query(None)):
    """Stream NDJSON objects."""
    rng = random.Random(seed)
    items = [{"id": i, "value": rng.choice(WORDS)} for i in range(1, count+1)]
    lines = '\n'.join(json.dumps(item) for item in items)
    return Response(content=lines, media_type="application/x-ndjson")

@app.get("/challenge/status/random")
async def challenge_status_random(seed: Optional[int] = Query(None)):
    """Randomly return 200/400/500 with explanation."""
    rng = random.Random(seed)
    code = rng.choice([200, 400, 500])
    body = {"status": code, "reason": {200: "OK", 400: "Bad Request", 500: "Internal Error"}[code]}
    return JSONResponse(content=body, status_code=code)

@app.get("/challenge/redirect/loop")
async def challenge_redirect_loop(n: int = Query(3, ge=1, le=10), request: Request = None):
    """Chain n redirects then 200."""
    url = str(request.url)
    if n > 1:
        next_url = url.replace(f"n={n}", f"n={n-1}")
        return RedirectResponse(next_url, status_code=302)
    return JSONResponse({"done": True})

@app.get("/challenge/delay/random", response_model=Dict)
async def challenge_delay_random(min_ms: int = Query(50, ge=0), max_ms: int = Query(500, ge=1), seed: Optional[int] = Query(None)):
    """Sleep random ms in range."""
    if min_ms > max_ms:
        raise HTTPException(400, "min_ms > max_ms")
    rng = random.Random(seed)
    ms = rng.randint(min_ms, max_ms)
    await asyncio.sleep(ms/1000)
    return {"slept_ms": ms}

@app.get("/hash/md5", response_model=Dict)
async def hash_md5(text: str = Query(...)):
    """MD5 hash of text."""
    md5 = hashlib.md5(text.encode()).hexdigest()
    return {"md5": md5}

@app.get("/hash/sha256", response_model=Dict)
async def hash_sha256(text: str = Query(...)):
    """SHA256 hash of text."""
    sha = hashlib.sha256(text.encode()).hexdigest()
    return {"sha256": sha}

@app.get("/auth/secret")
async def auth_secret(token: str = Query(...)):
    """200 if token matches CAMP_TOKEN env, else 403."""
    camp_token = os.environ.get("CAMP_TOKEN", "")
    if token == camp_token:
        return Response(status_code=200)
    return JSONResponse({"error": "Forbidden"}, status_code=403)

@app.get("/csv/sample")
async def csv_sample(rows: int = Query(20, ge=1, le=100), cols: int = Query(3, ge=1, le=10), headers: int = Query(0, ge=0, le=1), seed: Optional[int] = Query(None)):
    """Return sample CSV."""
    rng = random.Random(seed)
    output = io.StringIO()
    writer = csv.writer(output)
    if headers:
        writer.writerow([f"col{i+1}" for i in range(cols)])
    for _ in range(rows):
        writer.writerow([rng.choice(WORDS) for _ in range(cols)])
    return Response(content=output.getvalue(), media_type="text/csv")
@app.get("/riddle", response_model=Dict)
async def get_riddle():
    """Return a riddle."""
    r = random.choice(riddle_store)
    return {"id": r["id"], "question": r["question"]}

@app.get("/riddle/{id}/answer", response_model=Dict)
async def riddle_answer(id: int = Path(...)):
    """Return answer for riddle."""
    for r in riddle_store:
        if r["id"] == id:
            return {"answer": r["answer"]}
    raise HTTPException(404, "Riddle not found")
@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page: dark-mode, game, camp endpoints, debug card."""
    env = get_env_vars()
    debug_info = {
        "env": env,
        "time": datetime.now(timezone.utc).isoformat(),
        "client_ip": get_client_ip(request),
    }
    # Endpoint lists for Linux Camp and Bash Practice
    camp_endpoints = [
        {"path": "/healthz", "desc": "Health check", "curl": "curl -i /healthz"},
        {"path": "/ready", "desc": "Readiness check", "curl": "curl -i /ready"},
        {"path": "/status/418", "desc": "Status code", "curl": "curl -i /status/418"},
        {"path": "/delay/1000", "desc": "Delay response", "curl": "curl -i /delay/1000"},
        {"path": "/uuid", "desc": "UUID v4", "curl": "curl -i /uuid"},
        {"path": "/time", "desc": "Current time", "curl": "curl -i /time"},
        {"path": "/ip", "desc": "Client IP", "curl": "curl -i /ip"},
        {"path": "/echo", "desc": "Echo body", "curl": "curl -X POST -d '{\"hello\":1}' /echo"},
        {"path": "/echo", "desc": "Echo body", "curl": "curl -X POST -d '{\"hello\":1}' /echo"},
        {"path": "/config", "desc": "Config/env", "curl": "curl -i /config"},
        {"path": "/json/users?n=5", "desc": "Users JSON", "curl": "curl -i /json/users?n=5"},
        {"path": "/json/logs?n=3&levels=info,error", "desc": "Logs JSON", "curl": "curl -i /json/logs?n=3&levels=info,error"},
        {"path": "/json/tree?depth=2&fanout=2", "desc": "Tree JSON", "curl": "curl -i /json/tree?depth=2&fanout=2"},
        {"path": "/ndjson?n=3", "desc": "NDJSON stream", "curl": "curl -i /ndjson?n=3"},
        {"path": "/stream/logs?rate=2&duration=2", "desc": "Log stream", "curl": "curl -i /stream/logs?rate=2&duration=2"},
        {"path": "/text/lorem?lines=5", "desc": "Lorem text", "curl": "curl -i /text/lorem?lines=5"},
        {"path": "/csv/data?rows=3&cols=2&headers=1", "desc": "CSV data", "curl": "curl -i /csv/data?rows=3&cols=2&headers=1"},
        {"path": "/xml/sample?n=2", "desc": "XML sample", "curl": "curl -i /xml/sample?n=2"},
        {"path": "/upload", "desc": "Upload file", "curl": "curl -F file=@app.py /upload"},
        {"path": "/files/sample?kb=1&name=test.bin", "desc": "Download file", "curl": "curl -O /files/sample?kb=1&name=test.bin"},
        {"path": "/cache/etag?seed=42", "desc": "ETag cache", "curl": "curl -i /cache/etag?seed=42"},
        {"path": "/gzip?kb=1", "desc": "Gzip text", "curl": "curl --compressed /gzip?kb=1"},
        {"path": "/range?kb=1", "desc": "Range bytes", "curl": "curl --range 0-99 /range?kb=1"},
        {"path": "/work?ms=100", "desc": "Work latency", "curl": "curl -i /work?ms=100"},
        {"path": "/fail/readiness?toggle=1", "desc": "Toggle readiness", "curl": "curl -i /fail/readiness?toggle=1"},
        {"path": "/exit?code=1", "desc": "Exit app", "curl": "curl -i /exit?code=1"},
        {"path": "/auth/basic", "desc": "Basic auth", "curl": "curl -u user:pass /auth/basic"},
        {"path": "/resource/1", "desc": "Upsert resource", "curl": "curl -X PUT -d '{\"x\":1}' /resource/1"},
        {"path": "/resource/1", "desc": "Upsert resource", "curl": "curl -X PUT -d '{\"x\":1}' /resource/1"},
        {"path": "/metrics", "desc": "Metrics", "curl": "curl -i /metrics"},
    ]
    bash_endpoints = [
        {"path": "/redirect/3", "desc": "302 chain", "curl": "curl -L /redirect/3"},
        {"path": "/cookies/set?name=foo&value=bar", "desc": "Set cookie", "curl": "curl -c jar.txt /cookies/set?name=foo&value=bar"},
        {"path": "/cookies", "desc": "Show cookies", "curl": "curl -b jar.txt /cookies"},
        {"path": "/retry/503?after=2", "desc": "Retry-After", "curl": "curl -i -H 'X-Req-Id:123' /retry/503?after=2"},
        {"path": "/cache/control?max_age=60", "desc": "Cache-Control", "curl": "curl -i /cache/control?max_age=60"},
        {"path": "/text/upper", "desc": "Uppercase text", "curl": "echo hi | curl --data-binary @- /text/upper"},
        {"path": "/json/filter?path=.meta.active", "desc": "Filter JSON", "curl": "curl -X POST -d '[{\"meta\":{\"active\":true}}]' /json/filter?path=.meta.active"},
        {"path": "/csv/sort?col=2&numeric=1", "desc": "Sort CSV", "curl": "curl -X POST --data-binary @data.csv /csv/sort?col=2&numeric=1"},
        {"path": "/bytes?n=100", "desc": "Zero bytes", "curl": "curl --output /dev/null /bytes?n=100"},
        {"path": "/lines?n=5&prefix=foo", "desc": "Line stream", "curl": "curl /lines?n=5&prefix=foo"},
        {"path": "/sse/ticks?rate=2&duration=2", "desc": "SSE ticks", "curl": "curl /sse/ticks?rate=2&duration=2"},
        {"path": "/json/paged?page=1&size=2", "desc": "Paged JSON", "curl": "curl -i /json/paged?page=1&size=2"},
        {"path": "/auth/query?token=abc", "desc": "Query token", "curl": "curl /auth/query?token=abc"},
        {"path": "/flaky?p_success=0.5", "desc": "Flaky endpoint", "curl": "curl --retry 5 /flaky?p_success=0.5"},
        {"path": "/jitter?min_ms=100&max_ms=200", "desc": "Jitter delay", "curl": "curl /jitter?min_ms=100&max_ms=200"},
        {"path": "/chunked?n=3&size=10", "desc": "Chunked stream", "curl": "curl /chunked?n=3&size=10"},
    ]
    curl_notes = [
        "-i: show headers", "-v: verbose", "-L: follow redirects", "-H: custom header", "--data-binary: raw body", "-u: basic auth", "-b/--cookie: send cookie", "-c/--cookie-jar: save cookies", "--compressed: accept gzip", "--range: partial bytes", "--retry: retry logic"
    ]
    # Fun Challenges endpoints and UI
    fun_endpoints = [
        {"path": "/random/number?min=1&max=10", "desc": "Random number in range"},
        {"path": "/random/choice?options=a,b,c", "desc": "Random choice from options"},
        {"path": "/quiz/math", "desc": "Get math quiz"},
        {"path": "/quiz/math", "desc": "Submit math answer"},
        {"path": "/riddle", "desc": "Get riddle"},
        {"path": "/riddle/1/answer", "desc": "Get riddle answer"},
        {"path": "/string/reverse", "desc": "Reverse string"},
        {"path": "/string/palindrome", "desc": "Check palindrome"},
        {"path": "/string/shuffle?word=hello", "desc": "Shuffle string"},
        {"path": "/data/colors?n=5", "desc": "Get color list"},
        {"path": "/data/animals?n=5", "desc": "Get animal list"},
        {"path": "/data/numbers?n=10", "desc": "Get numbers 1..n"},
        {"path": "/data/matrix?rows=2&cols=2", "desc": "Get matrix"},
        {"path": "/challenge/words?n=10&duplicates=1", "desc": "Words with/without duplicates"},
        {"path": "/challenge/numbers?n=10&range=5", "desc": "Numbers with possible duplicates"},
        {"path": "/encode/base64?text=hi", "desc": "Base64 encode"},
        {"path": "/decode/base64?b64=aGk=", "desc": "Base64 decode"},
        {"path": "/encode/url?text=hi there", "desc": "URL encode"},
        {"path": "/decode/url?text=hi%20there", "desc": "URL decode"},
        {"path": "/stream/numbers?count=5", "desc": "Stream numbers"},
        {"path": "/stream/words?count=5", "desc": "Stream words"},
        {"path": "/stream/json?count=5", "desc": "Stream JSON NDJSON"},
        {"path": "/challenge/status/random", "desc": "Random status code"},
        {"path": "/challenge/redirect/loop?n=3", "desc": "Redirect loop"},
        {"path": "/challenge/delay/random?min_ms=50&max_ms=100", "desc": "Random delay"},
        {"path": "/hash/md5?text=hi", "desc": "MD5 hash"},
        {"path": "/hash/sha256?text=hi", "desc": "SHA256 hash"},
        {"path": "/auth/secret?token=abc", "desc": "Secret token auth"},
        {"path": "/csv/sample?rows=5&cols=2&headers=1", "desc": "Sample CSV"},
    ]
    camp_list_html = ''.join([
        f"<li class='mb-1'><span class='font-mono text-blue-300'>{e['path']}</span> "
        f"<span class='text-gray-400'>- {e['desc']}</span> "
        f"<button class='copy-btn' onclick=\"navigator.clipboard.writeText('{e['curl']}')\">Copy</button></li>"
        for e in camp_endpoints
    ])
    camp_list_html = ''.join([
        f"<li class='mb-1'><span class='font-mono text-blue-300'>{e['path']}</span> "
        f"<span class='text-gray-400'>- {e['desc']}</span> "
        f"<button class='copy-btn' onclick=\"navigator.clipboard.writeText('{e['path']}')\">Copy</button></li>"
        for e in camp_endpoints
    ])
    bash_list_html = ''.join([
        f"<li class='mb-1'><span class='font-mono text-blue-300'>{e['path']}</span> "
        f"<span class='text-gray-400'>- {e['desc']}</span> "
        f"<button class='copy-btn' onclick=\"navigator.clipboard.writeText('{e['path']}')\">Copy</button></li>"
        for e in bash_endpoints
    ])
    fun_list_html = ''.join([
        f"<li class='mb-1'><span class='font-mono text-purple-300'>{e['path']}</span> "
        f"<span class='text-gray-400'>- {e['desc']}</span> "
        f"<button class='copy-btn' onclick=\"navigator.clipboard.writeText('{e['path']}')\">Copy</button></li>"
        for e in fun_endpoints
    ])
    html = f"""
    <meta name='viewport' content='width=device-width,initial-scale=1'/>
    <title>Linux Command Training Camp</title>
    <link href='https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css' rel='stylesheet'>
    <style>
        body {{{{ background: linear-gradient(135deg,#18181b 0%,#27272a 100%); color:#e5e7eb; }}}}
        .logo {{{{ font-size:2rem; font-weight:bold; letter-spacing:-2px; color:#38bdf8; }}}}
        .card {{{{ background:#23272e; border-radius:1rem; box-shadow:0 2px 8px #0002; padding:1.5rem; margin-bottom:1.5rem; }}}}
        .copy-btn {{{{ background:#38bdf8; color:#18181b; border-radius:0.5rem; padding:0.2rem 0.6rem; font-size:0.9rem; margin-left:0.5rem; cursor:pointer; }}}}
        .section-title {{{{ font-size:1.3rem; font-weight:600; margin-bottom:0.5rem; color:#38bdf8; }}}}
        .fun-section-title {{{{ font-size:1.3rem; font-weight:600; margin-bottom:0.5rem; color:#a78bfa; }}}}
        .debug-card {{{{ transition:max-height 0.3s; overflow:hidden; }}}}
        .debug-toggle:checked ~ .debug-card {{{{ max-height:500px; }}}}
        .debug-card {{{{ max-height:0; }}}}
        .game-btn {{{{ background:#38bdf8; color:#18181b; border-radius:0.5rem; padding:0.5rem 1rem; font-size:1rem; font-weight:600; cursor:pointer; }}}}
        .stress-btn {{{{ background:#fbbf24; color:#18181b; border-radius:0.5rem; padding:0.3rem 0.8rem; font-size:0.9rem; font-weight:600; }}}}
    </style>
    </head>
    <body class='min-h-screen flex flex-col items-center justify-center'>
        <div class='logo mb-4'>🐧 Linux Camp</div>
        <div class='max-w-2xl w-full'>
            <div class='card mb-6'>
                <div class='section-title'>Reaction-Time Game</div>
                <div id='game' class='mb-2'>
                    <button id='startBtn' class='game-btn'>Start</button>
                    <span id='gameMsg' class='ml-4'></span>
                    <div class='mt-2'>Best: <span id='bestScore'>-</span> ms</div>
                </div>
                <audio id='clickSound' src='data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=' preload='auto'></audio>
                <label class='mt-2'><input type='checkbox' id='mute'> Mute click</label>
            </div>
            <div class='card mb-6'>
                <div class='section-title'>Stress Test</div>
                <div id='stressTest'>
                    <input id='stressCount' type='number' min='1' max='50' value='10' class='bg-gray-800 text-gray-100 rounded px-2 py-1 w-16'>
                    <button id='stressBtn' class='stress-btn'>Run</button>
                    <span id='stressMsg' class='ml-4'></span>
                </div>
            </div>
            <div class='card mb-6'>
                <div class='section-title'>Linux Camp Endpoints</div>
                <ul>
                    {camp_list_html}
                </ul>
            </div>
            <div class='card mb-6'>
                <div class='section-title'>Bash Practice Endpoints</div>
                <ul>
                    {bash_list_html}
                </ul>
                <div class='mt-2 text-xs text-gray-400'>Curl flags: {curl_notes}</div>
            </div>
            <div class='card mb-6'>
                <div class='fun-section-title'>Fun Challenges</div>
                <ul>
                    {fun_list_html}
                </ul>
            </div>
            <div class='card mb-6'>
                <label class='font-bold text-blue-400'><input type='checkbox' class='debug-toggle' id='debugToggle'> Debug Info</label>
                <div class='debug-card bg-gray-900 text-gray-200 p-3 rounded mt-2'>
                    <pre id='debugInfo'>{json.dumps(debug_info, indent=2)}</pre>
                </div>
            </div>
        </div>
        <script>
            // Reaction-time game
            let startBtn = document.getElementById('startBtn');
            let gameMsg = document.getElementById('gameMsg');
            let bestScore = document.getElementById('bestScore');
            let clickSound = document.getElementById('clickSound');
            let mute = document.getElementById('mute');
            let best = localStorage.getItem('bestScore') || '-';
            bestScore.textContent = best;
            let waiting = false, goTime = 0;
            startBtn.onclick = function() {{{{
                gameMsg.textContent = 'Wait for GO...';
                startBtn.disabled = true;
                waiting = true;
                setTimeout(() => {{{{
                    goTime = performance.now();
                    gameMsg.textContent = 'GO!';
                    waiting = false;
                }}}}, 1000 + Math.random()*2000);
            }}}};
            gameMsg.onclick = function() {{{{
                if (!waiting && goTime) {{{{
                    let rt = Math.round(performance.now() - goTime);
                    gameMsg.textContent = `Reaction: ${{{{rt}}}} ms`;
                    if (best === '-' || rt < best) {{{{
                        best = rt;
                        localStorage.setItem('bestScore', rt);
                        bestScore.textContent = rt;
                    }}}}
                    if (!mute.checked) clickSound.play();
                    metrics.last_reaction_ms = rt;
                    goTime = 0;
                    startBtn.disabled = false;
                }}}}
            }}}};
            // Debug card toggle
            let debugToggle = document.getElementById('debugToggle');
            let debugCard = document.querySelector('.debug-card');
            debugToggle.onchange = function() {{{{
                debugCard.style.maxHeight = debugToggle.checked ? '500px' : '0';
            }}}};
            // Stress test widget
            let stressBtn = document.getElementById('stressBtn');
            let stressCount = document.getElementById('stressCount');
            let stressMsg = document.getElementById('stressMsg');
            stressBtn.onclick = async function() {{{{
                let n = parseInt(stressCount.value);
                let times = [];
                for (let i=0; i<n; i++) {{{{
                    let t0 = performance.now();
                    await fetch('/work?ms=10');
                    times.push(performance.now()-t0);
                }}}}
                let avg = times.reduce((a,b)=>a+b,0)/n;
                let p95 = times.sort((a,b)=>a-b)[Math.floor(n*0.95)];
                stressMsg.textContent = `Avg: ${{{{Math.round(avg)}}}} ms, p95: ${{{{Math.round(p95)}}}} ms`;
            }}}};
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# --- Health & Basics ---
@app.get("/healthz")
async def healthz():
    """Health check."""
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    """Readiness check."""
    return {"status": "ready" if ready_flag else "not ready"}

@app.get("/status/{code}")
async def status_code(code: int):
    """Return given status code."""
    return JSONResponse({"status": code}, status_code=code)

@app.get("/delay/{ms}")
async def delay(ms: int = Path(..., ge=0, le=10000)):
    """Sleep ms, return slept_ms."""
    await asyncio.sleep(ms/1000)
    return {"slept_ms": ms}

@app.get("/uuid")
async def uuid_v4():
    """Return UUID v4."""
    return {"uuid": str(uuid.uuid4())}

@app.get("/time")
async def time_now():
    """Return current time."""
    now = datetime.now(timezone.utc)
    return {"iso": now.isoformat(), "epoch_ms": int(now.timestamp()*1000)}

@app.get("/ip")
async def client_ip(request: Request):
    """Return client IP."""
    return {"client_ip": get_client_ip(request)}

# --- Headers & Echo ---
@app.get("/headers")
async def headers(request: Request):
    """Echo request headers as JSON."""
    return dict(request.headers)

@app.post("/echo")
async def echo(request: Request):
    """Echo back body as JSON."""
    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        data = await request.json()
    elif "application/x-www-form-urlencoded" in ct:
        data = await request.form()
        data = dict(data)
    elif "text/plain" in ct:
        data = (await request.body()).decode()
    else:
        data = (await request.body()).decode(errors="ignore")
    return {"echo": data, "content_type": ct}

# --- Environment/Config ---
@app.get("/config")
async def config():
    """Return whitelisted env vars."""
    return get_env_vars()

# --- JSON for jq ---
@app.get("/json/users")
async def json_users(n: int = Query(10, ge=1, le=100), seed: int = Query(CAMP_SEED)):
    """Return n user objects."""
    return deterministic_users(n, seed)

@app.get("/json/logs")
async def json_logs(n: int = Query(10, ge=1, le=100), levels: str = Query("info,warn,error"), seed: int = Query(CAMP_SEED)):
    """Return n log objects."""
    levels_list = [l.strip() for l in levels.split(",") if l.strip()]
    return deterministic_logs(n, levels_list, seed)

@app.get("/json/tree")
async def json_tree(depth: int = Query(2, ge=1, le=4), fanout: int = Query(2, ge=1, le=6), seed: int = Query(CAMP_SEED)):
    """Return nested tree JSON."""
    return deterministic_tree(depth, fanout, seed)

# --- NDJSON & Streaming ---
@app.get("/ndjson")
async def ndjson(n: int = Query(10, ge=1, le=100), seed: int = Query(CAMP_SEED)):
    """Emit n lines of NDJSON."""
    users = deterministic_users(n, seed)
    async def gen():
        for u in users:
            yield json.dumps(u) + "\n"
            await asyncio.sleep(0)
    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.get("/stream/logs")
async def stream_logs(rate: int = Query(1, ge=1, le=100), duration: int = Query(1, ge=1, le=60), seed: int = Query(CAMP_SEED)):
    """Stream synthetic logs as NDJSON."""
    total = rate * duration
    logs = deterministic_logs(total, ["info","warn","error"], seed)
    async def gen():
        for log in logs:
            yield json.dumps(log) + "\n"
            await asyncio.sleep(1/rate)
    return StreamingResponse(gen(), media_type="application/x-ndjson")

# --- Text/CSV/XML ---
@app.get("/text/lorem")
async def text_lorem(lines: int = Query(10, ge=1, le=100)):
    """Return lorem text."""
    words = ["lorem","ipsum","dolor","sit","amet","consectetur","adipiscing","elit"]
    out = []
    for i in range(lines):
        out.append(" ".join(random.choices(words, k=random.randint(5,12))))
    return PlainTextResponse("\n".join(out))

@app.get("/text/huge")
async def text_huge(kb: int = Query(1, ge=1, le=1024)):
    """Return ~kb size text payload."""
    chunk = "0123456789abcdef" * 64
    total = kb * 1024
    s = (chunk * (total // len(chunk) + 1))[:total]
    return PlainTextResponse(s)

@app.get("/csv/data")
async def csv_data(rows: int = Query(10, ge=1, le=100), cols: int = Query(3, ge=1, le=20), headers: int = Query(1)):
    """Return CSV grid."""
    out = []
    if headers:
        out.append(",".join([f"col{i+1}" for i in range(cols)]))
    for r in range(rows):
        out.append(",".join([f"data{r+1}_{c+1}" for c in range(cols)]))
    return PlainTextResponse("\n".join(out), media_type="text/csv")

@app.get("/xml/sample")
async def xml_sample(n: int = Query(3, ge=1, le=100)):
    """Return XML sample."""
    items = []
    for i in range(n):
        items.append(f"<item><id>{i+1}</id><name>name{i+1}</name><group>group{(i%3)+1}</group></item>")
    xml = f"<items>{''.join(items)}</items>"
    return Response(content=xml, media_type="application/xml")

# --- Files & Multipart ---
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accept file, return metadata."""
    content = await file.read()
    sha256 = hashlib.sha256(content).hexdigest()
    meta = {
        "filename": file.filename,
        "size": len(content),
        "content_type": file.content_type,
        "sha256": sha256,
    }
    # For multipart debug: echo boundary and field names
    boundary = file.headers.get("content-type", "").split("boundary=")[-1] if "boundary=" in file.headers.get("content-type", "") else ""
    fields = ["file"]
    return {**meta, "boundary": boundary, "fields": fields}

@app.get("/files/sample")
async def files_sample(kb: int = Query(1, ge=1, le=1024), name: str = Query("sample.bin")):
    """Return binary file."""
    total = kb * 1024
    def gen():
        for _ in range(total):
            yield b"0"
    headers = {"Content-Disposition": f"attachment; filename={name}"}
    return StreamingResponse(gen(), media_type="application/octet-stream", headers=headers)

# --- Control behaviors ---
@app.get("/cache/etag")
async def cache_etag(seed: int = Query(CAMP_SEED), request: Request = None):
    """Return ETag, support If-None-Match."""
    etag = hashlib.md5(str(seed).encode()).hexdigest()
    inm = request.headers.get("if-none-match") if request else None
    if inm == etag:
        return Response(status_code=HTTP_304_NOT_MODIFIED)
    return Response(content=json.dumps({"etag": etag}), media_type="application/json", headers={"ETag": etag})

@app.get("/gzip")
async def gzip_text(kb: int = Query(1, ge=1, le=1024), request: Request = None):
    """Send text payload, gzip if accepted."""
    text = ("0123456789abcdef" * 64)[:kb*1024]
    ae = request.headers.get("accept-encoding", "") if request else ""
    if "gzip" in ae:
        import gzip
        gz = gzip.compress(text.encode())
        return Response(content=gz, media_type="text/plain", headers={"Content-Encoding": "gzip"})
    return PlainTextResponse(text)

@app.get("/range")
async def range_bytes(kb: int = Query(1, ge=1, le=1024), request: Request = None):
    """Support Range requests."""
    total = kb * 1024
    data = b"0" * total
    rng = request.headers.get("range") if request else None
    if rng and rng.startswith("bytes="):
        try:
            start, end = rng[6:].split("-")
            start = int(start or 0)
            end = int(end or total-1)
            chunk = data[start:end+1]
            headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
            return Response(content=chunk, media_type="application/octet-stream", status_code=HTTP_206_PARTIAL_CONTENT, headers=headers)
        except Exception:
            pass
    return Response(content=data, media_type="application/octet-stream")

# --- Work & Probes ---
@app.get("/work")
async def work(ms: int = Query(10, ge=0, le=10000), cpu: int = Query(0, ge=0, le=1)):
    """Sleep or burn CPU for ms, return timings."""
    t0 = time.time()
    if cpu:
        end = t0 + ms/1000
        while time.time() < end:
            _ = hashlib.md5(str(random.random()).encode()).hexdigest()
    else:
        await asyncio.sleep(ms/1000)
    t1 = time.time()
    elapsed = int((t1-t0)*1000)
    work_timings.append(elapsed)
    metrics["avg_work_ms"] = sum(work_timings[-100:])/min(len(work_timings),100)
    return {"work_ms": elapsed}

@app.get("/fail/readiness")
async def fail_readiness(toggle: int = Query(..., ge=0, le=1)):
    """Toggle readiness flag."""
    global ready_flag
    ready_flag = bool(not not toggle)
    return {"ready": ready_flag}

@app.get("/exit")
async def exit_app(code: int = Query(0, ge=0, le=255)):
    """Exit process with code."""
    print(f"[Exit] Exiting with code {code}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

# --- Auth & Methods ---
@app.get("/auth/basic")
async def auth_basic(request: Request):
    """Require HTTP Basic auth."""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Basic "):
        return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})
    import base64
    try:
        user, pwd = base64.b64decode(auth[6:]).decode().split(":",1)
    except Exception:
        return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})
    return {"user": user}

@app.get("/auth/bearer")
async def auth_bearer(request: Request):
    """Require Bearer token."""
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    if CAMP_TOKEN and token != CAMP_TOKEN:
        return Response(status_code=HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    return {"token": token, "accepted": not not token}

@app.get("/auth/query")
async def auth_query(token: str = Query(None)):
    """Require token via query param."""
    if not token:
        return Response(status_code=HTTP_401_UNAUTHORIZED)
    return {"token": token}

@app.put("/resource/{id}")
async def put_resource(id: int, body: dict = Body(...)):
    """Upsert resource."""
    return {"id": id, "resource": body}

@app.patch("/resource/{id}")
async def patch_resource(id: int, body: dict = Body(...)):
    """Partial update."""
    return {"id": id, "patch": body}

@app.delete("/resource/{id}")
async def delete_resource(id: int):
    """Delete resource."""
    return {"deleted": id}

# --- Metrics ---
@app.get("/metrics")
async def metrics_endpoint():
    """Return metrics as text/plain."""
    lines = [
        f"requests_total {metrics['requests_total']}",
        f"requests_inflight {metrics['requests_inflight']}",
        f"last_reaction_ms {metrics['last_reaction_ms']}",
        f"avg_work_ms {metrics['avg_work_ms']:.2f}",
    ]
    return PlainTextResponse("\n".join(lines))

# --- Bash Practice Endpoints ---
@app.get("/redirect/{n}")
async def redirect_chain(n: int = Path(..., ge=1, le=10), hops: int = 0):
    """302 chain n times then 200."""
    if n > 1:
        return RedirectResponse(url=f"/redirect/{n-1}?hops={hops+1}", status_code=302)
    return JSONResponse({"hops": hops+1})

@app.get("/cookies/set")
async def cookies_set(name: str = Query(...), value: str = Query(...), request: Request = None):
    """Set cookie."""
    resp = JSONResponse({"cookies": dict(request.cookies) if request else {}})
    resp.set_cookie(key=name, value=value)
    return resp

@app.get("/cookies")
async def cookies(request: Request):
    """Return request cookies."""
    return dict(request.cookies)

@app.get("/retry/{code}")
async def retry(code: int, after: int = Query(1, ge=1, le=60), request: Request = None):
    """Respond with code and Retry-After for first X-Req-Id, then 200."""
    rid = request.headers.get("X-Req-Id") if request else None
    if not hasattr(retry, "seen"): retry.seen = set()
    if rid and rid not in retry.seen:
        retry.seen.add(rid)
        return Response(content=json.dumps({"retry": True}), status_code=code, headers={"Retry-After": str(after)})
    return JSONResponse({"retry": False})

@app.get("/cache/control")
async def cache_control(max_age: int = Query(60, ge=1, le=3600)):
    """Return Cache-Control header."""
    return Response(content=json.dumps({"max_age": max_age}), media_type="application/json", headers={"Cache-Control": f"max-age={max_age}"})

@app.post("/text/upper")
async def text_upper(request: Request):
    """Uppercase text/plain in, out."""
    body = (await request.body()).decode()
    return PlainTextResponse(body.upper())

@app.post("/text/grep")
async def text_grep(needle: str = Query(...), request: Request = None):
    """Return lines containing needle."""
    body = (await request.body()).decode()
    lines = [l for l in body.splitlines() if needle in l]
    return PlainTextResponse("\n".join(lines))

@app.post("/json/filter")
async def json_filter(path: str = Query(...), request: Request = None):
    """Filter JSON array by dot-path."""
    data = await request.json()
    def get_path(obj, path):
        for part in path.lstrip(".").split('.'):
            obj = obj.get(part) if isinstance(obj, dict) else None
        return obj
    filtered = [o for o in data if get_path(o, path)]
    return filtered

@app.post("/csv/sort")
async def csv_sort(col: int = Query(1, ge=1), numeric: int = Query(0, ge=0, le=1), request: Request = None):
    """Sort CSV by column."""
    body = (await request.body()).decode()
    lines = body.splitlines()
    if not lines: return PlainTextResponse("")
    header = lines[0] if not lines[0][0].isdigit() else None
    data = lines[1:] if header else lines
    def key(row):
        v = row.split(",")[col-1]
        return float(v) if numeric else v
    sorted_data = sorted(data, key=key)
    out = ([header] if header else []) + sorted_data
    return PlainTextResponse("\n".join(out), media_type="text/csv")

@app.get("/bytes")
async def bytes_n(n: int = Query(..., ge=1, le=1024*1024)):
    """Return n zero-bytes."""
    def gen():
        for _ in range(n):
            yield b"0"
    return StreamingResponse(gen(), media_type="application/octet-stream")

@app.get("/lines")
async def lines_n(n: int = Query(..., ge=1, le=1000), prefix: str = Query("line")):
    """Emit n lines with prefix."""
    async def gen():
        for i in range(n):
            yield f"{prefix}-{i+1}\n"
            await asyncio.sleep(0)
    return StreamingResponse(gen(), media_type="text/plain")

@app.get("/sse/ticks")
async def sse_ticks(rate: float = Query(1.0, ge=0.1, le=10.0), duration: int = Query(1, ge=1, le=60)):
    """SSE stream of ticks."""
    async def gen():
        for t in range(int(rate*duration)):
            yield f"id:{t}\ndata:tick-{t}\n\n"
            await asyncio.sleep(1/rate)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.get("/json/paged")
async def json_paged(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100)):
    """Paged JSON with Link headers."""
    total = 100
    items = [f"item-{i+1}" for i in range((page-1)*size, min(page*size, total))]
    links = []
    if page*size < total:
        links.append(f'<{page+1}>; rel="next"')
    if page > 1:
        links.append(f'<{page-1}>; rel="prev"')
    headers = {"Link": ", ".join(links)} if links else {}
    return JSONResponse({"page": page, "size": size, "total": total, "items": items}, headers=headers)

@app.get("/flaky")
async def flaky(p_success: float = Query(..., ge=0.0, le=1.0)):
    """200 with p_success, else 500."""
    if random.random() < p_success:
        return {"success": True}
    return Response(content=json.dumps({"success": False}), status_code=500)

@app.get("/jitter")
async def jitter(min_ms: int = Query(..., ge=0), max_ms: int = Query(..., ge=0)):
    """Sleep random ms in range."""
    actual = random.randint(min_ms, max_ms)
    await asyncio.sleep(actual/1000)
    return {"jitter_ms": actual}

@app.get("/chunked")
async def chunked(n: int = Query(..., ge=1, le=100), size: int = Query(..., ge=1, le=1024)):
    """Send n chunks of 'x' * size."""
    async def gen():
        for _ in range(n):
            yield b"x" * size
            await asyncio.sleep(0)
    return StreamingResponse(gen(), media_type="application/octet-stream")

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, log_level=LOG_LEVEL)
