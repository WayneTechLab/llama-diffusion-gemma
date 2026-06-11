#!/usr/bin/env python3
"""
DiffusionGemma server with full Ollama-compatible API + OpenAI-compatible API.

Listens on port 11435.  Acts as an Ollama-compatible proxy:
  - Requests for diffusion-gemma (or the HuggingFace alias) are handled locally
    via llama-diffusion-cli with the Metal OOM fixes applied.
  - All other model requests are forwarded to the real Ollama at localhost:11434.

Usage:
  OLLAMA_HOST=http://localhost:11435 ollama run diffusion-gemma
  curl http://localhost:11435/api/chat  ...
  curl http://localhost:11435/v1/chat/completions  ...  (OpenAI-compat)
"""
import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH   = "/Users/waynetechlab/AI/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
BINARY       = "/Users/waynetechlab/AI/llama.cpp-diffusiongemma/build/bin/llama-diffusion-cli"
NGL          = "99"          # Metal GPU layers
OLLAMA_UPSTREAM = "http://localhost:11450"
PORT         = 11434

# Model name aliases that this server handles (everything else proxies upstream)
DIFFUSION_NAMES = {
    "diffusion-gemma",
    "hf.co/unsloth/diffusiongemma-26B-A4B-it-GGUF:Q4_K_M",
    "hf.co/unsloth/diffusiongemma-26B-A4B-it-GGUF",
}

app = FastAPI(title="DiffusionGemma Ollama-compatible API")
_infer_lock = asyncio.Semaphore(1)  # one inference at a time (16 GB model on GPU)


# ── Helpers ────────────────────────────────────────────────────────────────────

def is_diffusion_model(name: Optional[str]) -> bool:
    if not name:
        return False
    return name.strip() in DIFFUSION_NAMES or name.startswith("diffusion-gemma")


def messages_to_prompt(messages: List[Dict]) -> str:
    parts = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"<system>{content}</system>")
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def build_cmd(prompt: str, n: int = 128, temp: float = 0.7,
              steps: int = 16, algo: int = 4, block: int = 16) -> List[str]:
    return [
        BINARY, "-m", MODEL_PATH, "-p", prompt,
        "-ngl", NGL, "-n", str(n),
        "--temp", str(temp),
        "--diffusion-steps", str(steps),
        "--diffusion-algorithm", str(algo),
        "--diffusion-block-length", str(block),
        "--log-colors", "off",
    ]


def parse_output(raw: str) -> str:
    match = re.search(r"<channel\|>(.*?)(?:total time:|$)", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    raw = re.sub(r"^<\|channel>thought\s*", "", raw.strip())
    lines = [l for l in raw.splitlines() if not l.startswith("total time:")]
    return "\n".join(lines).strip()


async def run_inference(cmd: List[str], timeout: int = 900) -> str:
    async with _infer_lock:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise HTTPException(status_code=504, detail="Inference timed out")
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail="Inference failed - check GPU/model")
        return parse_output(stdout.decode("utf-8", errors="replace"))


# ── Proxy helper ───────────────────────────────────────────────────────────────

async def proxy_to_ollama(request: Request) -> JSONResponse:
    body  = await request.body()
    async with httpx.AsyncClient(timeout=900) as client:
        resp = await client.request(
            method  = request.method,
            url     = f"{OLLAMA_UPSTREAM}{request.url.path}",
            headers = {k: v for k, v in request.headers.items() if k.lower() != "host"},
            content = body,
        )
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ── Ollama API ─────────────────────────────────────────────────────────────────

@app.get("/api/tags")
async def ollama_tags(request: Request):
    """Merge our model into the upstream Ollama tag list."""
    our_model = {
        "name":        "diffusion-gemma:latest",
        "model":       "diffusion-gemma:latest",
        "modified_at": "2026-06-11T00:00:00.000000000-07:00",
        "size":        16806811129,
        "digest":      "0ce1d29bcc425f42f97490390eaa4b2ced90bbd6c943b76544a6a6804db681ec",
        "details": {
            "parent_model":       "",
            "format":             "gguf",
            "family":             "diffusion-gemma",
            "families":           ["diffusion-gemma"],
            "parameter_size":     "26B",
            "quantization_level": "Q4_K_M",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_UPSTREAM}/api/tags")
            data = r.json()
            data["models"] = [our_model] + [
                m for m in data.get("models", [])
                if not is_diffusion_model(m.get("name"))
            ]
            return JSONResponse(data)
    except Exception:
        return JSONResponse({"models": [our_model]})


@app.post("/api/chat")
async def ollama_chat(request: Request):
    body = await request.json()
    if not is_diffusion_model(body.get("model")):
        return await proxy_to_ollama(request)

    messages  = body.get("messages", [])
    stream    = body.get("stream", True)
    opts      = body.get("options", {})
    n         = opts.get("num_predict", body.get("max_tokens", 128))
    temp      = opts.get("temperature", 0.7)
    steps     = opts.get("diffusion_steps", 16)
    prompt    = messages_to_prompt(messages)
    cmd       = build_cmd(prompt, n=n, temp=temp, steps=steps)
    ts        = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if stream:
        async def stream_gen():
            async with _infer_lock:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
            content = parse_output(stdout.decode("utf-8", errors="replace"))
            # Ollama streams word by word; we emit in one chunk then done
            chunk = {"model": "diffusion-gemma", "created_at": ts,
                     "message": {"role": "assistant", "content": content},
                     "done": False}
            yield json.dumps(chunk) + "\n"
            done  = {"model": "diffusion-gemma", "created_at": ts,
                     "message": {"role": "assistant", "content": ""},
                     "done": True, "done_reason": "stop"}
            yield json.dumps(done) + "\n"

        return StreamingResponse(stream_gen(), media_type="application/x-ndjson")

    content = await run_inference(cmd)
    return JSONResponse({
        "model":      "diffusion-gemma",
        "created_at": ts,
        "message":    {"role": "assistant", "content": content},
        "done":       True,
        "done_reason": "stop",
    })


@app.post("/api/generate")
async def ollama_generate(request: Request):
    body = await request.json()
    if not is_diffusion_model(body.get("model")):
        return await proxy_to_ollama(request)

    prompt = body.get("prompt", "")
    opts   = body.get("options", {})
    n      = opts.get("num_predict", body.get("max_tokens", 128))
    temp   = opts.get("temperature", 0.7)
    steps  = opts.get("diffusion_steps", 16)
    cmd    = build_cmd(prompt, n=n, temp=temp, steps=steps)
    ts     = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stream = body.get("stream", True)

    if stream:
        async def stream_gen():
            async with _infer_lock:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
            content = parse_output(stdout.decode("utf-8", errors="replace"))
            yield json.dumps({"model": "diffusion-gemma", "created_at": ts,
                              "response": content, "done": False}) + "\n"
            yield json.dumps({"model": "diffusion-gemma", "created_at": ts,
                              "response": "", "done": True}) + "\n"

        return StreamingResponse(stream_gen(), media_type="application/x-ndjson")

    content = await run_inference(cmd)
    return JSONResponse({
        "model": "diffusion-gemma", "created_at": ts,
        "response": content, "done": True,
    })


@app.post("/api/show")
async def ollama_show(request: Request):
    body = await request.json()
    if not is_diffusion_model(body.get("model", body.get("name"))):
        return await proxy_to_ollama(request)
    return JSONResponse({
        "modelfile": "FROM diffusion-gemma\n",
        "details": {"family": "diffusion-gemma", "parameter_size": "26B",
                    "quantization_level": "Q4_K_M"},
    })


@app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE", "HEAD"])
async def proxy_other_ollama(request: Request, path: str):
    return await proxy_to_ollama(request)


# Ollama CLI does HEAD / before any other request as a liveness check
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return JSONResponse({"status": "ok"})


# ── OpenAI-compatible API ──────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "diffusion-gemma"
    messages: List[Message]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    diffusion_steps: Optional[int] = 16
    diffusion_algorithm: Optional[int] = 4
    diffusion_block_length: Optional[int] = 16

def make_openai_response(content: str, req_id: str) -> dict:
    return {
        "id": f"chatcmpl-{req_id}", "object": "chat.completion",
        "created": int(time.time()), "model": "diffusion-gemma",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [
        {"id": "diffusion-gemma", "object": "model", "created": 0, "owned_by": "local"}
    ]}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt  = messages_to_prompt([m.dict() for m in req.messages])
    cmd     = build_cmd(prompt, n=req.max_tokens, temp=req.temperature,
                        steps=req.diffusion_steps, algo=req.diffusion_algorithm,
                        block=req.diffusion_block_length)

    if req.stream:
        req_id = uuid.uuid4().hex
        async def stream_gen():
            async with _infer_lock:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                stdout, _ = await proc.communicate()
            content = parse_output(stdout.decode("utf-8", errors="replace"))
            chunk = {"id": f"chatcmpl-{req_id}", "object": "chat.completion.chunk",
                     "created": int(time.time()), "model": "diffusion-gemma",
                     "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n"
            done = {"id": f"chatcmpl-{req_id}", "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": "diffusion-gemma",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    content = await run_inference(cmd)
    return make_openai_response(content, uuid.uuid4().hex)


if __name__ == "__main__":
    import socket
    import uvicorn

    # Create a dual-stack socket (IPv4 + IPv6) by disabling IPV6_V6ONLY.
    # macOS sets IPV6_V6ONLY=1 by default, so binding to '::' alone misses IPv4.
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("::", PORT))
    sock.listen(128)

    print(f"DiffusionGemma proxy on port {PORT} (IPv4+IPv6)")
    uvicorn.run(app, fd=sock.fileno(), log_level="warning")
