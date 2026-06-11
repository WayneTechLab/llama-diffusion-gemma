#!/usr/bin/env python3
"""
OpenAI-compatible API server wrapping llama-diffusion-cli for DiffusionGemma.
Exposes /v1/chat/completions and /v1/models so any OpenAI client works.
"""
import asyncio
import json
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_PATH = "/Users/waynetechlab/AI/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
BINARY    = "/Users/waynetechlab/AI/llama.cpp-diffusiongemma/build/bin/llama-diffusion-cli"
MODEL_ID  = "diffusion-gemma"

# Both Metal OOM root causes fixed: sc_embT now falls back to CPU when GPU memory is tight,
# and n_ubatch is correctly sized to canvas_length for the KV-cache path.
NGL = "99"  # GPU layers: 99 = all layers on Metal

app = FastAPI(title="DiffusionGemma OpenAI-compatible API")
_infer_lock = asyncio.Semaphore(1)  # one inference at a time — model is 16 GB on GPU


# ── Pydantic models ────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: List[Message]
    max_tokens: Optional[int] = 64   # keep low for interactive speed on CPU
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False
    diffusion_steps: Optional[int] = 32   # fewer steps = faster, slightly lower quality
    diffusion_algorithm: Optional[int] = 4
    diffusion_block_length: Optional[int] = 16


# ── Helpers ────────────────────────────────────────────────────────────────────

def messages_to_prompt(messages: List[Message]) -> str:
    """Convert chat messages to a single prompt string."""
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"<system>{m.content}</system>")
        elif m.role == "user":
            parts.append(f"<start_of_turn>user\n{m.content}<end_of_turn>")
        elif m.role == "assistant":
            parts.append(f"<start_of_turn>model\n{m.content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)

def build_cmd(prompt: str, req: ChatRequest) -> List[str]:
    # Let the binary handle n_ubatch sizing (fixed to use canvas_length for KV path).
    # Do not pass -ub to avoid overriding the internal calculation.
    return [
        BINARY,
        "-m", MODEL_PATH,
        "-p", prompt,
        "-ngl", NGL,
        "-n", str(req.max_tokens),
        "--temp", str(req.temperature),
        "--diffusion-steps", str(req.diffusion_steps),
        "--diffusion-algorithm", str(req.diffusion_algorithm),
        "--diffusion-block-length", str(req.diffusion_block_length),
        "--log-colors", "off",
    ]

def parse_output(raw: str) -> str:
    """Extract assistant reply from diffusion-cli stdout.
    Format: <|channel>thought\\n[thinking]\\n<channel|>[reply]\\ntotal time:...
    If response completes, return only the final reply.
    If truncated mid-think, strip the channel header and return what we have.
    """
    import re
    # Full response: extract text after closing <channel|> tag
    match = re.search(r"<channel\|>(.*?)(?:total time:|$)", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Truncated thinking: strip <|channel>thought header + timing footer
    raw = re.sub(r"^<\|channel>thought\s*", "", raw.strip())
    lines = [l for l in raw.splitlines() if not l.startswith("total time:")]
    return "\n".join(lines).strip()

def make_response(content: str, req_id: str, finish_reason: str = "stop") -> dict:
    return {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }

def make_stream_chunk(delta: str, req_id: str, finish_reason=None) -> str:
    chunk = {
        "id": f"chatcmpl-{req_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "delta": {"content": delta} if delta else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = messages_to_prompt(req.messages)
    cmd    = build_cmd(prompt, req)

    if req.stream:
        async def stream_gen():
            req_id = uuid.uuid4().hex
            yield make_stream_chunk("", req_id)  # role delta
            async with _infer_lock:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                async for line in proc.stdout:
                    text = line.decode("utf-8", errors="replace")
                    yield make_stream_chunk(text, req_id)
                await proc.wait()
            yield make_stream_chunk("", req_id, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    # Non-streaming
    async with _infer_lock:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=600)
            except asyncio.TimeoutError:
                proc.kill()
                raise HTTPException(status_code=504, detail="Inference timed out")
            output = parse_output(stdout.decode("utf-8", errors="replace"))
            if proc.returncode != 0 and not output:
                raise HTTPException(status_code=500, detail="Inference failed - check model/memory")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return make_response(output, uuid.uuid4().hex)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8081, log_level="info")
