"""Shared pieces of the bench tools (stdlib only): rank 0's OpenAI-compatible API, the greedy continuation past EOS,
and a deterministic filler-text generator. Run from the control machine over the LAN or on rank 0.

Defaults come from the caller's environment, then the recipe's .env, then .env.example: the URL is
http://<first NODES address>:<PORT> (override: --url or INKLING2_URL), the bearer token API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EOS = 200006  # <|content_model_end_sampling|>: generation always stops here; greedy continues past it by re-requesting


def recipe_setting(key: str, default: str = "") -> str:
    """KEY from the environment, else the recipe's .env, else .env.example (the controller's precedence)."""
    if os.environ.get(key):
        return os.environ[key]
    for name in (".env", ".env.example"):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith(key + "="):
                    continue
                v = line[len(key) + 1:].strip()
                if v[:1] in "\"'":
                    v = v[1:v.index(v[0], 1)] if v[0] in v[1:] else v[1:]
                else:
                    v = v.split("#", 1)[0].strip()
                return v
    return default


def default_url() -> str:
    if os.environ.get("INKLING2_URL"):
        return os.environ["INKLING2_URL"]
    node = recipe_setting("NODES", "192.168.1.121").split()[0]
    return f"http://{node}:{recipe_setting('PORT', '8000')}"


class ApiError(RuntimeError):
    def __init__(self, status: int, body, path: str):
        self.status, self.body, self.path = status, body, path
        super().__init__(f"{path}: HTTP {status}: {str(body)[:500]}")


class Api:
    def __init__(self, url: str, model: str | None = None, api_key: str | None = None, timeout: float = 3600.0):
        self.url = url.rstrip("/")
        self.api_key = api_key if api_key is not None else recipe_setting("API_KEY")
        self.timeout = timeout
        self._model = model

    def _request(self, method: str, path: str, body=None, timeout: float | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        return urllib.request.urlopen(req, timeout=timeout or self.timeout)

    def call(self, method: str, path: str, body=None, timeout: float | None = None) -> tuple[int, object, float]:
        """(status, parsed JSON or text, seconds); never raises on an HTTP error status."""
        t0 = time.monotonic()
        try:
            with self._request(method, path, body, timeout) as r:
                status, raw = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        dt = time.monotonic() - t0
        try:
            return status, json.loads(raw), dt
        except ValueError:
            return status, raw.decode(errors="replace"), dt

    def post(self, path: str, body: dict, timeout: float | None = None) -> tuple[dict, float]:
        status, out, dt = self.call("POST", path, body, timeout)
        if status != 200 or not isinstance(out, dict):
            raise ApiError(status, out, path)
        return out, dt

    def stream(self, path: str, body: dict, timeout: float | None = None):
        """POST with stream=true; yields (seconds since the request, event dict) per SSE data line. Raises ApiError on
        a non-200 answer."""
        t0 = time.monotonic()
        try:
            r = self._request("POST", path, dict(body, stream=True), timeout)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                raise ApiError(e.code, json.loads(raw), path) from None
            except ValueError:
                raise ApiError(e.code, raw.decode(errors="replace"), path) from None
        with r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                yield time.monotonic() - t0, json.loads(data)

    @property
    def model(self) -> str:
        if self._model is None:
            status, out, _ = self.call("GET", "/v1/models", timeout=30)
            if status != 200 or not isinstance(out, dict) or not out.get("data"):
                raise ApiError(status, out, "/v1/models")
            self._model = out["data"][0]["id"]
        return self._model

    def tokenize(self, text: str) -> list[int]:
        out, _ = self.post("/tokenize", {"model": self.model, "text": text})
        return [int(t) for t in out["tokens"]]

    def complete(self, prompt, max_tokens: int, **extra) -> tuple[dict, float]:
        body = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0,
                "return_token_ids": True, **extra}
        return self.post("/v1/completions", body)

    def chat(self, messages: list, max_tokens: int, **extra) -> tuple[dict, float]:
        """A greedy chat completion with the generated token ids (`choices[0].token_ids`)."""
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                "return_token_ids": True, **extra}
        return self.post("/v1/chat/completions", body)

    def metrics(self) -> dict[str, float]:
        """/metrics as {sample name with its labels: value} (Prometheus text format)."""
        status, out, _ = self.call("GET", "/metrics", timeout=30)
        if status != 200 or not isinstance(out, str):
            raise ApiError(status, out, "/metrics")
        return parse_metrics(out)

    def prompt_logprobs(self, prompt_ids: list[int], k: int) -> tuple[list, float]:
        out, dt = self.post("/v1/completions", {"model": self.model, "prompt": prompt_ids, "prompt_logprobs": k,
                                                "temperature": 0})
        if "prompt_logprobs" not in out:
            raise ApiError(200, out, "/v1/completions (prompt_logprobs missing)")
        return out["prompt_logprobs"], dt


def generated_ids(resp: dict) -> tuple[list[int], str]:
    """The generated ids of a /v1/completions answer and its finish reason; the EOS that ended a `stop` answer is
    appended when the server left it out."""
    ch = resp["choices"][0]
    ids = ch.get("token_ids")
    if ids is None:
        raise ApiError(200, resp, "/v1/completions (no token_ids: return_token_ids unsupported?)")
    ids = [int(t) for t in ids]
    finish = ch.get("finish_reason")
    if finish == "stop" and (not ids or ids[-1] != EOS):
        ids.append(EOS)
    return ids, finish


# The Inkling 2 reply framing (special tokens; crates/server/src/formats/inkling2.rs): the server reads it and takes it out,
# so neither `content` nor `reasoning_content` may hold one of these.
FRAMING = ("<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>",
           "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>", "<|content_model_end_sampling|>")


def spec_counters(usage: dict | None) -> dict | None:
    """usage.spec (drafted, accepted, rounds) with the derived acceptance rate (accepted / drafted) and the tokens per
    round (completion tokens / rounds); None when the server reported no spec block."""
    s = (usage or {}).get("spec")
    if not isinstance(s, dict):
        return None
    d, a, r = int(s.get("drafted", 0)), int(s.get("accepted", 0)), int(s.get("rounds", 0))
    n = int((usage or {}).get("completion_tokens", 0))
    return {"drafted": d, "accepted": a, "rounds": r, "acceptance": round(a / d, 4) if d else None,
            "tokens_per_round": round(n / r, 4) if r else None}


def cached_tokens(usage: dict | None) -> int | None:
    """usage.prompt_tokens_details.cached_tokens: prompt tokens a prefix-cache hit supplied (None: not reported)."""
    v = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return None if v is None else int(v)


def parse_metrics(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def greedy(api: Api, prompt: list[int], n: int) -> tuple[list[int], int, float]:
    """Greedy (temperature 0) continuation of exactly n tokens from token ids `prompt`. Generation stops at EOS, so a
    continuation past it re-requests with prompt + everything so far (EOS included). Returns (ids, requests, seconds)."""
    got: list[int] = []
    requests, seconds = 0, 0.0
    while len(got) < n:
        resp, dt = api.complete(prompt + got, n - len(got))
        ids, finish = generated_ids(resp)
        requests += 1
        seconds += dt
        if not ids:
            raise ApiError(200, resp, f"/v1/completions (no progress at {len(got)}/{n}, finish {finish})")
        got += ids
    return got[:n], requests, seconds


def add_api_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--url", default=default_url(), help="rank 0's API (default: %(default)s)")
    ap.add_argument("--model", help="served model name (default: the first id of /v1/models)")
    ap.add_argument("--timeout", type=float, default=7200.0, help="per-request seconds (default: %(default)s)")


def api_from_args(a) -> Api:
    return Api(a.url, a.model, timeout=a.timeout)


# Filler: varied, deterministic sentences (a seed fixes the text), so a long prompt is reproducible but not one phrase
# repeated (a repeated phrase lets attention shortcut a long-context test).
_SUBJ = ["The archivist", "A courier", "The night clerk", "An engineer", "The harbor master", "A botanist", "The auditor",
         "A cartographer", "The baker", "A violinist", "The surveyor", "A pilot", "The librarian", "A geologist"]
_VERB = ["catalogued", "inspected", "repaired", "counted", "photographed", "described", "measured", "relocated",
         "labelled", "sketched", "weighed", "returned", "polished", "recorded"]
_OBJ = ["copper lanterns", "maps of the northern coast", "glass beakers", "wooden crates", "letters from abroad",
        "brass compasses", "seed packets", "old ledgers", "ceramic tiles", "silk banners", "iron keys", "tide tables"]
_PLACE = ["in the east wing", "near the river gate", "on the third floor", "behind the market", "at the old mill",
          "beside the station", "in the cellar", "under the clock tower", "along the canal", "inside the greenhouse"]
_TIME = ["before dawn", "on a rainy Tuesday", "after the festival", "during the long winter", "at noon",
         "late in the evening", "in early spring", "the following week"]


def filler_sentences(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.append(f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.randint(2, 997)} {rng.choice(_OBJ)} "
                   f"{rng.choice(_PLACE)} {rng.choice(_TIME)} (entry {seed}-{i}).")
    return out
