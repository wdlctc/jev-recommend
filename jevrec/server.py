"""TypeSafe-compatible ``POST /v1/systemone`` server around DecisionEngine.

  python -m jevrec.server --model Qwen/Qwen3-4B --readout letters --port 8790

Requests are served one at a time (one GPU lock); every question in a request
is answered and ``usage.input_tokens`` reports the tokens actually processed.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .decide import DecisionEngine


def make_handler(engine: DecisionEngine, name: str):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/health"):
                return self._send(200, {"ok": True, "model": name})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "use POST /v1/systemone"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                questions = req["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("'questions' must be a non-empty object")
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            answers, tokens = {}, 0
            start = time.perf_counter()
            try:
                with lock:
                    for qid, q in questions.items():
                        answers[qid], n = engine.answer(req.get("state", ""), q)
                        tokens += n
            except (ValueError, KeyError, TypeError) as e:
                return self._send(422, {"error": f"cannot answer: {e}"})
            self._send(200, {"model": name, "answers": answers,
                             "usage": {"input_tokens": tokens, "output_tokens": 0},
                             "latency_ms": 1000 * (time.perf_counter() - start)})

        def log_message(self, *args):  # keep stdout for errors only
            pass

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--adapter", default=None, help="optional LoRA adapter directory")
    p.add_argument("--readout", choices=["letters", "branches"], default="letters")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--name", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    args = p.parse_args()
    engine = DecisionEngine(args.model, args.readout, temperature=args.temperature, adapter=args.adapter)
    name = args.name or f"jevrec-{args.model.split('/')[-1]}-{args.readout}"
    # Warm up kernels so the first measured request is not a cold start.
    for _ in range(3):
        engine.answer("warm up", {"type": "noul", "instructions": "Is this a warm-up?"})
    print(f"serving {name} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(engine, name)).serve_forever()


if __name__ == "__main__":
    main()
