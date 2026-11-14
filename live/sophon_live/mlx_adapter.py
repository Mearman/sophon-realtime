"""The Mac engine (mac/engine.py, MLX) as a gateway engine, and an MLX transcript model.

MlxAdapter has the same methods and event stream as VllmOmniAdapter (see adapters.py), but the model runs in this process instead of behind a /v1/realtime socket and the tool bridge: tool calls come from Session's on_tool_call and go back through Session.tool_result. The model is loaded once per process (MlxEngine.load) and holds one conversation at a time. Not carried over from the CUDA engine: the voice and the language are those the model was loaded with, and a history passed at the start is ignored (the Mac engine's history is audio).

MlxASR is the gateway's transcript model on MLX (Qwen3-ASR through mlx-audio) in place of the transformers one, which is hard-coded to CUDA."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid

import numpy as np

log = logging.getLogger("sophon_live.mlx")
FR = 320                                     # the Mac engine takes 20 ms frames at 16 kHz
MLX_ASR_DEFAULT = "mlx-community/Qwen3-ASR-0.6B-8bit"


class MlxEngine:
    """The model, loaded once, and the one conversation it can hold."""
    brain = None
    lang = None
    busy = False

    @classmethod
    def load(cls, root: str, lang: str, voice: str) -> None:
        mac = os.environ.get("SOPHON_MAC_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "mac")
        if mac not in sys.path:
            sys.path.insert(0, mac)
        from engine import Brain, raise_qos
        raise_qos()                              # the gateway's own thread, which paces the audio
        cls.brain = Brain(root, lang=lang, voice=voice, sys_prompt=os.path.join(mac, "system_prompt.json"))
        cls.brain.warmup()
        cls.lang = lang


class MlxAdapter:
    """One conversation on the Mac engine, as the gateway's engine."""

    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.cfg: dict = {"extra_body": {}}
        self.sess = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._rid: str | None = None
        self._buf = np.zeros(0, np.float32)
        self._closed = False
        self._owns = False

    async def start(self, *, instructions: str, voice: str, lang: str, tools: list, history: list) -> None:
        if MlxEngine.brain is None:
            raise RuntimeError("the Mac engine is not loaded")
        if MlxEngine.busy:
            raise RuntimeError("the Mac engine holds one conversation at a time")
        MlxEngine.busy = self._owns = True
        if lang != MlxEngine.lang:
            log.warning("the model was loaded for %r and the session asks for %r: keeping %r", MlxEngine.lang, lang, MlxEngine.lang)
        if history:
            log.warning("the Mac engine does not carry a text history over (%d messages ignored)", len(history))
        from engine import Session
        self.loop = asyncio.get_running_loop()
        self.cfg["extra_body"] = {"tools": tools, "language": lang, "history": history}
        MlxEngine.brain.set_instructions(instructions)
        self.sess = Session(MlxEngine.brain, self._play, log=self._log, tools=tools, on_tool_call=self._on_tool_call, on_text=self._on_text)

    # ----------------------------------------------------------------------------------------------------------------------- engine to gateway
    def _put(self, item: dict) -> None:
        if self.loop is not None and not self._closed:
            self.loop.call_soon_threadsafe(self.q.put_nowait, item)

    def _rid_now(self) -> str:
        if self._rid is None:
            self._rid = "resp_" + uuid.uuid4().hex[:16]
        return self._rid

    def _play(self, x) -> None:                 # the engine's worker thread: 24 kHz reply audio, or None when the user cut in
        if x is None:
            self._put({"t": "cancel"})
        else:
            self._put({"t": "audio", "pcm": np.asarray(x, np.float32), "sr": 24000, "rid": self._rid_now()})

    def _log(self, m: str) -> None:
        e = json.loads(m)
        if e.get("ev") == "reply_start":
            self._rid = "resp_" + uuid.uuid4().hex[:16]
        elif e.get("ev") == "reply_done":
            rid, self._rid = self._rid_now(), None
            self._put({"t": "done", "rid": rid, "status": "cancelled" if e.get("cancelled") else "completed"})
        elif e.get("ev") == "error":
            self._put({"t": "error", "message": f"{e.get('where')}: {e.get('err')}"})

    def _on_text(self, delta: str) -> None:      # the spoken text as it is written, ahead of the audio (the gateway releases it as the audio plays)
        self._put({"t": "text", "delta": delta, "rid": self._rid_now()})

    def _on_tool_call(self, call: dict) -> None:
        self._put({"t": "tool_call", "call_id": call["id"], "name": call["name"], "arguments": call["arguments"]})

    # ----------------------------------------------------------------------------------------------------------------------- gateway to engine
    async def send_audio(self, f32: np.ndarray) -> None:
        if self.sess is None or self._closed:
            return
        self._buf = np.concatenate([self._buf, np.asarray(f32, np.float32)])
        while len(self._buf) >= FR:
            f, self._buf = self._buf[:FR], self._buf[FR:]
            self.sess.feed(f)

    async def set_instructions(self, text: str) -> None:
        MlxEngine.brain.set_instructions(text)
        MlxEngine.brain.set_tools(self.cfg["extra_body"].get("tools"))

    async def tool_result(self, call_id: str | None, output: str, speak: bool = True) -> bool:
        return bool(self.sess is not None and self.sess.tool_result(call_id, output, speak=speak))

    async def close(self) -> None:
        self._closed = True
        if self.sess is not None:
            self.sess.close()
            await asyncio.to_thread(self.sess.stopped.wait, 15)       # the model is shared: the next conversation waits for this worker
        if self._owns:
            MlxEngine.busy = False


class MlxASR:
    """Input transcript on MLX: Qwen3-ASR through mlx-audio. A callable from 16 kHz float32 audio to text, like the gateway's ASR."""

    def __init__(self, model_id: str = MLX_ASR_DEFAULT):
        from mlx_audio.stt import load
        self.model = load(model_id)

    def __call__(self, a16k: np.ndarray) -> str:
        r = self.model.generate(np.asarray(a16k, np.float32))
        return (getattr(r, "text", r) or "").strip()
