"""Tool round trips through Session with a scripted stand-in for the model (no weights are loaded). Needs MLX (Apple silicon) to import
engine.py, so it is skipped elsewhere. Run from the repository root: python3 -m unittest discover -s mac/tests -q"""
import concurrent.futures, json, os, queue, sys, threading, time, types, unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import numpy as np
    import engine
except ImportError as exc:                       # MLX is only installable on Apple silicon
    engine = None; SKIP = str(exc)

AUDIO = lambda: np.zeros(2400, np.float32)
CALL = {"id": "call_1", "name": "get_time", "arguments": {}}


class FakeBrain:
    """Replies from a script: each item is (text, audio or None, calls); records what reply() was asked."""
    last_prefill = 0; timing = {}

    def __init__(self, script):
        self.script = list(script); self.asked = []; self.tools = None; self.instructions = None; self.threads = []
        self._ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def submit(self, fn, *args): return self._ex.submit(fn, *args)

    def set_tools(self, tools): self.tools = tools

    def set_instructions(self, text): self.instructions = text

    def reply(self, history, user, emit, cancelled, max_tok=0, said="", on_text=None):
        self.asked.append({"user": user, "history": [dict(h) for h in history]}); self.threads.append(threading.current_thread())
        text, a, calls = self.script.pop(0)
        if on_text is not None and text: on_text(text)
        if a is not None: emit(a)
        return text, a, [dict(c) for c in calls]


def new_reply(ukey="u1"):
    return {"user": np.zeros(16000, np.float32), "ukey": ukey, "cancel": False, "done": False, "gen_s": 0.0, "t_sil": time.time(), "t_out": None, "sgen": 0}


@unittest.skipIf(engine is None, "MLX not available")
class ToolRoundTrip(unittest.TestCase):
    def session(self, script, on_tool_call=None, timeout=5.0):
        self.events = queue.Queue(); self.seen = []; self.played = []
        self.brain = FakeBrain(script)
        self.s = engine.Session(self.brain, self.played.append, log=lambda m: self.events.put(json.loads(m)), tools=[{"type": "function", "name": "get_time"}],
                                on_tool_call=on_tool_call, tool_timeout=timeout)
        return self.s

    def wait_for(self, name, n=1, within=5.0):
        """The events named `name` seen so far (all events are kept), once there are at least n of them."""
        end = time.time() + within
        while time.time() < end:
            while True:
                try: self.seen.append(self.events.get_nowait())
                except queue.Empty: break
            got = [e for e in self.seen if e["ev"] == name]
            if len(got) >= n: return got
            time.sleep(0.05)
        self.fail(f"no {name} event ({n} wanted); saw {[e['ev'] for e in self.seen]}")

    def start(self, s):
        r = new_reply(); s.reply = r; s.jobs.put(("reply", r)); return r

    def test_call_result_and_spoken_answer(self):
        calls = []
        def handler(c): calls.append(c); threading.Thread(target=lambda: self.s.tool_result(c["id"], '{"time": "10:15"}')).start()
        s = self.session([("Let me check.", AUDIO(), [CALL]), ("It is ten fifteen.", AUDIO(), [])], on_tool_call=handler)
        self.start(s)
        self.wait_for("reply_done", 2)
        self.assertEqual([c["name"] for c in calls], ["get_time"])
        self.assertEqual(self.brain.tools, [{"type": "function", "name": "get_time"}])       # the session hands its tools to the brain
        self.assertIsNotNone(self.brain.asked[0]["user"])
        self.assertIsNone(self.brain.asked[1]["user"])                                   # the follow-up has no new user audio
        e = s.history[-1]
        self.assertEqual(e["steps"][0]["text"], "Let me check."); self.assertEqual(e["steps"][0]["results"], ['{"time": "10:15"}'])
        self.assertEqual(e["a_text"], "It is ten fifteen."); self.assertEqual(len(s.history), 1)
        self.assertEqual(len(self.played), 2)                                             # lead-in and answer were both played

    def test_history_is_rendered_with_the_call_as_text(self):
        stub = types.SimpleNamespace(_system=lambda: "SYS")
        entry = {"user": np.zeros(10, np.float32), "ukey": "u", "a_audio": None, "akey": "a", "a_text": "It is ten fifteen.",
                 "steps": [{"text": "Let me check.", "calls": [CALL], "results": ["10:15"]}]}
        msgs, audios = engine.Brain._conv(stub, [entry])
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool", "assistant"])
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["name"], "get_time"); self.assertEqual(msgs[3]["content"], "10:15")
        self.assertEqual([k for k, _ in audios], ["u"])                                    # no assistant audio for a turn with a call

    def test_an_unanswered_call_is_not_rendered_as_a_result(self):
        stub = types.SimpleNamespace(_system=lambda: "SYS")
        entry = {"user": np.zeros(10, np.float32), "ukey": "u", "a_audio": None, "akey": "a", "a_text": "",
                 "steps": [{"text": "Let me check.", "calls": [CALL], "results": [None]}]}
        msgs, _ = engine.Brain._conv(stub, [entry])
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"]); self.assertNotIn("tool_calls", msgs[2])

    def test_a_missing_answer_times_out_into_an_error_result_and_a_reply(self):
        s = self.session([("Let me check.", AUDIO(), [CALL]), ("Sorry, I could not.", AUDIO(), [])], timeout=0.3)
        self.start(s)
        tr = self.wait_for("tool_result")[0]
        self.assertTrue(tr["timed_out"]); self.assertTrue(tr["followup"])
        self.wait_for("reply_done", 2)
        self.assertIn("error", json.loads(s.history[-1]["steps"][0]["results"][0]))

    def test_a_result_after_a_newer_turn_is_kept_but_not_spoken(self):
        s = self.session([("Let me check.", AUDIO(), [CALL])])
        self.start(s); self.wait_for("reply_done")
        s.history.append({"user": np.zeros(10, np.float32), "ukey": "u2", "a_audio": None, "akey": "a2", "a_text": "Hello."})
        self.assertTrue(s.tool_result("call_1", "late"))
        self.assertFalse(self.wait_for("tool_result")[0]["followup"])
        self.assertEqual(s.history[0]["steps"][0]["results"], ["late"]); self.assertEqual(len(self.brain.asked), 1)

    def test_a_result_while_the_user_is_speaking_is_not_spoken(self):
        s = self.session([("Let me check.", AUDIO(), [CALL])])
        self.start(s); self.wait_for("reply_done")
        s.speech_ms = 400
        s.tool_result("call_1", "x")
        self.assertFalse(self.wait_for("tool_result")[0]["followup"])

    def test_an_unknown_or_answered_call_id_is_ignored(self):
        s = self.session([("Let me check.", AUDIO(), [CALL])])
        self.start(s); self.wait_for("reply_done")
        self.assertFalse(s.tool_result("nope", "x"))
        s.speech_ms = 400; self.assertTrue(s.tool_result("call_1", "x")); self.assertFalse(s.tool_result("call_1", "again"))

    def test_rounds_are_capped(self):
        script = [("Checking.", AUDIO(), [dict(CALL, id=f"call_{i}")]) for i in range(engine.MAX_STEPS + 1)]
        s = self.session(script, on_tool_call=lambda c: threading.Thread(target=lambda: self.s.tool_result(c["id"], "r")).start())
        self.start(s); self.wait_for("reply_done", engine.MAX_STEPS + 1)
        self.assertEqual(len(s.history[-1]["steps"]), engine.MAX_STEPS)

    def test_a_cancelled_reply_announces_no_call(self):
        s = self.session([("Let me check.", AUDIO(), [CALL])])
        r = new_reply(); r["cancel"] = True; s.reply = r; s.jobs.put(("reply", r))
        self.wait_for("reply_done"); time.sleep(0.2); self.wait_for("reply_done")
        self.assertEqual([e for e in self.seen if e["ev"] == "tool_call"], [])

    def test_every_conversation_runs_on_the_same_model_thread(self):
        """MLX generation streams are thread-local: a second conversation on a new thread cannot use the first one's cache."""
        brain = FakeBrain([("One.", AUDIO(), []), ("Two.", AUDIO(), [])]); events = queue.Queue()
        for _ in range(2):
            s = engine.Session(brain, lambda a: None, log=lambda m: events.put(json.loads(m)))
            r = new_reply(); s.reply = r; s.jobs.put(("reply", r))
            end = time.time() + 5
            while time.time() < end and sum(1 for e in list(events.queue) if e["ev"] == "reply_done") < len(brain.threads): time.sleep(0.02)
            s.close(); self.assertTrue(s.stopped.wait(5))
        self.assertEqual(len(brain.threads), 2); self.assertEqual(len(set(brain.threads)), 1)

    def test_the_quality_of_service_class_can_be_raised_on_a_thread(self):
        errors = []
        def run():
            try: engine.raise_qos()
            except OSError as exc: errors.append(exc)
        t = threading.Thread(target=run); t.start(); t.join(5)
        self.assertEqual(errors, [])

    def test_tools_are_cleared_for_a_session_without_them(self):
        self.session([]); b = FakeBrain([]); b.tools = ["stale"]
        engine.Session(b, lambda a: None, log=lambda m: None)
        self.assertEqual(b.tools, None)


if __name__ == "__main__":
    unittest.main()
