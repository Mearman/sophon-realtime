"""Manual check of tool calling against the real weights (not collected by unittest: it loads the model, about 26 GB, so stop any other
Sophon process first). Usage: python mac/tests/check_tools_with_model.py [--root ~/sophon-mac/models]
It speaks three requests with the macOS `say` voice and prints, for each, the spoken lead-in, the parsed calls and the timing."""
import argparse, json, os, subprocess, sys, tempfile, time, wave
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from engine import Brain, SR

TOOLS = [
    {"type": "function", "name": "get_time", "description": "The current local time.", "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "set_light", "description": "Turn a light on or off.",
     "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "on": {"type": "boolean"}}, "required": ["room", "on"]}},
]


def clip(text):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["say", "-v", "Samantha", "-o", f"{d}/c.aiff", text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", f"{d}/c.aiff", f"{d}/c.wav"], check=True)
        with wave.open(f"{d}/c.wav") as w: return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0


def turn(brain, history, user, label):
    got = []; first = []; t0 = time.time()
    def emit(a):
        if not first: first.append(time.time() - t0)
        got.append(a)
    text, audio, calls = brain.reply(history, user, emit, lambda: False)
    spoken = sum(len(a) for a in got) / 24000
    print(f"{label}: lead-in/text={text!r}; calls={[(c['name'], c['arguments']) for c in calls]}; spoken {spoken:.1f}s; "
          f"first audio {first[0]:.2f}s" if first else f"{label}: text={text!r}; calls={[(c['name'], c['arguments']) for c in calls]}; no audio", f"(total {time.time() - t0:.2f}s)")
    return text, audio, calls


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=os.path.expanduser("~/sophon-mac/models")); ap.add_argument("--lang", default="en")
    a = ap.parse_args()
    brain = Brain(a.root, lang=a.lang, sys_prompt=os.path.join(HERE, "..", "system_prompt.json")); brain.warmup(); brain.set_tools(TOOLS)
    rendered = brain.tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}], tools=brain.tools, tokenize=False, add_generation_prompt=True)
    print("template renders <tools>:", "<tools>" in rendered)

    ask = ("k1", clip("What time is it?"))
    text, _, calls = turn(brain, [], ask, "1 asks the time")
    assert calls and calls[0]["name"] == "get_time", "expected a get_time call"
    entry = {"user": ask[1], "ukey": "k1", "a_audio": None, "akey": "a1", "a_text": "",
             "steps": [{"text": text, "calls": calls, "results": [json.dumps({"time": "10:15 in the morning"})]}]}
    text2, _, calls2 = turn(brain, [entry], None, "2 answers after the result")
    assert not calls2 and text2.strip(), "expected a spoken answer and no further call"
    entry["a_text"] = text2

    light = ("k2", clip("Turn off the kitchen light."))
    text3, _, calls3 = turn(brain, [entry], light, "3 second turn with the first in history")
    assert calls3 and calls3[0]["name"] == "set_light" and calls3[0]["arguments"].get("on") is False, "expected set_light(on=false)"
    print("kitchen" in json.dumps(calls3[0]["arguments"]).lower() and "room argument names the kitchen" or "room argument does not name the kitchen")

    chat = ("k3", clip("How are you today?"))
    _, _, calls4 = turn(brain, [], chat, "4 chit-chat")
    assert not calls4, "chit-chat should not call a tool"
    print("OK")


if __name__ == "__main__":
    main()
