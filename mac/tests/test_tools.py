"""Tests for mac/tools.py (pure Python). Run from the repository root: python3 -m unittest discover -s mac/tests -q"""
import json, os, sys, unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools import MAX_STEPS, SYS_TOOL, TC_CLOSE, TC_OPEN, WAIT, cut_tool, flatten, has_tool, normalize_tools, parse_tool_calls


class CutTool(unittest.TestCase):
    def test_plain_text_is_spoken_whole(self):
        self.assertEqual(cut_tool("It is ten fifteen.", final=True), "It is ten fifteen.")

    def test_text_before_a_call_only(self):
        self.assertEqual(cut_tool('Let me check.<tool_call>{"name": "t"}</tool_call>'), "Let me check.")

    def test_a_partial_marker_is_held_back_while_streaming(self):
        self.assertEqual(cut_tool("Let me check.<to"), "Let me check.")
        self.assertEqual(cut_tool("Let me check.<tool_cal"), "Let me check.")
        self.assertEqual(cut_tool("a < b"), "a < b")          # a lone "<" followed by a space is not a marker

    def test_a_partial_marker_is_kept_when_the_text_is_final(self):
        self.assertEqual(cut_tool("Let me check.<to", final=True), "Let me check.<to")

    def test_wait_is_silent(self):
        self.assertEqual(cut_tool(WAIT + " nothing new", final=True), "")
        self.assertEqual(cut_tool("<wa"), "")                  # streaming: could still become <wait>

    def test_a_call_with_no_lead_in_is_silent(self):
        self.assertEqual(cut_tool('<tool_call>{"name": "t"}</tool_call>'), "")


class Parse(unittest.TestCase):
    def test_one_call(self):
        c = parse_tool_calls('Sure.<tool_call>{"name": "light", "arguments": {"room": "kitchen", "on": false}}</tool_call>')
        self.assertEqual([(x["name"], x["arguments"]) for x in c], [("light", {"room": "kitchen", "on": False})])
        self.assertTrue(c[0]["id"].startswith("call_"))

    def test_several_calls_get_distinct_ids(self):
        c = parse_tool_calls('<tool_call>{"name": "a"}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>')
        self.assertEqual([x["name"] for x in c], ["a", "b"])
        self.assertEqual(len({x["id"] for x in c}), 2)

    def test_arguments_as_a_string_and_as_parameters(self):
        self.assertEqual(parse_tool_calls('<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>')[0]["arguments"], {"x": 1})
        self.assertEqual(parse_tool_calls('<tool_call>{"name": "a", "parameters": {"y": 2}}</tool_call>')[0]["arguments"], {"y": 2})

    def test_a_missing_close_tag_still_parses(self):
        self.assertEqual(parse_tool_calls('<tool_call>{"name": "a", "arguments": {"x": 1}}')[0]["arguments"], {"x": 1})

    def test_malformed_calls_are_skipped(self):
        self.assertEqual(parse_tool_calls("<tool_call>not json</tool_call>"), [])
        self.assertEqual(parse_tool_calls('<tool_call>{"arguments": {}}</tool_call>'), [])
        self.assertEqual(parse_tool_calls(""), [])

    def test_has_tool(self):
        self.assertTrue(has_tool("x" + TC_OPEN)); self.assertFalse(has_tool("x"))


class Normalize(unittest.TestCase):
    def test_both_styles_give_chat_completions_style(self):
        rt = {"type": "function", "name": "a", "description": "d", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}
        cc = {"type": "function", "function": {"name": "a", "description": "d", "parameters": rt["parameters"]}}
        self.assertEqual(normalize_tools([rt]), normalize_tools([cc]))
        self.assertEqual(normalize_tools([rt])[0]["function"]["name"], "a")

    def test_missing_parameters_get_an_empty_schema_and_junk_is_dropped(self):
        out = normalize_tools([{"type": "function", "name": "a"}, "junk", {"type": "function"}, None])
        self.assertEqual(out, [{"type": "function", "function": {"name": "a", "parameters": {"type": "object", "properties": {}}}}])
        self.assertEqual(normalize_tools(None), [])


class Flatten(unittest.TestCase):
    def test_tools_go_into_the_system_text_and_calls_become_text(self):
        tools = normalize_tools([{"type": "function", "name": "a"}])
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "a", "arguments": {"x": 1}}}]},
                {"role": "tool", "tool_call_id": "1", "content": "done"}]
        out = flatten(msgs, tools)
        self.assertIn("<tools>", out[0]["content"]); self.assertTrue(out[0]["content"].startswith("S"))
        self.assertIn(TC_OPEN, out[2]["content"]); self.assertNotIn("tool_calls", out[2])
        self.assertEqual(out[3], {"role": "user", "content": "<tool_response>\ndone\n</tool_response>"})
        self.assertEqual(json.loads(out[2]["content"].split(TC_OPEN)[1].split(TC_CLOSE)[0]), {"name": "a", "arguments": {"x": 1}})
        self.assertEqual(msgs[2]["role"], "assistant"); self.assertIn("tool_calls", msgs[2])      # the input is not modified


class Constants(unittest.TestCase):
    def test_a_system_instruction_exists_for_each_language(self):
        self.assertEqual(sorted(SYS_TOOL), ["en", "zh"]); self.assertGreater(MAX_STEPS, 0)


if __name__ == "__main__":
    unittest.main()
