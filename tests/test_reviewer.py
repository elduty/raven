"""Tests for reviewer.py — claude CLI invocation and response parsing."""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

os.environ.setdefault("GITEA_URL", "https://gitea.example.com")
os.environ.setdefault("GITEA_TOKEN", "test-token")

from raven.reviewer import (
    _parse_response,
    _strip_lockfiles_and_binaries,
    respond_to_comment,
    review_diff,
    severity_gte,
)


# ------------------------------------------------------------------ #
#  _parse_response                                                    #
# ------------------------------------------------------------------ #

class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "severity": "medium",
            "summary": "Found a potential bug",
            "findings": [{"severity": "medium", "message": "Off-by-one error"}],
        })
        result = _parse_response(raw)
        assert result["severity"] == "medium"
        assert result["summary"] == "Found a potential bug"
        assert len(result["findings"]) == 1

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"severity": "high", "summary": "XSS risk", "findings": []}\n```'
        result = _parse_response(raw)
        assert result["severity"] == "high"

    def test_malformed_json_returns_fallback(self):
        result = _parse_response("This is not JSON at all.")
        assert result["severity"] == "high"
        assert result["_parse_error"] is True

    def test_invalid_json_syntax(self):
        result = _parse_response("{severity: bad json}")
        assert result["_parse_error"] is True

    def test_unknown_severity_normalised_to_low(self):
        raw = json.dumps({"severity": "critical", "summary": "Bad", "findings": []})
        result = _parse_response(raw)
        assert result["severity"] == "low"

    def test_findings_with_unknown_severity(self):
        raw = json.dumps({
            "severity": "low",
            "summary": "ok",
            "findings": [{"severity": "extreme", "message": "oops"}],
        })
        result = _parse_response(raw)
        assert result["findings"][0]["severity"] == "low"

    def test_empty_findings_list(self):
        raw = json.dumps({"severity": "low", "summary": "LGTM", "findings": []})
        result = _parse_response(raw)
        assert result["findings"] == []

    def test_findings_with_file_and_line(self):
        raw = json.dumps({
            "severity": "high",
            "summary": "Bug",
            "findings": [{"severity": "high", "message": "Issue", "file": "server.py", "line": 42}],
        })
        result = _parse_response(raw)
        assert result["findings"][0]["file"] == "server.py"
        assert result["findings"][0]["line"] == 42

    def test_findings_without_file_line_omits_fields(self):
        raw = json.dumps({
            "severity": "low",
            "summary": "ok",
            "findings": [{"severity": "low", "message": "General"}],
        })
        result = _parse_response(raw)
        assert "file" not in result["findings"][0]
        assert "line" not in result["findings"][0]

    def test_json_with_trailing_text(self):
        raw = 'Here is my review: {"severity": "high", "summary": "XSS", "findings": []} and some more text'
        result = _parse_response(raw)
        assert result["severity"] == "high"
        assert result.get("_parse_error") is not True

    def test_greedy_trap_multiple_braces(self):
        raw = 'The code uses {key: value} syntax. {"severity": "medium", "summary": "Bug", "findings": []}'
        result = _parse_response(raw)
        assert result["severity"] == "medium"


# ------------------------------------------------------------------ #
#  review_diff (with mocked subprocess)                              #
# ------------------------------------------------------------------ #

class TestReviewDiff:
    def _make_result(self, stdout, returncode=0):
        m = MagicMock()
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = ""
        return m

    def test_successful_review(self):
        review_json = json.dumps({"severity": "low", "summary": "ok", "findings": []})
        with patch("subprocess.run", return_value=self._make_result(review_json)) as mock_run:
            result = review_diff("diff content\n", "user/myrepo")
        assert result["severity"] == "low"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "--effort" in cmd
        assert "max" in cmd

    def test_claude_exit_nonzero_raises(self):
        with patch("subprocess.run", return_value=self._make_result("", returncode=1)):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                review_diff("diff", "user/repo")

    def test_timeout_raises(self):
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300)):
            with pytest.raises(RuntimeError, match="timed out"):
                review_diff("diff", "user/repo")

    def test_claude_not_found_raises(self):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="not found"):
                review_diff("diff", "user/repo")

    def test_claude_md_included_in_prompt(self):
        review_json = json.dumps({"severity": "low", "summary": "ok", "findings": []})
        with patch("subprocess.run", return_value=self._make_result(review_json)) as mock_run:
            review_diff("diff content\n", "user/repo", claude_md="This is a game engine.")
        stdin_input = mock_run.call_args[1]["input"]
        assert "This is a game engine." in stdin_input

    def test_prompt_passed_via_stdin_with_print_flag(self):
        review_json = json.dumps({"severity": "low", "summary": "ok", "findings": []})
        with patch("subprocess.run", return_value=self._make_result(review_json)) as mock_run:
            review_diff("diff content\n", "user/myrepo")
        cmd = mock_run.call_args[0][0]
        assert "-p" in cmd
        # Prompt is NOT a positional arg after -p — it's delivered via stdin
        p_idx = cmd.index("-p")
        assert p_idx + 1 >= len(cmd) or cmd[p_idx + 1].startswith("--")
        stdin_input = mock_run.call_args[1]["input"]
        assert "diff content" in stdin_input


# ------------------------------------------------------------------ #
#  _strip_lockfiles_and_binaries                                     #
# ------------------------------------------------------------------ #

class TestStripLockfiles:
    def test_strips_yarn_lock(self):
        diff = (
            "diff --git a/yarn.lock b/yarn.lock\n"
            "index abc..def 100644\n"
            "--- a/yarn.lock\n"
            "+++ b/yarn.lock\n"
            "@@ -1,3 +1,3 @@\n"
            " existing line\n"
            "-old dep\n"
            "+new dep\n"
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "+print('hello')\n"
        )
        result = _strip_lockfiles_and_binaries(diff)
        assert "yarn.lock" not in result
        assert "src/app.py" in result

    def test_strips_binary_image(self):
        diff = (
            "diff --git a/assets/logo.png b/assets/logo.png\n"
            "Binary files a/assets/logo.png and b/assets/logo.png differ\n"
            "diff --git a/main.py b/main.py\n"
            "+code\n"
        )
        result = _strip_lockfiles_and_binaries(diff)
        assert "logo.png" not in result
        assert "main.py" in result

    def test_keeps_regular_files(self):
        diff = (
            "diff --git a/src/utils.py b/src/utils.py\n"
            "+def foo(): pass\n"
        )
        result = _strip_lockfiles_and_binaries(diff)
        assert "utils.py" in result

    def test_strips_package_lock(self):
        diff = (
            "diff --git a/package-lock.json b/package-lock.json\n"
            "+{ 'version': 2 }\n"
            "diff --git a/index.js b/index.js\n"
            "+const x = 1;\n"
        )
        result = _strip_lockfiles_and_binaries(diff)
        assert "package-lock.json" not in result
        assert "index.js" in result


# ------------------------------------------------------------------ #
#  severity_gte                                                       #
# ------------------------------------------------------------------ #

class TestChunkedReviewAllFail:
    """Fix 2: if every chunk fails, _parse_error must be set."""

    def _make_result(self, stdout, returncode=0):
        m = MagicMock()
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = ""
        return m

    def test_all_chunks_fail_sets_parse_error(self):
        big_diff = (
            "diff --git a/a.py b/a.py\n" + "+line\n" * 200 +
            "diff --git a/b.py b/b.py\n" + "+line\n" * 200
        )
        with patch("subprocess.run", return_value=self._make_result("", returncode=1)):
            import raven.reviewer as rev
            old_max = rev.MAX_DIFF_LINES
            rev.MAX_DIFF_LINES = 100
            try:
                result = review_diff(big_diff, "user/repo")
            finally:
                rev.MAX_DIFF_LINES = old_max
        assert result.get("_parse_error") is True
        assert result["chunked"] is True
        assert result["chunks_reviewed"] == 0

    def test_partial_success_no_parse_error(self):
        big_diff = (
            "diff --git a/a.py b/a.py\n" + "+line\n" * 200 +
            "diff --git a/b.py b/b.py\n" + "+line\n" * 200
        )
        with patch("subprocess.run", side_effect=[
            self._make_result(json.dumps({"severity": "low", "summary": "ok", "findings": []})),
            self._make_result("", returncode=1),
        ]):
            import raven.reviewer as rev
            old_max = rev.MAX_DIFF_LINES
            rev.MAX_DIFF_LINES = 100
            try:
                result = review_diff(big_diff, "user/repo")
            finally:
                rev.MAX_DIFF_LINES = old_max
        assert result.get("_parse_error") is not True
        assert result["chunked"] is True
        assert result["chunks_reviewed"] == 1


class TestSeverityGte:
    def test_same_level(self):
        assert severity_gte("medium", "medium") is True

    def test_higher(self):
        assert severity_gte("high", "medium") is True
        assert severity_gte("high", "low") is True
        assert severity_gte("medium", "low") is True

    def test_lower(self):
        assert severity_gte("low", "medium") is False
        assert severity_gte("low", "high") is False
        assert severity_gte("medium", "high") is False


# ------------------------------------------------------------------ #
#  respond_to_comment — file/line context                             #
# ------------------------------------------------------------------ #

class TestRespondToCommentFileContext:
    def _make_result(self, stdout, returncode=0):
        m = MagicMock()
        m.returncode = returncode
        m.stdout = stdout
        m.stderr = ""
        return m

    def test_respond_to_comment_includes_file_context(self):
        with patch("raven.reviewer.subprocess") as mock_sub:
            mock_sub.run.return_value = self._make_result("Response text")
            respond_to_comment(
                "why is this bad?", [], "diff content", "owner/repo",
                file_path="server.py", line=42,
            )
        prompt = mock_sub.run.call_args[1]["input"]
        assert "server.py" in prompt
        assert "line 42" in prompt

    def test_respond_to_comment_file_only_no_line(self):
        with patch("raven.reviewer.subprocess") as mock_sub:
            mock_sub.run.return_value = self._make_result("Response text")
            respond_to_comment(
                "explain this", [], "diff content", "owner/repo",
                file_path="utils.py",
            )
        prompt = mock_sub.run.call_args[1]["input"]
        assert "utils.py" in prompt
        assert "line" not in prompt.lower().split("utils.py")[1].split("\n")[0]

    def test_respond_to_comment_no_file_context(self):
        with patch("raven.reviewer.subprocess") as mock_sub:
            mock_sub.run.return_value = self._make_result("Response text")
            respond_to_comment(
                "general question", [], "diff content", "owner/repo",
            )
        prompt = mock_sub.run.call_args[1]["input"]
        assert "Code Location" not in prompt
