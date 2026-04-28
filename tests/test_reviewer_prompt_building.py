"""Tests for raven.reviewer's prompt-building helpers and end-to-end
prompt assembly: _build_rules_section, _build_pr_context_section, and
the review_diff path that combines them.

These tests live in their own file (separate from test_reviewer.py) to
keep the prompt-construction surface visible and editable independently
of the backend dispatch / response-parsing tests in test_reviewer.py.
The Popen-based end-to-end tests intentionally exercise the full
prompt-assembly path through the ClaudeCLIBackend subprocess so that
prompt-shape regressions (rules ordering, chunked-mode comment
omission, PR-context truncation) are caught at the boundary that
actually feeds the model.
"""

import os

from unittest.mock import MagicMock, patch



class TestPromptBuilding:
    def test_build_rules_empty_returns_empty_string(self):
        """Nothing to inject → no section header either."""
        from raven.reviewer import _build_rules_section
        assert _build_rules_section(None, "deadbeef") == ""
        assert _build_rules_section({}, "deadbeef") == ""

    def test_build_rules_renders_each_file_wrapped(self):
        from raven.reviewer import _build_rules_section
        rules = {
            ".claude/rules/security.md": "Always parameterize SQL.",
            ".claude/rules/style.md": "Use PEP 8.",
        }
        section = _build_rules_section(rules, "cafef00d")
        assert "Repository Rules" in section
        assert ".claude/rules/security.md" in section
        assert ".claude/rules/style.md" in section
        assert "Always parameterize SQL." in section
        # Every rule file wrapped in the randomised untrusted-input tag
        assert '<untrusted_input_cafef00d type="repo_file">' in section

    def test_build_rules_truncates_oversized_file(self):
        """A single huge rule file mustn't blow past the per-item cap."""
        import raven.reviewer as rev
        original = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 50
        try:
            rules = {".claude/rules/big.md": "x" * 200}
            section = rev._build_rules_section(rules, "cafef00d")
        finally:
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original
        assert "x" * 50 in section
        assert "x" * 200 not in section
        assert "truncated" in section

    def test_build_rules_respects_global_budget(self):
        """Total cap across all rule files; later files dropped when
        budget exhausted."""
        import raven.reviewer as rev
        original_total = rev.REVIEW_RULES_TOTAL_CHARS
        original_item = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        rev.REVIEW_RULES_TOTAL_CHARS = 200
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 10_000
        try:
            rules = {
                ".claude/rules/a.md": "A" * 150,  # fits
                ".claude/rules/b.md": "B" * 100,  # chopped to fit budget
                ".claude/rules/c.md": "C" * 100,  # dropped entirely
            }
            section = rev._build_rules_section(rules, "cafef00d")
        finally:
            rev.REVIEW_RULES_TOTAL_CHARS = original_total
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original_item
        assert "A" * 150 in section
        assert ".claude/rules/a.md" in section
        # b.md present with some of its content + marker
        assert ".claude/rules/b.md" in section
        assert "truncated at global cap" in section
        # c.md dropped — budget was exhausted
        assert ".claude/rules/c.md" not in section
        assert "C" * 100 not in section

    def test_build_rules_zero_total_disables_global_cap(self):
        """Symmetric with REVIEW_PR_CONTEXT_TOTAL_CHARS=0: zero means
        "no global cap" (per-file cap still applies)."""
        import raven.reviewer as rev
        original = rev.REVIEW_RULES_TOTAL_CHARS
        rev.REVIEW_RULES_TOTAL_CHARS = 0
        try:
            rules = {f".claude/rules/{n}.md": f"content-{n}" for n in range(5)}
            section = rev._build_rules_section(rules, "cafef00d")
        finally:
            rev.REVIEW_RULES_TOTAL_CHARS = original
        for n in range(5):
            assert f"content-{n}" in section

    def test_review_diff_forwards_rules_to_prompt(self):
        """End-to-end: rules reach the CLI prompt payload."""
        import json
        import raven.reviewer as rev
        review_json = json.dumps({"severity": "low", "summary": "ok", "findings": []})

        fake_proc = MagicMock()
        fake_proc.__enter__.return_value = fake_proc
        fake_proc.__exit__.return_value = None
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = (review_json, "")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc):
            rev.review_diff(
                "diff --git a/x.py b/x.py\n+line\n", "owner/repo",
                rules={".claude/rules/security.md": "UNIQUE-RULE-MARKER"},
            )

        prompt = fake_proc.communicate.call_args[1]["input"]
        assert "Repository Rules" in prompt
        assert "UNIQUE-RULE-MARKER" in prompt

    def test_rules_appear_after_prompt_template(self, monkeypatch):
        """Recency: rules are positioned between the prompt template and
        the diff so they are the last guidance Claude reads before the
        review target. Together with the explicit 'take precedence'
        header, this makes rules beat conflicting prompt-template text."""
        from raven.ai import _reset_backend_cache
        from raven.reviewer import review_diff

        captured = {}

        fake_backend = MagicMock()
        fake_backend.name = "claude_cli"

        def fake_complete(prompt, **kwargs):
            captured["prompt"] = prompt
            return '{"severity":"low","summary":"ok","findings":[]}'

        fake_backend.complete.side_effect = fake_complete
        monkeypatch.setattr("raven.ai._cached_backend", fake_backend)

        review_diff(
            "diff --git a/x b/x\n+line\n",
            "owner/repo",
            rules={".claude/rules/sec.md": "UNIQUE-RULE-MARKER"},
            prompt_override="UNIQUE-PROMPT-TEMPLATE-MARKER",
        )
        _reset_backend_cache()

        prompt = captured["prompt"]
        idx_template = prompt.find("UNIQUE-PROMPT-TEMPLATE-MARKER")
        idx_rules = prompt.find("UNIQUE-RULE-MARKER")
        idx_diff = prompt.find("## Diff to Review")
        assert idx_template != -1 and idx_rules != -1 and idx_diff != -1
        assert idx_template < idx_rules < idx_diff

    def test_rules_section_header_states_precedence(self):
        """Rules header explicitly claims precedence over the prompt so
        the model knows which side wins in a conflict."""
        from raven.reviewer import _build_rules_section
        section = _build_rules_section(
            {".claude/rules/security.md": "content"}, "cafef00d",
        )
        assert "precedence" in section.lower()

    def test_build_pr_context_empty_returns_empty_string(self):
        """Nothing to say → no section header either. Keeps the prompt
        lean on PRs that open without a description and no comments."""
        from raven.reviewer import _build_pr_context_section
        assert _build_pr_context_section("", "", None, "deadbeef") == ""
        assert _build_pr_context_section("", "", [], "deadbeef") == ""

    def test_build_pr_context_includes_title_description_comments(self):
        from raven.reviewer import _build_pr_context_section
        section = _build_pr_context_section(
            pr_title="Add retry to API client",
            pr_description="Network reliability fix.\nReferences DEV-123.",
            pr_comments=[{"user": {"login": "alice"}, "body": "Please also log the retry count"}],
            tag_id="cafef00d",
        )
        assert "PR Context" in section
        assert "Add retry to API client" in section
        assert "DEV-123" in section
        assert "alice" in section
        # Every user-content block wrapped under the randomised tag
        assert '<untrusted_input_cafef00d type="pr_title">' in section
        assert '<untrusted_input_cafef00d type="pr_description">' in section
        assert '<untrusted_input_cafef00d type="pr_conversation">' in section

    def test_build_pr_context_filters_bot_own_comments(self):
        """Including the bot's own review comments would feed the model
        its prior findings as if they were new developer context,
        doubling up observations on re-review. The bot login is
        deployment-specific (``BITBUCKET_DC_USERNAME``, or whatever user
        owns the Gitea token — e.g. ``raven-bot``, ``ci-raven``), so it
        must be passed in. Case-insensitive match handles provider
        normalisation differences."""
        from raven.reviewer import _build_pr_context_section
        section = _build_pr_context_section(
            pr_title="", pr_description="",
            pr_comments=[
                {"user": {"login": "Raven-Bot"}, "body": "earlier review"},
                {"user": {"login": "raven-bot"}, "body": "re-review"},
                {"user": {"login": "alice"}, "body": "human comment"},
            ],
            tag_id="cafef00d",
            bot_user="raven-bot",
        )
        assert "human comment" in section
        assert "earlier review" not in section
        assert "re-review" not in section

    def test_build_pr_context_survives_null_body_in_comment(self):
        """Same null-key-vs-null-value trap as the login fix: a provider
        returning ``{"body": None}`` (edited-empty comment, some weird
        intermediate state) would otherwise reach
        ``_truncate_for_context(None)`` → ``len(None)`` → TypeError and
        crash the entire review."""
        from raven.reviewer import _build_pr_context_section
        section = _build_pr_context_section(
            pr_title="", pr_description="",
            pr_comments=[
                {"user": {"login": "alice"}, "body": None},
                {"user": {"login": "bob"}, "body": "real content"},
            ],
            tag_id="cafef00d",
        )
        # Neither call crashed; the null body renders as an empty string
        assert "real content" in section
        assert "alice" in section

    def test_build_pr_context_truncates_oversized_title(self):
        """The per-item-cap rationale applies to titles too: PR titles
        have no enforced length in most providers, so a huge paste (or
        adversarial title) could dominate the prompt before the diff."""
        import raven.reviewer as rev
        original = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 40
        try:
            section = rev._build_pr_context_section(
                pr_title="t" * 200, pr_description="",
                pr_comments=None, tag_id="cafef00d",
            )
        finally:
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original
        assert "t" * 40 in section
        assert "t" * 200 not in section
        assert "truncated" in section

    def test_build_pr_context_survives_null_login_in_comment(self):
        """Regression guard: ``dict.get(key, default)`` returns ``default``
        only when the key is *absent*, not when the value is ``None``.
        A comment shaped like ``{"user": {"login": None}}`` (deleted
        author, anonymous comment) would make ``.get("login", "").lower()``
        crash with AttributeError. The extra ``or ""`` makes both
        filter and render paths tolerant."""
        from raven.reviewer import _build_pr_context_section
        section = _build_pr_context_section(
            pr_title="", pr_description="",
            pr_comments=[
                {"user": {"login": None}, "body": "ghost comment"},
                {"user": None, "body": "no user object"},
                {"user": {"login": "alice"}, "body": "human comment"},
            ],
            tag_id="cafef00d",
            bot_user="raven-bot",
        )
        # All three comments render without crashing
        assert "ghost comment" in section
        assert "no user object" in section
        assert "human comment" in section
        # The null-login entries fall back to the "unknown" label
        assert "unknown" in section

    def test_build_pr_context_no_bot_user_applies_no_filter(self):
        """If no bot_user is passed (or it's empty), all comments pass
        through. Callers that don't have the bot login available should
        still get something workable rather than a mis-filter."""
        from raven.reviewer import _build_pr_context_section
        section = _build_pr_context_section(
            pr_title="", pr_description="",
            pr_comments=[
                {"user": {"login": "raven"}, "body": "raven comment"},
                {"user": {"login": "alice"}, "body": "alice comment"},
            ],
            tag_id="cafef00d",
            bot_user="",
        )
        assert "raven comment" in section
        assert "alice comment" in section

    def test_build_pr_context_caps_at_review_comment_context(self):
        """Comments grow without bound on long-lived PRs; keep only the
        last REVIEW_COMMENT_CONTEXT so the prompt isn't dominated by old
        resolved discussions."""
        import raven.reviewer as rev
        original = rev.REVIEW_COMMENT_CONTEXT
        rev.REVIEW_COMMENT_CONTEXT = 3
        try:
            comments = [
                {"user": {"login": "alice"}, "body": f"comment-{i}"}
                for i in range(10)
            ]
            section = rev._build_pr_context_section("", "", comments, "cafef00d")
        finally:
            rev.REVIEW_COMMENT_CONTEXT = original
        # Only the last 3 survive
        assert "comment-9" in section
        assert "comment-8" in section
        assert "comment-7" in section
        assert "comment-6" not in section
        assert "comment-0" not in section

    def test_build_pr_context_respects_global_total_budget(self):
        """Small PRs with long discussions can have the conversation
        dwarf the diff, anchoring the model on back-and-forth instead
        of the code. REVIEW_PR_CONTEXT_TOTAL_CHARS caps the whole
        section; comments are added newest-first until the budget is
        hit, so the most recent (usually most relevant) survive."""
        import raven.reviewer as rev
        original_total = rev.REVIEW_PR_CONTEXT_TOTAL_CHARS
        original_item = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        # Tight total budget — a title + description + one comment just
        # fits; additional comments should be dropped entirely.
        rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = 200
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 50
        try:
            comments = [
                {"user": {"login": "alice"}, "body": f"comment-body-{i}" + "-" * 40}
                for i in range(10)
            ]
            section = rev._build_pr_context_section(
                pr_title="short",
                pr_description="short-desc",
                pr_comments=comments,
                tag_id="cafef00d",
            )
        finally:
            rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = original_total
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original_item
        # Title + description always fit
        assert "short" in section
        assert "short-desc" in section
        # Newest comment (comment-body-9) prioritised; oldest dropped
        assert "comment-body-9" in section
        assert "comment-body-0" not in section
        # Rough sanity on overall size — budget caps raw content at 200
        # chars; rendered section adds section header + per-subsection
        # wrapping tags (~300 chars overhead). The uncapped path would
        # blow well past 1500 chars (10 full comments × wrapping).
        assert len(section) < 700

    def test_global_budget_truncation_appends_marker_not_mid_word(self):
        """Regression guard: at the global-budget boundary, an
        overflowing item used to be silently chopped mid-word (e.g.
        remaining=3 → ``text[:3]``) with no truncation marker. That
        contradicts the docstring's drop-or-mark semantics and can
        surface a 1-3 char stub as a "comment". Now: chop with a
        visible marker, or drop the item entirely when the budget is
        too small to fit even a marker."""
        import raven.reviewer as rev
        original_total = rev.REVIEW_PR_CONTEXT_TOTAL_CHARS
        original_item = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        # Title fits, description gets chopped at budget boundary
        rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = 100
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 10_000  # disable per-item trimming
        try:
            section = rev._build_pr_context_section(
                pr_title="short",
                pr_description="x" * 500,  # way over remaining budget
                pr_comments=None,
                tag_id="cafef00d",
            )
        finally:
            rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = original_total
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original_item
        # Description prefix present, but with a marker — not silently cut
        assert "xxx" in section
        assert "x" * 500 not in section
        assert "truncated at global cap" in section

    def test_global_budget_drops_stub_when_too_small_for_marker(self):
        """If the remaining budget is smaller than the marker itself,
        don't emit a 1-char junk stub — drop the overflow entirely."""
        import raven.reviewer as rev
        original_total = rev.REVIEW_PR_CONTEXT_TOTAL_CHARS
        original_item = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        # After the title, ~5 chars remain — smaller than the marker.
        rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = 10
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 10_000
        try:
            section = rev._build_pr_context_section(
                pr_title="title-5ch",  # 9 chars, leaves 1
                pr_description="would not fit even the marker",
                pr_comments=[{"user": {"login": "alice"}, "body": "tiny"}],
                tag_id="cafef00d",
            )
        finally:
            rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = original_total
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original_item
        # Title made it. Description and comment were dropped (budget
        # too small for even a marker). No junk stub visible.
        assert "title-5ch" in section
        assert "would not fit" not in section
        assert "tiny" not in section
        # And no naked truncation stub lurking in the output
        assert "### Description" not in section

    def test_build_pr_context_zero_total_disables_global_cap(self):
        """Symmetric with the item-char knob: zero means "no global cap"
        (per-item caps still apply). Gives operators a way to fall back
        to the pre-cap behaviour."""
        import raven.reviewer as rev
        original = rev.REVIEW_PR_CONTEXT_TOTAL_CHARS
        rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = 0
        try:
            comments = [
                {"user": {"login": "alice"}, "body": f"comment-{i}"}
                for i in range(10)
            ]
            section = rev._build_pr_context_section(
                "t", "d", comments, "cafef00d",
            )
        finally:
            rev.REVIEW_PR_CONTEXT_TOTAL_CHARS = original
        # All 10 comments survive (bounded only by REVIEW_COMMENT_CONTEXT)
        for i in range(10):
            assert f"comment-{i}" in section

    def test_build_pr_context_zero_cap_disables_comments(self):
        """Regression guard: the naive ``list[-N:]`` idiom is broken at
        ``N == 0`` because ``list[-0:]`` evaluates to ``list[0:]`` (the
        full list). A user setting ``RAVEN_REVIEW_COMMENT_CONTEXT=0`` to
        turn the feature off would otherwise get the *opposite* of what
        they asked for. Zero must disable the comments subsection."""
        import raven.reviewer as rev
        original = rev.REVIEW_COMMENT_CONTEXT
        rev.REVIEW_COMMENT_CONTEXT = 0
        try:
            section = rev._build_pr_context_section(
                pr_title="title", pr_description="",
                pr_comments=[{"user": {"login": "alice"}, "body": "noise"}],
                tag_id="cafef00d",
            )
        finally:
            rev.REVIEW_COMMENT_CONTEXT = original
        assert "title" in section
        assert "noise" not in section
        assert "Recent Comments" not in section

    def test_build_pr_context_truncates_oversized_description(self):
        """A spec pasted into the PR description would otherwise inflate
        the prompt and dominate the diff. Cap via
        ``_truncate_for_context`` and emit a marker so the model can tell
        content was dropped."""
        import raven.reviewer as rev
        original = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 50
        try:
            long_desc = "x" * 200
            section = rev._build_pr_context_section(
                pr_title="", pr_description=long_desc,
                pr_comments=None, tag_id="cafef00d",
            )
        finally:
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original
        # Prefix preserved, full text NOT present, truncation marker shown
        assert "x" * 50 in section
        assert "x" * 200 not in section
        assert "truncated" in section

    def test_build_pr_context_truncates_oversized_comment_bodies(self):
        import raven.reviewer as rev
        original = rev.REVIEW_PR_CONTEXT_ITEM_CHARS
        rev.REVIEW_PR_CONTEXT_ITEM_CHARS = 30
        try:
            section = rev._build_pr_context_section(
                pr_title="", pr_description="",
                pr_comments=[{"user": {"login": "alice"}, "body": "y" * 200}],
                tag_id="cafef00d",
            )
        finally:
            rev.REVIEW_PR_CONTEXT_ITEM_CHARS = original
        assert "y" * 30 in section
        assert "y" * 200 not in section
        assert "truncated" in section

    def test_review_diff_drops_comments_in_chunked_mode(self):
        """Chunked reviews split by file; comments are PR-wide context
        and would otherwise get replicated into every per-file prompt
        (with defaults: 20 comments × 4000 chars ≈ 80KB × N files).
        Title and description are short enough to carry intent in every
        chunk; comments are not."""
        import json
        import raven.reviewer as rev
        review_json = json.dumps({"severity": "low", "summary": "ok", "findings": []})

        # Force chunked path by shrinking MAX_DIFF_LINES
        big_diff = (
            "diff --git a/a.py b/a.py\n" + "+line\n" * 200
            + "diff --git a/b.py b/b.py\n" + "+line\n" * 200
        )

        captured_prompts: list[str] = []

        def make_proc(*args, **kwargs):
            fake = MagicMock()
            fake.__enter__.return_value = fake
            fake.__exit__.return_value = None
            fake.returncode = 0
            def communicate(input, timeout):
                captured_prompts.append(input)
                return (review_json, "")
            fake.communicate.side_effect = communicate
            return fake

        old_max = rev.MAX_DIFF_LINES
        rev.MAX_DIFF_LINES = 100
        try:
            with patch("raven.ai.claude_cli.subprocess.Popen", side_effect=make_proc):
                rev.review_diff(
                    big_diff, "owner/repo",
                    pr_title="Refactor API client",
                    pr_description="Rework the retry logic",
                    pr_comments=[{"user": {"login": "alice"}, "body": "UNIQUE-COMMENT-MARKER"}],
                )
        finally:
            rev.MAX_DIFF_LINES = old_max

        # Chunked → at least 2 prompts captured
        assert len(captured_prompts) >= 2
        for prompt in captured_prompts:
            # Title + description always propagate (short, PR-wide intent)
            assert "Refactor API client" in prompt
            assert "Rework the retry logic" in prompt
            # Comments do NOT — chunked mode skips them to save tokens
            assert "UNIQUE-COMMENT-MARKER" not in prompt

    def test_review_diff_forwards_pr_context_to_chunk(self):
        """End-to-end that pr_title/description/comments reach the prompt
        passed to the Claude CLI (not just the _build helper)."""
        import json
        import raven.reviewer as rev
        review_json = json.dumps({"severity": "low", "summary": "ok", "findings": []})

        fake_proc = MagicMock()
        fake_proc.__enter__.return_value = fake_proc
        fake_proc.__exit__.return_value = None
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = (review_json, "")

        with patch("raven.ai.claude_cli.subprocess.Popen", return_value=fake_proc):
            rev.review_diff(
                "diff --git a/x.py b/x.py\n+line\n", "owner/repo",
                pr_title="Add retry to API client",
                pr_description="Network reliability fix",
                pr_comments=[{"user": {"login": "alice"}, "body": "LGTM conceptually"}],
            )

        prompt = fake_proc.communicate.call_args[1]["input"]
        assert "Add retry to API client" in prompt
        assert "Network reliability fix" in prompt
        assert "LGTM conceptually" in prompt
        assert "PR Context" in prompt
