"""Prior-findings-aware re-review: Raven's own threads are the identity
of its findings across re-reviews (spec 2026-09-29)."""

import contextlib
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

import raven.server as _server_mod
from raven.providers import GitProvider
from raven.server import (_collect_prior_findings, _format_inline_body,
                          _parse_inline_body, _process_pr, _previous_diffs,
                          _recent_prs)
from raven.severity import SeverityScale, default_scale


class TestInlineBody:
    """get_review_threads hands back only a thread's body, so the parser
    must read exactly what the formatter wrote."""

    @pytest.mark.parametrize("sev", ["low", "medium", "high"])
    def test_round_trip_default_scale(self, sev):
        scale = default_scale()
        body = _format_inline_body({"severity": sev, "message": "x is wrong"}, scale)
        assert _parse_inline_body(body, scale) == (sev, "x is wrong")

    def test_round_trip_custom_scale_multiline_message(self):
        scale = SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                              blocks_at_or_above="bug")
        msg = "first line\n\n`code` and **[bold]** text"
        body = _format_inline_body({"severity": "blocker", "message": msg}, scale)
        assert _parse_inline_body(body, scale) == ("blocker", msg)

    def test_missing_severity_formats_as_least_severe(self):
        scale = default_scale()
        body = _format_inline_body({"message": "m"}, scale)
        assert _parse_inline_body(body, scale) == ("low", "m")

    def test_tier_not_on_the_scale_is_none(self):
        scale = SeverityScale(ranks={"nit": 10, "bug": 20}, blocks_at_or_above="bug")
        assert _parse_inline_body("🔴 **[high]** m", scale) is None

    @pytest.mark.parametrize("body", ["", "Thanks, fixed.", "🦅 **Raven Review**\n\nok",
                                      "**[high]** no emoji", None])
    def test_foreign_body_is_none(self, body):
        assert _parse_inline_body(body, default_scale()) is None


KEY = "gitea:owner/repo#42"


def _diff(files: dict[str, list[str]]) -> str:
    """A unified diff adding ``lines`` as each new file."""
    return "".join(
        f"diff --git a/{name} b/{name}\nnew file mode 100644\n--- /dev/null\n"
        f"+++ b/{name}\n@@ -0,0 +1,{len(lines)} @@\n" + "".join(f"+{l}\n" for l in lines)
        for name, lines in files.items())


class FakePlatform:
    """A PR's inline threads kept across passes: submit_review opens one
    per inline comment, retract_finding resolves it, and a test can add,
    resolve or reply to threads as a user."""

    def __init__(self, bot: str = "Raven"):
        self.bot = bot
        self.threads: dict[int, dict] = {}
        self.retracted: list[int] = []
        self._next = 1000
        self.fail_submit = False

    def open_thread(self, file, line, body, replies=0) -> int:
        cid = self._next
        self._next += 1
        self.threads[cid] = {"file": file, "line": line, "body": body,
                             "replies": replies, "resolved": False}
        return cid

    def submit_review(self, repo, pr, body, approve=False, inline_comments=None, **kw):
        if self.fail_submit:
            raise RuntimeError("submit failed")
        posted = [{"file": c["file"], "line": c["line"],
                   "comment_id": self.open_thread(c["file"], c["line"], c["body"])}
                  for c in inline_comments or []]
        return {"id": self._next + 10_000, "inline_comments": posted}

    def retract_finding(self, repo, pr, comment_id) -> bool:
        self.retracted.append(comment_id)
        t = self.threads.get(comment_id)
        if t is None:
            return False
        t["resolved"] = True
        return True

    def get_review_threads(self, repo, pr, bot_user):
        return [{"comment_id": cid, **t} for cid, t in self.threads.items()]

    def get_resolved_comment_ids(self, repo, pr):
        return {cid for cid, t in self.threads.items() if t["resolved"]}

    def open_ids(self) -> set[int]:
        return {cid for cid, t in self.threads.items() if not t["resolved"]}

    def cid_for(self, message: str) -> int:
        [cid] = [c for c, t in self.threads.items() if t["body"].endswith(message)]
        return cid


def _provider(fake: FakePlatform) -> MagicMock:
    mc = MagicMock(spec=GitProvider)
    mc.name = "gitea"
    mc.submit_review.side_effect = fake.submit_review
    mc.retract_finding.side_effect = fake.retract_finding
    mc.get_review_threads.side_effect = fake.get_review_threads
    mc.get_resolved_comment_ids.side_effect = fake.get_resolved_comment_ids
    mc.get_authenticated_user.return_value = fake.bot
    mc.fetch_file.return_value = ""
    mc.get_pr_reviews.return_value = [{"user": {"login": fake.bot}, "state": "APPROVED"}]
    mc.get_pr_requested_reviewers.return_value = []
    mc.add_label_to_pr.return_value = None
    return mc


def _run_pass(mc, diff, head, script, mode="advisory", output="both", scale=None):
    """One push: _process_pr against ``diff`` at ``head``; ``script``
    stands in for review_diff (called with its kwargs). ``scale`` replaces
    the repo's resolved severity scale (default: the built-in one)."""
    mc.fetch_pr_diff.return_value = diff
    mc.get_pr_diff_head_sha.return_value = head
    mc.get_pr_head_sha.return_value = head
    payload = {"repo": "owner/repo", "sender": "alice", "pr_number": 42,
               "pr_title": "PR #42", "pr_url": "https://git/pulls/42",
               "head_sha": head, "head_ref": "feature", "base_ref": "main"}
    with contextlib.ExitStack() as stack:
        mock_review = stack.enter_context(patch("raven.server.review_diff", side_effect=script))
        stack.enter_context(patch("raven.server.notify"))
        stack.enter_context(patch("raven.server._wait_for_ci", return_value="success"))
        stack.enter_context(patch("raven.server.RAVEN_REVIEW_MODE", mode))
        stack.enter_context(patch("raven.server.RAVEN_REVIEW_OUTPUT", output))
        if scale is not None:
            stack.enter_context(patch("raven.server._fetch_severity_scale",
                                      return_value=scale))
        _process_pr(mc, payload)
    return mock_review


def _f(file, line, message, severity="medium"):
    return {"severity": severity, "file": file, "line": line, "message": message}


def _review(findings=(), answer=None, severity=None):
    """review_diff's result shape. Like review_diff, the top-level severity
    is derived from the findings unless a test pins it (the server trusts
    it for the fresh findings)."""
    if severity is None:
        order = ["low", "medium", "high"]
        severity = max((f["severity"] for f in findings), key=order.index, default="low")
    r = {"severity": severity, "summary": "ok", "findings": list(findings)}
    if answer is not None:
        r["prior_answer"] = answer
    return r


def _tracked() -> set[int]:
    entry = _previous_diffs.get(KEY)
    if entry is None:
        return set()
    return {f["comment_id"] for fl in entry.findings.values() for f in fl
            if f.get("comment_id") is not None}


def _cached(message: str) -> dict:
    [f] = [f for fl in _previous_diffs[KEY].findings.values() for f in fl
           if f.get("message") == message]
    return f


H = [str(i) * 40 for i in range(1, 10)]


class _InlineExecutor:
    """Runs a submitted task at once, so a merge dispatched by an approving
    pass finishes inside _run_pass, while _wait_for_ci is still patched."""

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        pass


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # Same guards as test_server.py's autouse fixtures: no merge on the
    # real CI-wait pool, and no parked re-run leaking into a later test.
    monkeypatch.setattr(_server_mod, "ci_wait_executor", _InlineExecutor())
    for store in (_previous_diffs, _recent_prs, _server_mod._rerun_requested,
                  _server_mod._in_progress_heads):
        store.clear()
    yield
    for store in (_previous_diffs, _recent_prs, _server_mod._rerun_requested,
                  _server_mod._in_progress_heads):
        store.clear()


def _open_pr(fake, mc):
    """Pass 1: a.py:2 and b.py:1 each get a finding thread."""
    _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"], "b.py": ["z = 3"]}), H[0],
              lambda d, r, **kw: _review([_f("a.py", 2, "a is wrong", "high"),
                                          _f("b.py", 1, "b is wrong")]))
    assert len(fake.open_ids()) == 2


A2 = {"a.py": ["x = 1", "y = 22", "w = 0"], "b.py": ["z = 3"]}  # a.py changed


class TestCollectPriorFindings:
    SCALE = default_scale()

    def _collect(self, threads, cached, scope, removed=(), resolved=(), gaps=(), cap=30):
        return _collect_prior_findings(threads, cached, set(scope), set(removed),
                                       set(resolved), set(gaps), self.SCALE, cap)

    def test_tracked_finding_is_the_cache_dict(self):
        f = {**_f("a.py", 2, "m"), "comment_id": 7}
        p = self._collect([{"comment_id": 7, "file": "a.py", "line": 9, "body": "x",
                            "replies": 3, "resolved": False}], {"a.py": [f]}, {"a.py"})
        assert p.findings == [f] and p.findings[0] is f
        assert p.replies == [3] and p.tracked == {id(f)} and p.untracked_acted == 0

    def test_tracked_finding_missing_from_listing_is_still_offered(self):
        f = {**_f("a.py", 2, "m"), "comment_id": 7}
        p = self._collect([], {"a.py": [f]}, {"a.py"})
        assert p.findings == [f]

    def test_resolved_tracked_finding_is_not_offered(self):
        f = {**_f("a.py", 2, "m"), "comment_id": 7}
        listed = [{"comment_id": 7, "file": "a.py", "line": 2, "body": "", "replies": 0,
                   "resolved": True}]
        assert self._collect(listed, {"a.py": [f]}, {"a.py"}).findings == []
        assert self._collect(None, {"a.py": [f]}, {"a.py"}, resolved={7}).findings == []

    def test_orphan_in_scope_is_parsed_and_offered(self):
        listed = [{"comment_id": 9, "file": "a.py", "line": 4,
                   "body": "🔴 **[high]** orphan", "replies": 1, "resolved": False}]
        p = self._collect(listed, {}, {"a.py"})
        assert p.findings == [{"severity": "high", "file": "a.py", "line": 4,
                               "message": "orphan", "comment_id": 9}]
        assert p.replies == [1] and p.tracked == set() and p.untracked_acted == 1

    def test_orphan_outside_scope_is_left_alone_and_not_counted(self):
        """Counted only when acted on: a left-alone thread would be
        recounted by every review of the PR, so the rate never fell."""
        listed = [{"comment_id": 9, "file": "other.py", "line": 4,
                   "body": "🔴 **[high]** orphan", "replies": 0, "resolved": False}]
        p = self._collect(listed, {}, {"a.py"}, removed={"gone.py"})
        assert p.findings == [] and p.moot == [] and p.untracked_acted == 0

    def test_non_finding_body_is_left_alone_and_not_counted(self):
        listed = [{"comment_id": 9, "file": "a.py", "line": 4, "body": "hand-edited note",
                   "replies": 0, "resolved": False}]
        p = self._collect(listed, {}, {"a.py"}, removed={"a.py"})
        assert p.findings == [] and p.moot == [] and p.untracked_acted == 0

    def test_moot_orphan_is_counted(self):
        listed = [{"comment_id": 9, "file": "gone.py", "line": 4,
                   "body": "🟡 **[low]** orphan", "replies": 0, "resolved": False}]
        p = self._collect(listed, {}, {"a.py"}, removed={"gone.py"})
        assert [m["comment_id"] for m in p.moot] == [9] and p.untracked_acted == 1

    def test_removed_files_are_moot(self):
        tracked = {**_f("gone.py", 1, "t"), "comment_id": 5}
        threadless = _f("gone.py", 2, "no thread")
        listed = [{"comment_id": 9, "file": "gone.py", "line": 3,
                   "body": "🟡 **[low]** orphan", "replies": 0, "resolved": False}]
        p = self._collect(listed, {"gone.py": [tracked, threadless]}, {"a.py"},
                          removed={"gone.py"})
        assert p.findings == []
        assert [m.get("comment_id") for m in p.moot] == [5, 9]

    def test_gap_markers_are_never_offered(self):
        marker = {"severity": "high", "file": "a.py", "message": "⚠️ gap", "gap_marker": True}
        assert self._collect(None, {"a.py": [marker]}, {"a.py"}, gaps={"a.py"}).findings == []

    def test_cap_keeps_severity_then_replies(self):
        cached = {"a.py": [{**_f("a.py", i, f"m{i}", sev), "comment_id": i}
                           for i, sev in enumerate(["low", "high", "low", "medium"], 1)]}
        listed = [{"comment_id": 3, "file": "a.py", "line": 3, "body": "", "replies": 2,
                   "resolved": False}]
        p = self._collect(listed, cached, {"a.py"}, cap=3)
        assert [f["message"] for f in p.findings] == ["m2", "m3", "m4"]
        assert [f["message"] for f in p.overflow] == ["m1"]

    def test_renamed_tier_over_cap_ranks_first(self):
        """scale.rank() normalizes: a tier the scale lost (or a missing
        severity) ranks as the most severe, so it is offered and re-judged
        instead of overflowing; it never raises."""
        cached = {"a.py": [
            {**_f("a.py", 1, "known-low", "low"), "comment_id": 1},
            {**_f("a.py", 2, "renamed", "critical"), "comment_id": 2},
            {"file": "a.py", "line": 3, "message": "no-severity", "comment_id": 3},
        ]}
        p = self._collect(None, cached, {"a.py"}, cap=2)
        assert sorted(f["message"] for f in p.findings) == ["no-severity", "renamed"]
        assert [f["message"] for f in p.overflow] == ["known-low"]


class TestPriorFindingsFlow:
    def test_prior_findings_offered_with_replies(self):
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        fake.threads[fake.cid_for("a is wrong")]["replies"] = 4
        mock_review = _run_pass(mc, _diff(A2), H[1], lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {0: None}}))
        [prior] = mock_review.call_args.kwargs["prior_findings"]
        assert prior == {"severity": "high", "file": "a.py", "line": 2,
                         "message": "a is wrong", "replies": 4}

    def test_kept_prior_stays_on_its_thread(self):
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        before = set(fake.threads)
        _run_pass(mc, _diff(A2), H[1],
                  lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {0: 3}}))
        assert set(fake.threads) == before and fake.retracted == []
        kept = _cached("a is wrong")
        assert kept["line"] == 3 and kept["comment_id"] == fake.cid_for("a is wrong")

    def test_superseded_prior_is_resolved_after_submit(self):
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        a = fake.cid_for("a is wrong")
        _run_pass(mc, _diff(A2), H[1], lambda d, r, **kw: _review(
            [_f("a.py", 3, "a is still wrong, reworded")], {"answered": {0}, "kept": {}}))
        assert fake.retracted == [a] and a not in fake.open_ids()
        assert fake.open_ids() == _tracked()

    @pytest.mark.parametrize("answer", [None, {"answered": set(), "kept": {}}])
    def test_unanswered_keeps_every_prior_and_resolves_nothing(self, answer):
        """The prompt tells the model not to restate what it keeps, so a
        lost answer (missing, voided, or a failed chunk) must keep the
        prior blocker in the verdict and the cache, never drop it."""
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0],
                  lambda d, r, **kw: _review([_f("a.py", 2, "blocker", "high")]), mode="all")
        cid = fake.cid_for("blocker")
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1],
                  lambda d, r, **kw: _review(answer=answer), mode="all")
        assert fake.retracted == [] and cid in _tracked()
        assert mc.submit_review.call_args.kwargs["approve"] is False
        assert fake.open_ids() == _tracked()

    def test_overflow_is_kept_not_resolved(self):
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0], lambda d, r, **kw: _review(
            [_f("a.py", 1, "offered", "high"), _f("a.py", 2, "overflow", "low")]))
        over = fake.cid_for("overflow")
        with patch("raven.server.RAVEN_PRIOR_FINDINGS_MAX", 1):
            mock_review = _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1],
                                    lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {}}))
        assert [p["message"] for p in mock_review.call_args.kwargs["prior_findings"]] == ["offered"]
        assert over in _tracked() and over not in fake.retracted
        assert fake.open_ids() == _tracked()

    def test_resolved_mid_review_restatement_still_posts(self):
        """The dedupe runs after the post-review resolved filter: a prior
        resolved during the review is dropped, so the model's word-for-word
        restatement must post rather than vanish with it."""
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        a = fake.cid_for("a is wrong")

        def resolve_and_restate(d, r, **kw):
            fake.threads[a]["resolved"] = True
            return _review([_f("a.py", 2, "a is wrong", "high")],
                           {"answered": {0}, "kept": {}})
        _run_pass(mc, _diff(A2), H[1], resolve_and_restate)
        assert a not in _tracked()
        assert any(t["body"].endswith("a is wrong") and cid != a
                   for cid, t in fake.threads.items())
        assert fake.open_ids() == _tracked()

    def test_renamed_tier_fails_closed(self):
        """A cached tier the current scale no longer has (a severities.json
        rename) reads as the most severe tier: an unanswered prior blocker
        must not start counting as the least severe and approve."""
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0],
                  lambda d, r, **kw: _review([_f("a.py", 2, "a is wrong", "high")]), mode="all")
        renamed = SeverityScale(ranks={"nit": 10, "bug": 20, "blocker": 30},
                                blocks_at_or_above="bug")
        mock_review = _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1],
                                lambda d, r, **kw: _review(severity="nit"),
                                mode="all", scale=renamed)
        [prior] = mock_review.call_args.kwargs["prior_findings"]
        assert prior["severity"] == "blocker"
        assert _cached("a is wrong")["severity"] == "blocker"
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_severity_raised_restatement_posts_fresh(self):
        """Severity is part of the dedupe key: a restatement at a raised
        tier is the 'raise it fresh' case, so the old low copy must not
        stand in for it."""
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0],
                  lambda d, r, **kw: _review([_f("a.py", 2, "a is wrong", "low")]), mode="all")
        low = fake.cid_for("a is wrong")
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1], lambda d, r, **kw: _review(
            [_f("a.py", 2, "a is wrong", "high")], {"answered": {0}, "kept": {}}), mode="all")
        assert low in fake.retracted
        assert _cached("a is wrong")["severity"] == "high"
        assert mc.submit_review.call_args.kwargs["approve"] is False
        assert fake.open_ids() == _tracked()

    def test_failed_submit_resolves_nothing_and_keeps_cache(self):
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        tracked = _tracked()
        fake.fail_submit = True
        _run_pass(mc, _diff(A2), H[1], lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {}}))
        assert fake.retracted == [] and _tracked() == tracked

    def test_kept_prior_counts_toward_the_verdict(self):
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0],
                  lambda d, r, **kw: _review([_f("a.py", 2, "blocker", "high")]), mode="all")
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1],
                  lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {0: None}}), mode="all")
        assert mc.submit_review.call_args.kwargs["approve"] is False

    def test_verbatim_restatement_counts_as_keep(self):
        """Same (severity, file, line, message) as a prior: a keep, even
        when the answer omits it, and the prior copy keeps its thread."""
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        before = set(fake.threads)
        _run_pass(mc, _diff(A2), H[1], lambda d, r, **kw: _review(
            [_f("a.py", 2, "a is wrong", "high")], {"answered": {0}, "kept": {}}))
        assert set(fake.threads) == before and fake.retracted == []
        assert fake.cid_for("a is wrong") in _tracked()

    def test_threads_none_falls_back_to_cache(self):
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        legacy = fake.open_thread("a.py", 2, "🔴 **[high]** a is wrong")
        mc.get_review_threads.side_effect = None
        mc.get_review_threads.return_value = None
        mock_review = _run_pass(mc, _diff(A2), H[1],
                                lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {}}))
        assert len(mock_review.call_args.kwargs["prior_findings"]) == 1
        assert legacy in fake.open_ids()  # an orphan is never touched on the fallback

    def test_summary_mode_resolves_nothing(self):
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0],
                  lambda d, r, **kw: _review([_f("a.py", 2, "a is wrong")]), output="summary")
        _run_pass(mc, _diff(A2), H[1],
                  lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {}}), output="summary")
        assert fake.threads == {} and mc.retract_finding.call_count == 0

    def test_inline_mode_body_names_a_kept_blocker(self):
        """Inline-only output posts no summary, and a kept prior isn't
        posted again on its line, so the minimal body is the only part of
        the pass that names the blocker it still counts."""
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0],
                  lambda d, r, **kw: _review([_f("a.py", 2, "kept blocker on a", "high")]),
                  mode="all", output="inline")
        cid = fake.cid_for("kept blocker on a")
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1],
                  lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {0: None}}),
                  mode="all", output="inline")
        submitted = mc.submit_review.call_args
        assert submitted.kwargs["approve"] is False
        assert submitted.kwargs["inline_comments"] == []
        assert "kept blocker on a" in submitted.args[2]
        assert fake.open_ids() == _tracked() == {cid}


class TestMultiPass:
    """A PR walked through the pushes that produced duplicate threads in
    the wild. After every answered pass: open Raven threads == tracked
    findings with a comment_id."""

    def _invariant(self, fake):
        assert fake.open_ids() == _tracked()

    def test_the_duplicate_sequence_keeps_one_thread_per_open_finding(self):
        fake = FakePlatform(); mc = _provider(fake)
        # 1. open: a.py:2 and b.py:1
        _open_pr(fake, mc)
        self._invariant(fake)
        a, b = fake.cid_for("a is wrong"), fake.cid_for("b is wrong")

        # 2. a.py changes; the model keeps its finding (b.py is carried)
        def keep_a(d, r, **kw):
            assert [p["message"] for p in kw["prior_findings"]] == ["a is wrong"]
            assert [c["message"] for c in kw["carried_findings"]] == ["b is wrong"]
            return _review(answer={"answered": {0}, "kept": {0: 3}})
        _run_pass(mc, _diff(A2), H[1], keep_a)
        assert fake.open_ids() == {a, b}
        self._invariant(fake)

        # 3. b.py leaves the PR and c.py arrives: full review; b.py is moot
        def full_1(d, r, **kw):
            assert [p["message"] for p in kw["prior_findings"]] == ["a is wrong"]
            return _review([_f("c.py", 1, "c is wrong")], {"answered": {0}, "kept": {0: None}})
        _run_pass(mc, _diff({"a.py": A2["a.py"], "c.py": ["q = 1"]}), H[2], full_1)
        c = fake.cid_for("c is wrong")
        assert b not in fake.open_ids() and fake.open_ids() == {a, c}
        self._invariant(fake)

        # 4. a force-push that is never reviewed drops c.py; the next push
        #    (a.py changed again) is compared with pass 3's cache, so c.py
        #    is a removed file and its thread is moot. The model restates
        #    a.py's finding in new words instead of keeping it.
        def full_2(d, r, **kw):
            return _review([_f("a.py", 1, "a is still wrong, reworded", "high")],
                           {"answered": {0}, "kept": {}})
        _run_pass(mc, _diff({"a.py": ["x = 2", "y = 22", "w = 0"]}), H[4], full_2)
        a2 = fake.cid_for("a is still wrong, reworded")
        assert fake.open_ids() == {a2}
        self._invariant(fake)

        # 5. the cache is wiped; two legacy copies of the finding are open
        #    too. The full review collapses them onto the one with replies.
        _previous_diffs.clear()
        body = "🔴 **[high]** a is still wrong, reworded"
        fake.open_thread("a.py", 1, body)
        talked = fake.open_thread("a.py", 1, body, replies=2)

        def full_3(d, r, **kw):
            priors = kw["prior_findings"]
            assert len(priors) == 3
            keep = [i for i, p in enumerate(priors) if p["replies"] == 2]
            return _review(answer={"answered": set(range(3)), "kept": {keep[0]: None}})
        before = set(fake.threads)
        _run_pass(mc, _diff({"a.py": ["x = 2", "y = 22", "w = 0"]}), H[5], full_3)
        assert set(fake.threads) == before  # nothing new posted
        assert fake.open_ids() == {talked}
        self._invariant(fake)

    def test_unanswered_pass_keeps_both_then_the_next_answer_collapses(self):
        """No answer keeps the prior (fail-safe) next to a reworded fresh
        copy; the invariant still holds, and the next answered pass
        collapses them."""
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        a = fake.cid_for("a is wrong")
        _run_pass(mc, _diff(A2), H[1], lambda d, r, **kw: _review(
            [_f("a.py", 3, "a reworded", "high")]))  # no answer: prior kept, nothing resolved
        assert a in fake.open_ids() and a in _tracked()
        self._invariant(fake)

        def collapse(d, r, **kw):
            msgs = sorted(p["message"] for p in kw["prior_findings"])
            assert msgs == ["a is wrong", "a reworded"]
            keep = [i for i, p in enumerate(kw["prior_findings"]) if p["message"] == "a reworded"]
            return _review(answer={"answered": {0, 1}, "kept": {keep[0]: None}})
        _run_pass(mc, _diff({"a.py": ["x = 9", "y = 22", "w = 0"], "b.py": ["z = 3"]}), H[2], collapse)
        assert a not in fake.open_ids()
        self._invariant(fake)

    def test_threadless_kept_prior_is_posted_once_and_tracked(self):
        """A kept prior with no thread (first seen in summary mode) is the
        one kept prior _posts_inline posts. Its new comment_id must land
        on the cached copy, or the next pass would offer the copy and its
        own thread (as an orphan) side by side."""
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0],
                  lambda d, r, **kw: _review([_f("a.py", 2, "a is wrong")]), output="summary")
        assert fake.threads == {}
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1],
                  lambda d, r, **kw: _review(answer={"answered": {0}, "kept": {0: None}}))
        [cid] = fake.open_ids()
        assert _cached("a is wrong")["comment_id"] == cid
        self._invariant(fake)

        def third(d, r, **kw):
            assert len(kw["prior_findings"]) == 1  # the tracked copy, no orphan twin
            return _review(answer={"answered": {0}, "kept": {0: None}})
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 4"]}), H[2], third)
        assert fake.open_ids() == {cid}
        self._invariant(fake)

    def test_concurrent_retraction_is_not_resurrected(self):
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        a = fake.cid_for("a is wrong")

        def retract_mid_review(d, r, **kw):
            entry = _previous_diffs[KEY]
            entry.findings = {k: [f for f in v if f.get("comment_id") != a]
                              for k, v in entry.findings.items()}
            fake.threads[a]["resolved"] = True  # the comment flow resolved it
            return _review(answer={"answered": {0}, "kept": {0: None}})
        _run_pass(mc, _diff(A2), H[1], retract_mid_review)
        assert a not in _tracked() and a not in fake.retracted
        self._invariant(fake)

    def test_concurrent_retraction_with_an_open_thread_is_not_resurrected(self):
        """The live-entry filter alone: the retraction's resolve failed (or
        the platform can't resolve), so the thread stays open and the
        post-review resolved set doesn't name it. Deleting the kept_live
        filter must fail this test. The open-threads invariant can't hold
        here by design (the thread is open, the finding is retracted), so
        assert on the cache directly."""
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        a = fake.cid_for("a is wrong")

        def retract_mid_review(d, r, **kw):
            entry = _previous_diffs[KEY]
            entry.findings = {k: [f for f in v if f.get("comment_id") != a]
                              for k, v in entry.findings.items()}
            return _review(answer={"answered": {0}, "kept": {0: None}})
        _run_pass(mc, _diff(A2), H[1], retract_mid_review)
        assert a not in _tracked()
        assert a in fake.open_ids() and a not in fake.retracted

    def test_user_resolve_during_review_wins(self):
        fake = FakePlatform(); mc = _provider(fake); _open_pr(fake, mc)
        a = fake.cid_for("a is wrong")

        def resolve_mid_review(d, r, **kw):
            fake.threads[a]["resolved"] = True
            return _review(answer={"answered": {0}, "kept": {0: None}})
        _run_pass(mc, _diff(A2), H[1], resolve_mid_review)
        assert a not in _tracked() and a not in fake.retracted
        self._invariant(fake)

    def test_two_findings_on_one_line_stay_separate(self):
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 2"]}), H[0], lambda d, r, **kw: _review(
            [_f("a.py", 2, "first issue"), _f("a.py", 2, "second issue")]))
        _run_pass(mc, _diff({"a.py": ["x = 1", "y = 3"]}), H[1], lambda d, r, **kw: _review(
            answer={"answered": {0, 1}, "kept": {0: None, 1: None}}))
        assert len(fake.open_ids()) == 2 and fake.retracted == []
        self._invariant(fake)


def _modify(src: str, dst: str | None = None) -> str:
    """A base file edited by the PR (``y = 1`` -> ``y = 2``); renamed to
    ``dst`` when given. A PR diff shows a rename only for a file the base
    branch has, so the rename scenario starts from an edit, not an add."""
    head = (f"diff --git a/{src} b/{dst}\nsimilarity index 80%\n"
            f"rename from {src}\nrename to {dst}\n") if dst else f"diff --git a/{src} b/{src}\n"
    return head + f"--- a/{src}\n+++ b/{dst or src}\n@@ -1,1 +1,1 @@\n-y = 1\n+y = 2\n"


class TestRenamedBetweenPasses:
    """A rename is not a file leaving the PR: the code still ships under
    the target, so the source's threads are offered as priors on the
    target (the grounding filter maps renamed-away paths the same way),
    never resolved as moot."""

    def test_collect_offers_a_renamed_sources_findings_on_the_target(self):
        f = {**_f("old.py", 1, "m"), "comment_id": 7}
        orphan = {"comment_id": 9, "file": "old.py", "line": 2,
                  "body": "🟡 **[low]** orphan", "replies": 0, "resolved": False}
        p = _collect_prior_findings([orphan], {"old.py": [f]}, {"new.py"}, {"old.py"},
                                    set(), set(), default_scale(), 30,
                                    renamed={"old.py": "new.py"})
        assert p.moot == []
        assert [x.get("comment_id") for x in p.findings] == [7, 9]
        assert p.moved[id(f)] == "new.py"
        assert [x for x in p.findings if x.get("comment_id") == 9][0]["file"] == "new.py"

    def test_rename_between_passes_keeps_the_thread(self):
        fake = FakePlatform(); mc = _provider(fake)
        _run_pass(mc, _diff({"a.py": ["x = 1"]}) + _modify("old.py"), H[0],
                  lambda d, r, **kw: _review([_f("old.py", 1, "old is wrong")]))
        cid = fake.cid_for("old is wrong")

        def keep(d, r, **kw):
            [prior] = kw["prior_findings"]
            assert prior["file"] == "new.py" and prior["message"] == "old is wrong"
            return _review(answer={"answered": {0}, "kept": {0: None}})
        _run_pass(mc, _diff({"a.py": ["x = 1"]}) + _modify("old.py", "new.py"), H[1], keep)
        assert fake.retracted == [] and cid in fake.open_ids()
        entry = _previous_diffs[KEY]
        assert [f["comment_id"] for f in entry.findings.get("new.py", [])] == [cid]
        assert fake.open_ids() == _tracked()

