"""reviewer.py — Runs claude CLI with a diff and parses the JSON response."""

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

CLAUDE_BIN = "/usr/bin/claude"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "max")
MAX_CONCURRENT_CLAUDE = max(int(os.environ.get("RAVEN_MAX_CONCURRENT_CLAUDE", "4")), 1)
_claude_semaphore = threading.Semaphore(MAX_CONCURRENT_CLAUDE)

# Load review prompt from prompts/review.md (relative to this package)
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "review.md"

def _load_review_prompt() -> str:
    """Load the review prompt template from prompts/review.md."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("prompts/review.md not found — using fallback prompt")
        return ""

_REVIEW_PROMPT_TEMPLATE = _load_review_prompt()

_RESPOND_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "respond.md"

def _load_respond_prompt() -> str:
    try:
        return _RESPOND_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "You are Raven, an AI code reviewer. Respond helpfully and concisely."

_RESPOND_PROMPT_TEMPLATE = _load_respond_prompt()

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def review_config_hash() -> str:
    """SHA256 of model + effort + prompt — changes when review config changes."""
    content = f"{CLAUDE_MODEL}:{CLAUDE_EFFORT}:{_REVIEW_PROMPT_TEMPLATE}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

# Binary / lock file extensions and names to strip from diffs
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".svg", ".tiff", ".tif", ".mp4", ".mp3", ".wav", ".ogg",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".a", ".o", ".pyc",
    ".woff", ".woff2", ".ttf", ".eot",
}
SKIP_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "poetry.lock",
    "Gemfile.lock",
    "composer.lock",
    "cargo.lock",
}
SKIP_SUFFIX_PATTERNS = [".lock"]


def _strip_lockfiles_and_binaries(diff: str) -> str:
    """Remove binary and lockfile sections from a unified diff."""
    lines = diff.splitlines(keepends=True)
    output: list[str] = []
    skip = False

    for line in lines:
        if line.startswith("diff --git "):
            # Determine if this file section should be skipped
            # e.g. "diff --git a/yarn.lock b/yarn.lock"
            parts = line.split(" ")
            filename = parts[-1].strip()
            # remove b/ prefix
            if filename.startswith("b/"):
                filename = filename[2:]
            basename = os.path.basename(filename)
            _, ext = os.path.splitext(basename)
            skip = (
                basename.lower() in SKIP_FILENAMES
                or ext.lower() in SKIP_EXTENSIONS
                or any(filename.endswith(p) for p in SKIP_SUFFIX_PATTERNS)
            )
            if not skip:
                output.append(line)
        elif line.startswith("Binary files"):
            # Always skip binary file lines
            skip = True
        else:
            if not skip:
                output.append(line)

    return "".join(output)


MAX_DIFF_LINES = int(os.environ.get("MAX_DIFF_LINES", "3000"))


def split_diff_by_file(diff: str) -> list[tuple[str, str]]:
    """Split a unified diff into (filename, chunk) pairs, one per file."""
    chunks: list[tuple[str, str]] = []
    current_file = None
    current_lines: list[str] = []

    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current_file and current_lines:
                chunks.append((current_file, "".join(current_lines)))
            parts = line.split(" ")
            filename = parts[-1].strip()
            if filename.startswith("b/"):
                filename = filename[2:]
            current_file = filename
            current_lines = [line]
        else:
            if current_lines is not None:
                current_lines.append(line)

    if current_file and current_lines:
        chunks.append((current_file, "".join(current_lines)))

    return chunks


def review_diff(diff: str, repo_name: str, claude_md: str = "", file_contents: dict[str, str] | None = None) -> dict:
    """Run claude CLI against the diff and return a structured review dict.

    For large diffs (> MAX_DIFF_LINES), splits by file and reviews each chunk
    separately, then merges findings into a single result.

    Returns:
        {
            "severity": "low"|"medium"|"high",
            "summary": str,
            "findings": [{"severity": ..., "message": ...}, ...],
            "chunked": bool,  # True if diff was split across multiple reviews
            "chunks_reviewed": int,
        }
    Raises:
        RuntimeError if claude exits non-zero or output cannot be parsed.
    """
    clean_diff = _strip_lockfiles_and_binaries(diff)
    line_count = clean_diff.count("\n")

    if line_count <= MAX_DIFF_LINES:
        result = _review_single_chunk(clean_diff, repo_name, claude_md, file_contents=file_contents)
        result["chunked"] = False
        result["chunks_reviewed"] = 1
        return result

    # Split by file and review each chunk
    file_chunks = split_diff_by_file(clean_diff)
    logger.info(
        "Diff too large (%d lines), splitting into %d file chunks for %s",
        line_count, len(file_chunks), repo_name,
    )

    all_findings: list[dict] = []
    max_severity = "low"
    summaries: list[str] = []
    errors: list[str] = []
    reviewed_count = 0
    # Filter out oversized chunks before dispatching
    reviewable = []
    for filename, chunk in file_chunks:
        chunk_lines = chunk.count("\n")
        if chunk_lines > MAX_DIFF_LINES * 3:
            logger.warning("Skipping oversized single-file chunk: %s (%d lines)", filename, chunk_lines)
            errors.append(f"`{filename}` skipped (too large: {chunk_lines} lines)")
        else:
            reviewable.append((filename, chunk))

    # Review chunks in parallel (bounded by _claude_semaphore)
    def _review_chunk(filename: str, chunk: str) -> tuple[str, dict | None, str | None]:
        try:
            chunk_files = {filename: file_contents[filename]} if file_contents and filename in file_contents else None
            result = _review_single_chunk(chunk, repo_name, claude_md, filename_hint=filename, file_contents=chunk_files)
            if result.get("_parse_error"):
                return filename, None, f"`{filename}` review output could not be parsed"
            return filename, result, None
        except Exception as e:
            logger.error("Chunk review failed for %s: %s", filename, e)
            return filename, None, f"`{filename}` review failed: {e}"

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CLAUDE) as chunk_pool:
        futures = {chunk_pool.submit(_review_chunk, fn, ch): fn for fn, ch in reviewable}
        for future in as_completed(futures):
            filename, chunk_result, error = future.result()
            if error:
                errors.append(error)
                continue
            reviewed_count += 1
            all_findings.extend(chunk_result["findings"])
            if SEVERITY_ORDER.get(chunk_result["severity"], 0) > SEVERITY_ORDER.get(max_severity, 0):
                max_severity = chunk_result["severity"]
            if chunk_result["summary"]:
                summaries.append(f"`{filename}`: {chunk_result['summary']}")

    if errors:
        for err in errors:
            all_findings.append({"severity": "low", "message": f"⚠️ {err}"})

    merged_summary = "; ".join(summaries[:3])
    if len(summaries) > 3:
        merged_summary += f" (+{len(summaries) - 3} more files)"

    # If no chunks were successfully reviewed, flag as parse error to block auto-merge
    if reviewed_count == 0 and file_chunks:
        logger.warning("All %d chunks failed for %s — flagging as parse error", len(file_chunks), repo_name)
        return {
            "severity": "high",
            "summary": "All review chunks failed — no files could be reviewed.",
            "findings": all_findings,
            "chunked": True,
            "chunks_reviewed": 0,
            "_parse_error": True,
        }

    result = {
        "severity": max_severity,
        "summary": merged_summary or "Multi-file review completed.",
        "findings": all_findings,
        "chunked": True,
        "chunks_reviewed": reviewed_count,
    }
    return result


def _review_single_chunk(diff: str, repo_name: str, claude_md: str = "", filename_hint: str = "",
                          file_contents: dict[str, str] | None = None) -> dict:
    """Review a single diff chunk with claude CLI."""
    file_context = f" (file: `{filename_hint}`)" if filename_hint else ""
    repo_context = f"\n\n## Repository Context\n{claude_md}" if claude_md else ""

    # Build full file context section
    files_section = ""
    if file_contents:
        parts = []
        for path, content in file_contents.items():
            parts.append(f"### `{path}`\n```\n{content}\n```")
        files_section = "\n\n## Full File Contents (for context — review the diff, not these files)\n\n" + "\n\n".join(parts)

    # Use the loaded prompt template, or fall back to a minimal inline prompt
    if _REVIEW_PROMPT_TEMPLATE:
        prompt = (
            f"## Repository: {repo_name}{file_context}{repo_context}\n\n"
            f"{_REVIEW_PROMPT_TEMPLATE}\n\n"
            f"## Diff to Review\n\n```diff\n{diff}\n```"
            f"{files_section}"
        )
    else:
        prompt = (
            f"You are a senior engineer reviewing a code diff for {repo_name}{file_context}.{repo_context}\n\n"
            f"Review this diff and respond with ONLY valid JSON:\n"
            f'{{"severity":"low|medium|high","summary":"one sentence","findings":[{{"severity":"...","message":"..."}}]}}\n\n'
            f"Diff:\n{diff}"
            f"{files_section}"
        )

    env = os.environ.copy()

    logger.info(
        "Running claude CLI for %s%s (model=%s effort=%s diff=%d lines)",
        repo_name, f"/{filename_hint}" if filename_hint else "",
        CLAUDE_MODEL, CLAUDE_EFFORT, diff.count("\n"),
    )
    with _claude_semaphore:
        try:
            result = subprocess.run(
                [
                    CLAUDE_BIN, "-p",
                    "--model", CLAUDE_MODEL,
                    "--effort", CLAUDE_EFFORT,
                    "--output-format", "text",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("claude CLI timed out after 300s")
        except FileNotFoundError:
            raise RuntimeError(f"claude CLI not found at {CLAUDE_BIN}")

    if result.returncode != 0:
        logger.error("claude CLI stderr: %s", result.stderr[:500])
        logger.error("claude CLI stdout: %s", result.stdout[:500])
        detail = result.stderr[:200] or result.stdout[:200]
        raise RuntimeError(f"claude CLI exited with code {result.returncode}: {detail}")

    return _parse_response(result.stdout)


def _parse_response(output: str) -> dict:
    """Extract and validate the JSON review from claude's output."""
    # Try markdown fence first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _validate_review(data)
        except json.JSONDecodeError:
            pass

    # Fallback: try raw_decode from each { position
    decoder = json.JSONDecoder()
    for i, ch in enumerate(output):
        if ch == '{':
            try:
                data, _ = decoder.raw_decode(output, i)
                return _validate_review(data)
            except json.JSONDecodeError:
                continue

    logger.warning("No JSON found in claude output: %s", output[:300])
    return {
        "severity": "high",
        "summary": "Review could not be parsed from Claude output.",
        "findings": [],
        "_parse_error": True,
    }


def _validate_review(data: dict) -> dict:
    """Normalise and validate a parsed review JSON object."""
    severity = str(data.get("severity", "low")).lower()
    if severity not in SEVERITY_ORDER:
        severity = "low"

    findings = []
    for f in data.get("findings", []):
        sev = str(f.get("severity", "low")).lower()
        if sev not in SEVERITY_ORDER:
            sev = "low"
        finding = {"severity": sev, "message": str(f.get("message", ""))}
        # Pass through file/line for inline comments (optional)
        if f.get("file"):
            finding["file"] = str(f["file"])
        if isinstance(f.get("line"), int) and f["line"] > 0:
            finding["line"] = f["line"]
        findings.append(finding)

    result = {
        "severity": severity,
        "summary": str(data.get("summary", "")),
        "findings": findings,
    }
    return result


def severity_gte(a: str, b: str) -> bool:
    """Return True if severity a is >= severity b."""
    return SEVERITY_ORDER.get(a, 0) >= SEVERITY_ORDER.get(b, 0)


def respond_to_comment(comment_body: str, conversation: list[dict], diff: str,
                        repo_name: str, claude_md: str = "",
                        file_path: str = "", line: int = 0) -> str:
    """Generate a conversational response to a developer's comment.

    Returns plain text (markdown). Raises RuntimeError on CLI failure.
    """
    repo_context = f"\n\n## Repository Context\n{claude_md}" if claude_md else ""

    location = ""
    if file_path:
        location = f"\n\n## Code Location\nFile: `{file_path}`"
        if line:
            location += f", line {line}"

    conv_lines = []
    for c in conversation:
        user = c.get("user", {}).get("login", "unknown")
        body = c.get("body", "")
        conv_lines.append(f"**{user}:** {body}")
    conv_text = "\n\n".join(conv_lines)

    prompt = (
        f"## Repository: {repo_name}{repo_context}{location}\n\n"
        f"{_RESPOND_PROMPT_TEMPLATE}\n\n"
        f"## PR Diff\n\n```diff\n{diff}\n```\n\n"
        f"## Conversation\n\n{conv_text}\n\n"
        f"## Comment to respond to\n\n{comment_body}\n\n"
        f"Write your response:"
    )

    env = os.environ.copy()
    logger.info("Generating response for %s (model=%s)", repo_name, CLAUDE_MODEL)

    with _claude_semaphore:
        try:
            result = subprocess.run(
                [
                    CLAUDE_BIN, "-p",
                    "--model", CLAUDE_MODEL,
                    "--effort", CLAUDE_EFFORT,
                    "--output-format", "text",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("claude CLI timed out after 300s")
        except FileNotFoundError:
            raise RuntimeError(f"claude CLI not found at {CLAUDE_BIN}")

    if result.returncode != 0:
        logger.error("claude CLI stderr: %s", result.stderr[:500])
        raise RuntimeError(f"claude CLI exited with code {result.returncode}")

    return result.stdout.strip()
