Copies of the audit's repro fixtures (docs/superpowers/research/2026-09-27-audit-repros/fixtures/).
The tests read these copies because scripts/push-to-github.sh strips docs/superpowers/ from the public mirror, where the originals don't exist.

`bbdc_binary_shapes.json` and `bbdc_binary_deleted.json` are real Bitbucket Data Center JSON diffs (`compare/diff`, captured 2026-10-01), trimmed to `diffs`. They diff synthetic probe files on throwaway branches: a modified binary, a binary source file, a PNG, an empty new file, a pure rename and a hunk-less same-path change; then a deleted binary. BB DC marks a binary entry `"binary": true` with no hunks.

`bbdc_cut_lines.json.gz` is a real Bitbucket Data Center JSON diff (`compare/diff`, captured 2026-10-01), trimmed to `diffs`. It diffs a synthetic probe file whose two new lines are over 6,000 characters long. Bitbucket cut each at 5,000 and flagged the line `"truncated": true`; nothing above the line level was flagged. It is gzipped because its own JSON lines are over 5,000 characters: Bitbucket would cut them in a PR's diff, so Raven could not review the PR that changes it.
