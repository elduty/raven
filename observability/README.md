# Raven observability bundle

Permanent metrics storage + a pre-built dashboard for Raven, as an opt-in
Docker Compose profile. Prometheus scrapes Raven's `/metrics`, retains ~5
years of history, and Grafana serves a provisioned dashboard.

Design: [`docs/superpowers/specs/2026-05-29-observability-stack-design.md`](../docs/superpowers/specs/2026-05-29-observability-stack-design.md).

## Quick start

```bash
# In .env (same file Raven reads):
#   RAVEN_METRICS_TOKEN=<a long random token>   # required — gates /metrics
#   GRAFANA_ADMIN_PASSWORD=<your password>      # Grafana admin login
#   GRAFANA_PORT=3000                           # optional, default 3000
#   GRAFANA_BIND=127.0.0.1                      # optional, default localhost-only

docker compose --profile observability up -d
```

Then open Grafana at `http://<host>:3000`, log in as `admin` /
`$GRAFANA_ADMIN_PASSWORD`, and the **Raven — Review, Reliability & Cost**
dashboard is already there (folder *Raven*), wired to Prometheus.

By default Grafana binds to `127.0.0.1` (loopback) so the `admin` fallback
password is never reachable off-host. Any non-loopback bind (`GRAFANA_BIND=0.0.0.0`,
`::`, or a routable IP like `192.168.1.5`) **requires** a non-default
`GRAFANA_ADMIN_PASSWORD`: the Grafana container refuses to start (fails fast,
non-zero exit) otherwise, so an exposed-but-unprotected dashboard can't happen
by accident.

Note that Grafana serves **plain HTTP**. A direct non-loopback bind therefore
sends the admin password and session cookies in cleartext — fine on a trusted
network or over a localhost tunnel, but for real off-host access put Grafana
behind a **TLS-terminating reverse proxy** (and still set a strong password).
The `GRAFANA_BIND` knob lowers the friction of exposing the port; it does not
add transport security.

Plain `docker compose up -d` (no `--profile`) still runs Raven alone — the
Prometheus and Grafana services do not start without the profile flag.

## What's in the dashboard

- **Throughput & verdicts** — reviews/min by severity, merges, CI failures, skips by reason.
- **Reliability & latency** — average review duration (mean, not p95 — see below), errors/min by type, comment activity (responses / retractions / verdict revisions).
- **Cost & tokens** — cost over the range, cost/hour by model, tokens/hour by kind, cost by repo.

A `repo` template variable (top-left) filters every panel; "All" shows the fleet.

## How the metrics token reaches Prometheus

`/metrics` is bearer-gated and Prometheus config has no env-var substitution.
At boot the prometheus service writes `$RAVEN_METRICS_TOKEN` to
`/tmp/raven_metrics_token` (busybox `sh`/`printf` in the image), and
`prometheus.yml` references it via `authorization.credentials_file`. The token
lives only in `.env`; nothing secret is committed. If the token is unset,
Prometheus starts but the `raven` target shows **DOWN** (404) rather than
silently scraping nothing.

## Monitoring a second Raven instance

`prometheus.yml` ships a commented `raven-bbdc` job showing how to scrape a
remote instance (its own URL + its own token file). Uncomment, set the target
and a second credentials file, and restart Prometheus.

## Retention

Prometheus runs with `--storage.tsdb.retention.time=1825d` (5 years) on the
`raven-prom-data` named volume. At Raven's cardinality (~thousands of series)
that's a few hundred MB. Adjust the flag in `docker-compose.yml` to taste.

## Known limitation: latency is an average, not a percentile

`raven_review_duration_seconds` is a Prometheus *summary* (sum + count), which
carries no quantiles — so the dashboard shows mean duration. True p95/p99
would require converting it to a histogram on the Raven side (a metrics-layer
change, deliberately out of scope for this bundle).

## Licensing

- **Prometheus** — Apache-2.0.
- **Grafana** — AGPLv3. Used here as the stock, unmodified upstream image for
  internal dashboard viewing; the AGPL network-service source obligation
  applies only if you modify Grafana's source *and* offer it as a service to
  third parties. The dashboard JSON and configs in this directory are
  configuration/data, not Grafana source.

## Verify it's working

```bash
docker compose --profile observability config >/dev/null && echo "compose valid"
docker compose --profile observability up -d
# Prometheus targets: http://<host>:9090 isn't published by default; check via Grafana
# or temporarily add a ports mapping. The `raven` target should be UP once a
# token is set. Trigger a review and the dashboard panels populate within a
# scrape interval (30s).
```
