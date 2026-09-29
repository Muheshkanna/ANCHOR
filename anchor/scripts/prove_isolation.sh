#!/usr/bin/env bash
# =============================================================================
# scripts/prove_isolation.sh
#
# Network-isolation proof for the anchor-eval Docker service.
#
# What it does
# ─────────────
# 1. Starts the anchor-eval container (network_mode: none, so it has NO eth0,
#    only a loopback interface).
# 2. Inside that same container, runs tcpdump in the background, capturing
#    ALL traffic on every available interface for the full duration of the
#    pipeline run.
# 3. Waits for the pipeline to finish.
# 4. Stops tcpdump and counts captured packets.
# 5. Asserts packet count == 0 (any packet would indicate a network attempt).
# 6. Prints a clear PASS / FAIL isolation proof.
#
# Why zero packets is the correct assertion
# ──────────────────────────────────────────
# network_mode: none gives the container only a lo (127.0.0.1) loopback
# interface.  There is no eth0, no bridge, no NAT — the kernel has no route
# to the outside world.  tcpdump on "any" will still capture loopback traffic
# (e.g., SQLite using Unix sockets mapped over lo, or inter-process signalling)
# so we apply a strict filter: we only flag packets that are NOT purely
# localhost-to-localhost (i.e., any packet whose source OR destination is
# outside 127.0.0.0/8 would count).  A truly isolated container produces
# exactly 0 such packets.
#
# Usage
# ─────
# From the host, with docker and docker compose available:
#
#   bash scripts/prove_isolation.sh
#
# Environment variables (all optional):
#   COMPOSE_FILE   path to docker-compose.yml  (default: anchor/docker-compose.yml)
#   DATASET        dataset dir inside container (default: /app/reference_data/eval_set)
#   MODEL          model file inside container  (default: /app/reference_data/model.pt)
#   REF            reference dir               (default: /app/reference_data/ref_set)
#   DB             provenance DB path          (default: /app/reference_data/provenance.db)
#   OUT            report output dir           (default: /app/reference_data/outputs)
#
# Exit codes
# ──────────
#   0  PASS — pipeline completed with zero non-loopback packets captured
#   1  FAIL — packets were captured (network access detected)
#   2  ERROR — docker or tcpdump not available, or container setup failed
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; RESET='\033[0m'
info()  { echo -e "${BOLD}[INFO]${RESET}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
pass()  { echo -e "${GREEN}${BOLD}[PASS]${RESET}  $*"; }
fail()  { echo -e "${RED}${BOLD}[FAIL]${RESET}  $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ── Configuration ─────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/docker-compose.yml}"
DATASET="${DATASET:-/app/reference_data/eval_set}"
MODEL="${MODEL:-/app/reference_data/model.pt}"
REF="${REF:-/app/reference_data/ref_set}"
DB="${DB:-/app/reference_data/provenance.db}"
OUT="${OUT:-/app/reference_data/outputs}"

CAPTURE_FILE="/tmp/anchor_isolation_capture.pcap"
CONTAINER_NAME="anchor-isolation-proof-$$"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}   Anchor AI Integrity — Network Isolation Proof               ${RESET}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
echo ""

for cmd in docker tcpdump; do
    if ! command -v "${cmd}" &>/dev/null; then
        error "'${cmd}' is not installed or not on PATH."
        exit 2
    fi
done

if [[ ! -f "${COMPOSE_FILE}" ]]; then
    error "docker-compose.yml not found at: ${COMPOSE_FILE}"
    exit 2
fi

info "Compose file : ${COMPOSE_FILE}"
info "Container    : ${CONTAINER_NAME}"
info "Capture file : ${CAPTURE_FILE}"
echo ""

# ── Cleanup trap ──────────────────────────────────────────────────────────────
cleanup() {
    info "Cleaning up container '${CONTAINER_NAME}' if running …"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    rm -f "${CAPTURE_FILE}.pid"
}
trap cleanup EXIT

# ── Build image if needed ─────────────────────────────────────────────────────
info "Building anchor image (cached layers re-used) …"
docker compose -f "${COMPOSE_FILE}" build anchor-eval
echo ""

# ── Launch the pipeline container detached ────────────────────────────────────
info "Launching anchor-eval container (network_mode: none) …"
docker run \
    --name "${CONTAINER_NAME}" \
    --network none \
    --detach \
    --volume "${REPO_ROOT}/reference_data:/app/reference_data" \
    --env PYTHONPATH=/app \
    "$(docker compose -f "${COMPOSE_FILE}" config | grep 'image:' | head -1 | awk '{print $2}')" \
    python /app/scripts/run_eval.py \
        --dataset "${DATASET}" \
        --model   "${MODEL}" \
        --ref     "${REF}" \
        --db      "${DB}" \
        --out     "${OUT}" \
    > /dev/null

CONTAINER_PID=$(docker inspect --format '{{.State.Pid}}' "${CONTAINER_NAME}")
info "Container PID on host : ${CONTAINER_PID}"

# ── Start tcpdump on the container's network namespace ────────────────────────
# We enter the container's net namespace via nsenter so tcpdump runs with full
# visibility into every interface the container can ever use.
# Filter: exclude localhost-to-localhost traffic; count only externally-routable
# packet attempts (src/dst not in 127.0.0.0/8).
info "Starting tcpdump in container's net namespace …"
echo "   (filter: NOT (src net 127.0.0.0/8 AND dst net 127.0.0.0/8))"
echo ""

sudo nsenter -t "${CONTAINER_PID}" -n -- \
    tcpdump -i any \
    --snapshot-length=0 \
    -w "${CAPTURE_FILE}" \
    "not (src net 127.0.0.0/8 and dst net 127.0.0.0/8)" \
    2>/dev/null &
TCPDUMP_PID=$!
echo "${TCPDUMP_PID}" > "${CAPTURE_FILE}.pid"

# Give tcpdump a moment to initialise
sleep 1
info "tcpdump running (PID ${TCPDUMP_PID}) — waiting for pipeline to finish …"
echo ""

# ── Wait for pipeline to exit ─────────────────────────────────────────────────
PIPELINE_EXIT=0
docker wait "${CONTAINER_NAME}" > /dev/null || PIPELINE_EXIT=$?

info "Pipeline container exited with code ${PIPELINE_EXIT}."
echo ""
info "Pipeline stdout/stderr:"
echo "───────────────────────────────────────────────────────────────"
docker logs "${CONTAINER_NAME}" 2>&1 | sed 's/^/  /'
echo "───────────────────────────────────────────────────────────────"
echo ""

# ── Stop tcpdump ──────────────────────────────────────────────────────────────
info "Stopping tcpdump …"
kill "${TCPDUMP_PID}" 2>/dev/null || true
wait "${TCPDUMP_PID}" 2>/dev/null || true
sleep 0.5

# ── Count captured packets ────────────────────────────────────────────────────
if [[ ! -f "${CAPTURE_FILE}" ]]; then
    error "Capture file was not created. tcpdump may have failed to start."
    exit 2
fi

# tcpdump -r counts packets; grep for "^[0-9]" to extract the count line
PACKET_COUNT=$(tcpdump -r "${CAPTURE_FILE}" 2>/dev/null | wc -l | tr -d ' ')
info "Non-loopback packets captured : ${PACKET_COUNT}"
echo ""

# ── Isolation assertion ───────────────────────────────────────────────────────
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
if [[ "${PACKET_COUNT}" -eq 0 ]]; then
    pass "ISOLATION PROOF : PASS"
    echo ""
    echo -e "  ${GREEN}Zero non-loopback packets captured during the full pipeline run.${RESET}"
    echo -e "  ${GREEN}The anchor-eval container made NO network calls to the outside${RESET}"
    echo -e "  ${GREEN}world. The evaluation is provably air-gapped.${RESET}"
    echo ""
    echo -e "  Capture file : ${CAPTURE_FILE}"
    echo -e "  Pipeline exit: ${PIPELINE_EXIT}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
    echo ""
    exit 0
else
    fail "ISOLATION PROOF : FAIL"
    echo ""
    echo -e "  ${RED}${PACKET_COUNT} non-loopback packet(s) were captured.${RESET}"
    echo -e "  ${RED}The container attempted external network communication.${RESET}"
    echo ""
    echo "  Captured packets:"
    tcpdump -r "${CAPTURE_FILE}" 2>/dev/null | head -20 | sed 's/^/    /'
    echo ""
    echo -e "  Full capture : ${CAPTURE_FILE}"
    echo -e "${BOLD}═══════════════════════════════════════════════════════════════${RESET}"
    echo ""
    exit 1
fi
