#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FLAMEGRAPH_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

TEST1_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/test1.py"
TEST2_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/test2.py"

OUT_BASE="${FLAMEGRAPH_DIR}/demores_dual"
TEST1_OUT="${OUT_BASE}/test1"
TEST2_OUT="${OUT_BASE}/test2"

mkdir -p "${TEST1_OUT}" "${TEST2_OUT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting profiling for test1.py"
cd "${FLAMEGRAPH_DIR}"

${PYTHON_BIN} gpu_profiler/run_gpu_trace.py \
  --command "${TEST1_CMD}" \
  --output-dir "${TEST1_OUT}" \
  >"${TEST1_OUT}/trace.log" 2>&1 &
PID1=$!

log "Starting profiling for test2.py"
${PYTHON_BIN} gpu_profiler/run_gpu_trace.py \
  --command "${TEST2_CMD}" \
  --output-dir "${TEST2_OUT}" \
  >"${TEST2_OUT}/trace.log" 2>&1 &
PID2=$!

FAIL=0
wait ${PID1} || FAIL=1
wait ${PID2} || FAIL=1

if [[ ${FAIL} -ne 0 ]]; then
  log "One or more profiling runs failed. Check logs under ${OUT_BASE}."
  exit 1
fi

log "Profiling complete. Results stored in ${OUT_BASE}/test1 and ${OUT_BASE}/test2."

TIMELINE_FILES=()
TEST1_TIMELINE="${TEST1_OUT}/cpu_gpu_timeline.txt"
TEST2_TIMELINE="${TEST2_OUT}/cpu_gpu_timeline.txt"

if [[ -f "${TEST1_TIMELINE}" ]]; then
  TIMELINE_FILES+=("${TEST1_TIMELINE}")
else
  log "Warning: ${TEST1_TIMELINE} not found; skipping."
fi

if [[ -f "${TEST2_TIMELINE}" ]]; then
  TIMELINE_FILES+=("${TEST2_TIMELINE}")
else
  log "Warning: ${TEST2_TIMELINE} not found; skipping."
fi

if [[ ${#TIMELINE_FILES[@]} -gt 0 ]]; then
  COMBINED_TIMELINE="${OUT_BASE}/combined_timeline.txt"
  log "Generating combined timeline at ${COMBINED_TIMELINE}"
  ${PYTHON_BIN} "${FLAMEGRAPH_DIR}/combine_timelines.py" "${TIMELINE_FILES[@]}" \
    --output "${COMBINED_TIMELINE}" || {
      log "Failed to generate combined timeline"
      exit 1
    }
else
  log "No per-run timelines found; skipping combined timeline generation."
fi
