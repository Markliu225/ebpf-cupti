#!/usr/bin/env bash
# sudo -E PYTHON_BIN=/home/mark/anaconda3/envs/ebpf-cupti/bin/python ./gpu_profiler/run.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FLAMEGRAPH_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PM_SAMPLING_DIR="${FLAMEGRAPH_DIR}/pm_sampling"
PM_SAMPLING_BIN="${PM_SAMPLING_DIR}/pm_sampling"
PM_SAMPLING_ENABLED=${PM_SAMPLING_ENABLED:-1}
PM_SAMPLING_DURATION=${PM_SAMPLING_DURATION:-0}
PM_SAMPLING_DECODE_INTERVAL=${PM_SAMPLING_DECODE_INTERVAL:-500}
PM_SAMPLING_PID=""

# TEST1_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/test1.py"
TEST1_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/workflow/run.py synthetic_gemm_tensor --mode train --epochs 10"
# TEST2_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/test2.py"
TEST2_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/workflow/run.py synthetic_gemm_fp32 --mode train --epochs 10"


OUT_BASE="${FLAMEGRAPH_DIR}/demores_dual"
TEST1_OUT="${OUT_BASE}/test1"
TEST2_OUT="${OUT_BASE}/test2"
PM_SAMPLING_LOG="${OUT_BASE}/pm_sampling.log"
PM_SAMPLING_CSV="${OUT_BASE}/pm_sampling_metrics.csv"

mkdir -p "${TEST1_OUT}" "${TEST2_OUT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

start_pm_sampling() {
  if [[ "${PM_SAMPLING_ENABLED}" -ne 1 ]]; then
    log "pm_sampling collector disabled via PM_SAMPLING_ENABLED"
    return 0
  fi

  if [[ ! -x "${PM_SAMPLING_BIN}" ]]; then
    log "pm_sampling binary not found, building in ${PM_SAMPLING_DIR}"
    (cd "${PM_SAMPLING_DIR}" && make pm_sampling) || {
      log "Failed to build pm_sampling"
      return 1
    }
  fi

  log "Starting pm_sampling collector (durationSec=${PM_SAMPLING_DURATION}, decodeIntervalMs=${PM_SAMPLING_DECODE_INTERVAL})"
  pushd "${PM_SAMPLING_DIR}" >/dev/null
  sudo ./pm_sampling \
    --durationSec "${PM_SAMPLING_DURATION}" \
    --csv "${PM_SAMPLING_CSV}" \
    --decodeIntervalMs "${PM_SAMPLING_DECODE_INTERVAL}" \
    --livePrint 0 \
    --no-hold \
    >"${PM_SAMPLING_LOG}" 2>&1 &
  PM_SAMPLING_PID=$!
  popd >/dev/null
  sleep 1
  if ! kill -0 "${PM_SAMPLING_PID}" 2>/dev/null; then
    log "pm_sampling collector failed to start; see ${PM_SAMPLING_LOG}"
    PM_SAMPLING_PID=""
    return 1
  fi
  log "pm_sampling running with PID ${PM_SAMPLING_PID}; logging to ${PM_SAMPLING_LOG}"
  return 0
}

stop_pm_sampling() {
  if [[ -z "${PM_SAMPLING_PID}" ]]; then
    return 0
  fi
  if kill -0 "${PM_SAMPLING_PID}" 2>/dev/null; then
    log "Stopping pm_sampling collector"
    kill -INT "${PM_SAMPLING_PID}" 2>/dev/null || true
  fi
  if ! wait "${PM_SAMPLING_PID}"; then
    log "pm_sampling exited with a non-zero status"
  else
    log "pm_sampling metrics saved to ${PM_SAMPLING_CSV}"
  fi
  PM_SAMPLING_PID=""
}

cleanup() {
  local exit_code=$?
  trap - EXIT
  stop_pm_sampling || true
  exit ${exit_code}
}

trap cleanup EXIT

log "Starting profiling for test1.py"
cd "${FLAMEGRAPH_DIR}"

start_pm_sampling || {
  log "Unable to launch pm_sampling collector"
  exit 1
}

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

stop_pm_sampling
