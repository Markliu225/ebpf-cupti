#!/usr/bin/env bash
# Baseline vs profiled overhead measurement (sequential, stable, minimal noise)
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/mark/anaconda3/envs/ebpf-cupti/bin/python}
PY_CONV=${PY_CONV:-/usr/bin/python3}
TIME_BIN=${TIME_BIN:-/usr/bin/time}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FLAMEGRAPH_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PM_SAMPLING_DIR="${FLAMEGRAPH_DIR}/pm_sampling"
PM_SAMPLING_BIN="${PM_SAMPLING_DIR}/pm_sampling"
PM_SAMPLING_ENABLED=${PM_SAMPLING_ENABLED:-1}
PM_SAMPLING_DURATION=${PM_SAMPLING_DURATION:-0}
PM_SAMPLING_DECODE_INTERVAL=${PM_SAMPLING_DECODE_INTERVAL:-500}
PM_SAMPLING_PID=""

TEST1_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/test1.py"
TEST2_CMD="cd ${FLAMEGRAPH_DIR} && ${PYTHON_BIN} gpu_profiler/test2.py"

OUT_BASE="${FLAMEGRAPH_DIR}/demores_overhead"
BASELINE_OUT="${OUT_BASE}/baseline"
PROFILE_OUT="${OUT_BASE}/profiled"
SUMMARY_TXT="${OUT_BASE}/summary.txt"

WARMUP=${WARMUP:-1}
REPEAT=${REPEAT:-1}

mkdir -p "${BASELINE_OUT}" "${PROFILE_OUT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

warmup_run() {
  local cmd="$1"
  local count=${2:-1}
  for _ in $(seq 1 "${count}"); do
    # shellcheck disable=SC2086
    ${TIME_BIN} -f "%e %M" -o /dev/null bash -c "$cmd" >/dev/null 2>&1 || true
  done
}

measure_cmd() {
  local label=$1; shift
  local cmd="$*"
  local tmp=$(mktemp)
  # shellcheck disable=SC2086
  ${TIME_BIN} -f "%e %M" -o "$tmp" bash -c "$cmd" >/dev/null 2>&1
  local secs rss_kb
  read -r secs rss_kb <"$tmp"
  rm -f "$tmp"
  local ms rss_mb
  ms=$(${PY_CONV} -c "import sys; print(int(float(sys.argv[1])*1000))" "$secs")
  rss_mb=$(${PY_CONV} -c "import sys; print(f'{int(sys.argv[1])/1024:.1f}')" "$rss_kb")
  echo "$label,$ms,$rss_mb"
}

start_pm_sampling() {
  if [[ "${PM_SAMPLING_ENABLED}" -ne 1 ]]; then
    log "pm_sampling collector disabled"
    return 0
  fi
  if [[ ! -x "${PM_SAMPLING_BIN}" ]]; then
    log "pm_sampling binary not found, building in ${PM_SAMPLING_DIR}"
    (cd "${PM_SAMPLING_DIR}" && make pm_sampling) || {
      log "Failed to build pm_sampling"
      return 1
    }
  fi
  log "Starting pm_sampling (durationSec=${PM_SAMPLING_DURATION}, decodeIntervalMs=${PM_SAMPLING_DECODE_INTERVAL})"
  pushd "${PM_SAMPLING_DIR}" >/dev/null
  sudo ./pm_sampling \
    --durationSec "${PM_SAMPLING_DURATION}" \
    --csv "${OUT_BASE}/pm_sampling_metrics.csv" \
    --decodeIntervalMs "${PM_SAMPLING_DECODE_INTERVAL}" \
    --livePrint 0 \
    --no-hold \
    >"${OUT_BASE}/pm_sampling.log" 2>&1 &
  PM_SAMPLING_PID=$!
  popd >/dev/null
  sleep 1
  if ! kill -0 "${PM_SAMPLING_PID}" 2>/dev/null; then
    log "pm_sampling failed to start; see ${OUT_BASE}/pm_sampling.log"
    PM_SAMPLING_PID=""
    return 1
  fi
  log "pm_sampling running with PID ${PM_SAMPLING_PID}"
}

stop_pm_sampling() {
  if [[ -z "${PM_SAMPLING_PID}" ]]; then return 0; fi
  if kill -0 "${PM_SAMPLING_PID}" 2>/dev/null; then
    log "Stopping pm_sampling"
    kill -INT "${PM_SAMPLING_PID}" 2>/dev/null || true
  fi
  wait "${PM_SAMPLING_PID}" || log "pm_sampling exited non-zero"
  PM_SAMPLING_PID=""
}

trap 'stop_pm_sampling' EXIT

run_pair() {
  local name=$1
  local base_cmd=$2
  local prof_out_dir="$3"

  warmup_run "$base_cmd" "$WARMUP"
  local base_row
  base_row=$(measure_cmd "baseline-${name}" "$base_cmd")

  mkdir -p "$prof_out_dir"
  start_pm_sampling || true
  warmup_run "${PYTHON_BIN} ${SCRIPT_DIR}/run_gpu_trace.py --command \"${base_cmd}\" --output-dir ${prof_out_dir}" "$WARMUP"
  local prof_row
  prof_row=$(measure_cmd "profiled-${name}" "${PYTHON_BIN} ${SCRIPT_DIR}/run_gpu_trace.py --command \"${base_cmd}\" --output-dir ${prof_out_dir}")
  stop_pm_sampling || true

  echo "$base_row"; echo "$prof_row"
}

rows=()
for _ in $(seq 1 "${REPEAT}"); do
  while IFS= read -r line; do rows+=("$line"); done < <(run_pair test1 "${TEST1_CMD}" "${PROFILE_OUT}/test1")
  while IFS= read -r line; do rows+=("$line"); done < <(run_pair test2 "${TEST2_CMD}" "${PROFILE_OUT}/test2")
done

if [[ ${#rows[@]} -eq 0 ]]; then
  log "No measurements captured"; exit 1
fi

extract_ms() { echo "$1" | cut -d',' -f2; }

base_ms_avg=$(printf "%s\n" "${rows[@]}" | awk -F',' '/^baseline/{print $2}' | \
  "${PY_CONV}" -c 'import sys; vals=[float(l) for l in sys.stdin if l.strip()]; print(int(sum(vals)/len(vals)) if vals else 0)')

prof_ms_avg=$(printf "%s\n" "${rows[@]}" | awk -F',' '/^profiled/{print $2}' | \
  "${PY_CONV}" -c 'import sys; vals=[float(l) for l in sys.stdin if l.strip()]; print(int(sum(vals)/len(vals)) if vals else 0)')

overhead_ms=$((prof_ms_avg - base_ms_avg))
overhead_pct=$(${PY_CONV} -c "import sys; b=float(sys.argv[1]); p=float(sys.argv[2]); print(0.0 if b==0 else (p-b)/b*100)" "$base_ms_avg" "$prof_ms_avg")

cat >"${SUMMARY_TXT}" <<EOF
模式, 平均时长(ms)
baseline, ${base_ms_avg}
profiled, ${prof_ms_avg}
开销(ms), ${overhead_ms}
开销占比, ${overhead_pct}%
EOF

log "Overhead summary written to ${SUMMARY_TXT}"
cat "${SUMMARY_TXT}"
