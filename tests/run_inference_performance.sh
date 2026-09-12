#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PERF_ENV_FILE="${URB_PERF_ENV_FILE:-${REPO_ROOT}/.env.performance}"

if [[ ! -f "${PERF_ENV_FILE}" ]]; then
    echo "error: performance environment file not found: ${PERF_ENV_FILE}" >&2
    echo "create it outside tests and set the three model directory variables" >&2
    exit 2
fi

# Preserve explicit caller overrides while loading machine-local defaults.
CALLER_THREADS="${URB_PERF_THREADS-}"
CALLER_NEW_TOKENS="${URB_PERF_NEW_TOKENS-}"
CALLER_PROMPT="${URB_PERF_PROMPT-}"
CALLER_SCENARIO="${URB_PERF_SCENARIO-}"
CALLER_OUTPUT_DIR="${URB_PERF_OUTPUT_DIR-}"
CALLER_GLM_SLOTS="${URB_PERF_GLM_EXPERT_SLOTS-}"
CALLER_DEEPSEEK_SLOTS="${URB_PERF_DEEPSEEK_EXPERT_SLOTS-}"
CALLER_KIMI_SLOTS="${URB_PERF_KIMI_EXPERT_SLOTS-}"

set -a
# shellcheck disable=SC1090
source "${PERF_ENV_FILE}"
set +a

[[ -n "${CALLER_THREADS}" ]] && export URB_PERF_THREADS="${CALLER_THREADS}"
[[ -n "${CALLER_NEW_TOKENS}" ]] && export URB_PERF_NEW_TOKENS="${CALLER_NEW_TOKENS}"
[[ -n "${CALLER_PROMPT}" ]] && export URB_PERF_PROMPT="${CALLER_PROMPT}"
[[ -n "${CALLER_SCENARIO}" ]] && export URB_PERF_SCENARIO="${CALLER_SCENARIO}"
[[ -n "${CALLER_GLM_SLOTS}" ]] && export URB_PERF_GLM_EXPERT_SLOTS="${CALLER_GLM_SLOTS}"
[[ -n "${CALLER_DEEPSEEK_SLOTS}" ]] && export URB_PERF_DEEPSEEK_EXPERT_SLOTS="${CALLER_DEEPSEEK_SLOTS}"
[[ -n "${CALLER_KIMI_SLOTS}" ]] && export URB_PERF_KIMI_EXPERT_SLOTS="${CALLER_KIMI_SLOTS}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export URB_PERF_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-${REPO_ROOT}/target/perf/${RUN_ID}}"

require_directory() {
    local variable_name="$1"
    local directory="${!variable_name-}"
    if [[ -z "${directory}" ]]; then
        echo "error: ${variable_name} is not set in ${PERF_ENV_FILE}" >&2
        exit 2
    fi
    if [[ ! -d "${directory}" ]]; then
        echo "error: ${variable_name} does not point to a directory" >&2
        exit 2
    fi
    if [[ ! -f "${directory}/config.json" ]]; then
        echo "error: ${variable_name} directory has no config.json" >&2
        exit 2
    fi
}

require_positive_integer() {
    local variable_name="$1"
    local value="${!variable_name-}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "error: ${variable_name} must be a positive integer" >&2
        exit 2
    fi
}

require_nonnegative_integer() {
    local variable_name="$1"
    local value="${!variable_name-}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "error: ${variable_name} must be a non-negative integer" >&2
        exit 2
    fi
}

require_directory URB_GLM52_DIR
require_directory URB_DEEPSEEK_V4_DIR
require_directory KIMI_K3_MODEL_DIR
require_positive_integer URB_PERF_THREADS
require_positive_integer URB_PERF_NEW_TOKENS
require_nonnegative_integer URB_PERF_GLM_EXPERT_SLOTS
require_nonnegative_integer URB_PERF_DEEPSEEK_EXPERT_SLOTS
require_nonnegative_integer URB_PERF_KIMI_EXPERT_SLOTS

mkdir -p "${URB_PERF_OUTPUT_DIR}"
cd "${REPO_ROOT}"

echo "Building release performance tests..."
cargo test --release --locked --test inference_performance --no-run

run_model() {
    local label="$1"
    local test_name="$2"
    local expert_slots="$3"

    echo
    echo "Running ${label} (${URB_PERF_THREADS} threads, ${expert_slots} expert slots/layer)..."
    URB_PERF_EXPERT_SLOTS="${expert_slots}" \
        cargo test --release --locked --test inference_performance "${test_name}" -- \
        --ignored --exact --nocapture --test-threads=1
}

run_model "GLM-5.2" "glm_52_inference_performance" "${URB_PERF_GLM_EXPERT_SLOTS}"
run_model "DeepSeek-V4" "deepseek_v4_inference_performance" "${URB_PERF_DEEPSEEK_EXPERT_SLOTS}"
run_model "Kimi-K3" "kimi_k3_inference_performance" "${URB_PERF_KIMI_EXPERT_SLOTS}"

echo
echo "All performance runs completed. Reports: ${URB_PERF_OUTPUT_DIR}"
