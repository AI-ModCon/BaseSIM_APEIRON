#!/usr/bin/env bash
# Fetch a staged SOLPS stream (and optionally the MATEY checkpoint) for this
# example. See examples/matey/STANDALONE.md for what each tier contains, and for
# which of them are cleared for distribution at all.
#
#   examples/matey/download_data.sh <dest> [--tier smoke|demo|figure] [--checkpoint]
#
# Override MATEY_ASSET_BASE to pull from a mirror or an on-site copy.
set -euo pipefail

DEST="${1:?usage: download_data.sh <dest> [--tier smoke|demo|figure] [--checkpoint]}"
shift
TIER="smoke"
WANT_CKPT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="${2:?--tier needs a value}"; shift 2 ;;
    --checkpoint) WANT_CKPT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

BASE="${MATEY_ASSET_BASE:-https://github.com/AI-ModCon/BaseSIM_APEIRON/releases/download/matey-data-v1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMS="${HERE}/checksums.txt"
mkdir -p "${DEST}"

# Verified against a committed checksum, and hard failure on mismatch. Silently
# wrong data is this example's established failure mode: a bad field-label map
# inflated NRMSE roughly six-fold without anything reporting an error, so an
# unverified download is not worth having.
fetch() {
  local name="$1" url="${BASE}/$1" out="${DEST}/$1"
  local want
  want="$(awk -v n="${name}" '$2 == n {print $1}' "${SUMS}" 2>/dev/null || true)"
  if [ -z "${want}" ]; then
    echo "no checksum for ${name} in ${SUMS}." >&2
    echo "That tier is not cleared for distribution -- see STANDALONE.md." >&2
    exit 3
  fi
  if [ -f "${out}" ] && echo "${want}  ${out}" | sha256sum -c --status; then
    echo "${name}: already present and verified"
    return
  fi
  echo "fetching ${name}"
  # -C - resumes a partial file: these are hundreds of megabytes and a dropped
  # connection should not mean starting again.
  curl -fL -C - -o "${out}" "${url}" || {
    echo "download failed: ${url}" >&2
    echo "If the tier exists but is gated, request access -- see STANDALONE.md." >&2
    exit 4
  }
  echo "${want}  ${out}" | sha256sum -c --status || {
    echo "CHECKSUM MISMATCH for ${name}; refusing to use it." >&2
    rm -f "${out}"
    exit 5
  }
  tar -xf "${out}" -C "${DEST}"
}

fetch "solps_stream_${TIER}.tar"
[ "${WANT_CKPT}" -eq 1 ] && {
  echo "The MATEY checkpoint is redistributed under MATEY's own terms; cite it"
  echo "as the model of record. See STANDALONE.md."
  fetch "matey_leadtime_1.tar"
}

STREAM="${DEST}/solps_stream"
echo
echo "done. Run with:"
echo "  --set data.path=${STREAM}"
[ "${WANT_CKPT}" -eq 1 ] && echo "  --set model.pretrained_path=${DEST}/leadtime_1/best_ckpt.tar"
exit 0
