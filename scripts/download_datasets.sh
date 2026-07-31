#!/usr/bin/env bash
# Clone the official SQuALITY and QMSum repositories into data/raw and record
# their commit hashes. Idempotent: skips a clone if the target already exists.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$HERE/data/raw"
MAN="$HERE/data/manifests"
mkdir -p "$RAW" "$MAN"

clone_if_missing() {
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then
    echo "[download] $dest already present, skipping clone"
  else
    echo "[download] cloning $url -> $dest"
    git clone --depth 1 "$url" "$dest"
  fi
}

clone_if_missing "https://github.com/nyu-mll/SQuALITY.git" "$RAW/squality"
clone_if_missing "https://github.com/Yale-LILY/QMSum.git" "$RAW/qmsum"

SQUALITY_HASH="$(git -C "$RAW/squality" rev-parse HEAD)"
QMSUM_HASH="$(git -C "$RAW/qmsum" rev-parse HEAD)"

echo "[download] squality HEAD: $SQUALITY_HASH"
echo "[download] qmsum HEAD:    $QMSUM_HASH"

cat > "$MAN/dataset_commits.json" <<EOF
{
  "squality": "$SQUALITY_HASH",
  "qmsum": "$QMSUM_HASH"
}
EOF

echo "[download] wrote $MAN/dataset_commits.json"

# quick structural sanity print (agent inspects to build QMSum loader)
echo "[download] --- SQuALITY data/ layout ---"
find "$RAW/squality/data" -maxdepth 2 -type f | head -30 || true
echo "[download] --- QMSum data/ALL layout ---"
find "$RAW/qmsum/data" -maxdepth 3 | head -60 || true
echo "DOWNLOAD_DONE_OK"
