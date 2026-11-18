#!/usr/bin/env bash
# Unpacks google-research/timesfm at the commit make_reference.py pins into
# target/timesfm3_upstream/<sha>/. The code is Apache-2.0; it is read by path and
# never installed or committed.
set -euo pipefail
SHA=918df02c9d366b2145bc5876135fa2131d3ed2a6
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$ROOT/target/timesfm3_upstream/$SHA"
if [[ -d "$DEST/src/timesfm3/torch" ]]; then echo "$DEST"; exit 0; fi
mkdir -p "$DEST"
TMP="$DEST.tar.gz"
curl -fsSL --max-time 300 -o "$TMP" "https://codeload.github.com/google-research/timesfm/tar.gz/$SHA"
tar -xzf "$TMP" -C "$DEST" --strip-components=1
rm -f "$TMP"
echo "$DEST"
