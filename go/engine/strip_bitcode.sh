#!/bin/bash
# strip_bitcode.sh IN.a OUT.a
#
# Copies a Rust staticlib to OUT.a with every Mach-O member's embedded LLVM bitcode
# (the __LLVM,__bitcode section) removed. Go 1.27.4's cmd/go refuses to link a cgo
# package whose -L directory holds an object with an LTO section ("disallowed LTO
# section"), and the archive cannot be built without one: the precompiled std,
# core and compiler_builtins members that rustc copies into every staticlib carry
# bitcode whatever the profile's lto or -C embed-bitcode says. The bitcode is
# only for a later rustc LTO pass; clang and ld64 linking this archive into a Go
# binary never read it, so removing it changes no code or symbol.
#
# Each member is relinked alone (`ld -r`, which drops the bitcode segment), so the
# archive keeps its members and their lazy selection by the final link. The result
# is checked here: the member count must match and no member may keep a bitcode
# section. Any failure leaves no OUT.a. (Symbols are not diffed here: Apple's nm
# cannot read the input's bitcode-bearing members.)
set -euo pipefail
export LC_ALL=C

in=${1:?usage: strip_bitcode.sh IN.a OUT.a}
out=${2:?usage: strip_bitcode.sh IN.a OUT.a}
[ -f "$in" ] || { echo "strip_bitcode: no such archive: $in" >&2; exit 1; }

outdir=$(dirname "$out")
mkdir -p "$outdir"
work=$(mktemp -d "${out}.work.XXXXXX")
trap 'rm -rf "$work"' EXIT
rm -f "$out"

mkdir "$work/in" "$work/clean"
(cd "$work/in" && ar x "$(cd "$(dirname "$in")" && pwd)/$(basename "$in")")

count=0
for f in "$work"/in/*.o; do
  ld -r -arch arm64 "$f" -o "$work/clean/$(basename "$f")"
  count=$((count + 1))
done
members=$(ar -t "$in" | grep -c '\.o$' || true)
if [ "$count" -eq 0 ] || [ "$count" -ne "$members" ]; then
  echo "strip_bitcode: relinked $count objects but $in has $members" >&2
  exit 1
fi

for f in "$work"/clean/*.o; do
  if otool -l "$f" | grep -q '__bitcode'; then
    echo "strip_bitcode: $(basename "$f") still has a bitcode section" >&2
    exit 1
  fi
done

libtool -static -no_warning_for_no_symbols -o "$work/out.a" "$work"/clean/*.o
mv "$work/out.a" "$out"
echo "strip_bitcode: $out ($count members)"
