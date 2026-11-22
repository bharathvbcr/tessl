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
[ -n "$in" ] && [ -n "$out" ] || { echo "strip_bitcode: IN.a and OUT.a must be non-empty" >&2; exit 2; }
[ -f "$in" ] || { echo "strip_bitcode: no such archive: $in" >&2; exit 1; }

# Both paths become absolute here, before any cd below: a relative IN resolved
# after `cd "$work/in"` pointed into the work directory, and the extract failed.
absdir() { (cd "$(dirname "$1")" && pwd -P) || { echo "strip_bitcode: cannot enter the directory of $1" >&2; return 1; }; }
mkdir -p "$(dirname "$out")"
in="$(absdir "$in")/$(basename "$in")"
out="$(absdir "$out")/$(basename "$out")"
[ -f "$in" ] || { echo "strip_bitcode: no such archive: $in" >&2; exit 1; }
[ "$in" != "$out" ] || { echo "strip_bitcode: IN and OUT are the same file: $in" >&2; exit 2; }

work=$(mktemp -d "${out}.work.XXXXXX")
trap 'rm -rf "$work"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
rm -f "$out"

mkdir "$work/in" "$work/clean"
(cd "$work/in" && ar x "$in")

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
[ -s "$work/out.a" ] || { echo "strip_bitcode: libtool wrote no archive" >&2; exit 1; }
# The archive as written, not only its members: no bitcode may survive the repack.
if otool -l "$work/out.a" 2>/dev/null | grep -q '__bitcode'; then
  echo "strip_bitcode: the repacked archive still has a bitcode section" >&2
  exit 1
fi
mv "$work/out.a" "$out"
echo "strip_bitcode: $out ($count members)"
