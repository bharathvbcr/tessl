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
# A leading-dash path would be read as an option by dirname, basename, ar and ld.
case "$in" in -*) in=./$in ;; esac
case "$out" in -*) out=./$out ;; esac
[ -f "$in" ] || { echo "strip_bitcode: no such archive: $in" >&2; exit 1; }

# Both paths become absolute here, before any cd below: a relative IN resolved
# after `cd "$work/in"` pointed into the work directory, and the extract failed.
absdir() { (cd "$(dirname "$1")" && pwd -P) || { echo "strip_bitcode: cannot enter the directory of $1" >&2; return 1; }; }
mkdir -p "$(dirname "$out")"
in="$(absdir "$in")/$(basename "$in")"
out="$(absdir "$out")/$(basename "$out")"
[ -f "$in" ] || { echo "strip_bitcode: no such archive: $in" >&2; exit 1; }
[ "$in" != "$out" ] || { echo "strip_bitcode: IN and OUT are the same file: $in" >&2; exit 2; }

# Succeeds when otool -l names a bitcode section in $1. An otool that fails stops the
# script: its output is captured first, so a failure (or a pipe closed early by
# grep -q, which would SIGPIPE otool) can never read as "no bitcode".
has_bitcode() {
  local listing
  listing=$(otool -l "$1") || { echo "strip_bitcode: otool -l failed on $1" >&2; exit 1; }
  case $listing in *__bitcode*) return 0 ;; *) return 1 ;; esac
}

work=$(mktemp -d "${out}.work.XXXXXX")
trap 'rm -rf "$work"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
rm -f "$out"

# Every member must be a relinkable object; the only other entry allowed is the
# symbol table. Anything else would be dropped from OUT without a word.
ar -t "$in" > "$work/members"
mkdir "$work/in" "$work/clean"
objects=0
while IFS= read -r name; do
  case $name in
    __.SYMDEF*) ;;
    *.o) objects=$((objects + 1)) ;;
    *) echo "strip_bitcode: $in has a member that is not an object: $name" >&2; exit 1 ;;
  esac
done < "$work/members"
# ar x writes one file per name, so a repeated name would list one object twice and lose the other.
sort "$work/members" | uniq -d > "$work/dups"
if [ -s "$work/dups" ]; then
  echo "strip_bitcode: $in lists a member name more than once: $(tr '\n' ' ' < "$work/dups")" >&2
  exit 1
fi
[ "$objects" -gt 0 ] || { echo "strip_bitcode: $in has no object members" >&2; exit 1; }
(cd "$work/in" && ar x "$in")

# The member list goes to libtool as a file: 6000 long names on its command line
# exceed ARG_MAX (the engine archive has 425 members).
count=0
: > "$work/filelist"
while IFS= read -r name; do
  case $name in __.SYMDEF*) continue ;; esac
  [ -f "$work/in/$name" ] || { echo "strip_bitcode: extracting $in did not produce $name (duplicate member names?)" >&2; exit 1; }
  ld -r -arch arm64 "$work/in/$name" -o "$work/clean/$name"
  if has_bitcode "$work/clean/$name"; then
    echo "strip_bitcode: $name still has a bitcode section" >&2
    exit 1
  fi
  echo "$work/clean/$name" >> "$work/filelist"
  count=$((count + 1))
done < "$work/members"
[ "$count" -eq "$objects" ] || { echo "strip_bitcode: relinked $count objects but $in has $objects" >&2; exit 1; }

libtool -static -no_warning_for_no_symbols -filelist "$work/filelist" -o "$work/out.a"
[ -s "$work/out.a" ] || { echo "strip_bitcode: libtool wrote no archive" >&2; exit 1; }
# The archive as written, not only its members: no bitcode may survive the repack.
if has_bitcode "$work/out.a"; then
  echo "strip_bitcode: the repacked archive still has a bitcode section" >&2
  exit 1
fi
mv "$work/out.a" "$out"
echo "strip_bitcode: $out ($count members)"
