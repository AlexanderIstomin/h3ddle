#!/bin/sh
# Reports untracked Swift sources that tracked code already depends on.
#
# Local builds see every file on disk, so a tree can compile perfectly while
# the commit is incomplete: staging a shared file such as AppModel.swift
# captures whatever a colleague has written into it, including references to
# types whose defining files are not committed yet. The push then breaks the
# build for everyone else. This has happened twice; the check is cheap.
#
# It compares type declarations in untracked files against references in
# tracked ones. Advisory by default so it never blocks work in progress; pass
# --strict (as CI does) to fail instead.
set -eu

repository_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repository_root"

strict=0
[ "${1:-}" = "--strict" ] && strict=1

untracked=$(git ls-files --others --exclude-standard -- '*.swift' || true)
[ -n "$untracked" ] || exit 0

findings=""
for file in $untracked; do
    # Types this untracked file introduces.
    # Top-level declarations only: nested ones are indented, and names like
    # CodingKeys appear in every file that encodes anything.
    names=$(sed -nE 's/^(public |internal |private |fileprivate |final |@[A-Za-z]+ )*(struct|class|enum|actor|protocol|typealias) ([A-Za-z_][A-Za-z0-9_]*).*/\3/p' "$file" | sort -u)
    for name in $names; do
        # Referenced anywhere that is tracked and not this file's own siblings?
        users=$(git grep -l -w "$name" -- '*.swift' 2>/dev/null || true)
        [ -n "$users" ] || continue
        findings="${findings}  ${name} (defined in ${file})\n"
        for user in $users; do
            findings="${findings}      referenced by ${user}\n"
        done
    done
done

[ -n "$findings" ] || exit 0

printf 'Untracked sources that committed code already references:\n' >&2
printf "$findings" >&2
printf 'Commit these files together, or the push breaks the build for others.\n' >&2
[ "$strict" -eq 1 ] && exit 1
exit 0
