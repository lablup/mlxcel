# Technical Reports

Per-PR bilingual (English and Korean) technical reports for merged pull requests, grouped by the minor release line that ships them.

## Layout

```
TECHNICAL_REPORTS/
  README.md            <- this file: layout and filing rules
  .keep-reports        <- opts the directory into git tracking
  <MAJOR>.<MINOR>/     <- e.g. 0.8/
    README.md          <- generated index, released lines only
    <PR>-<slug>-<YYYYMMDD>.en.md
    <PR>-<slug>-<YYYYMMDD>.ko.md
```

- Folders are split at the minor version only. Every patch release and pre-release of a line shares its folder: `v0.8.0-beta.1`, `v0.8.0` and `v0.8.1` all map to `0.8/`.
- Folder names are exact versions without the `v` prefix (`0.9`, `0.10`), so a plain alphabetical listing puts `0.10` before `0.2`.
- Filename: `<PR-number>-<kebab-title>-<merge-date>.<lang>.md`. A few older reports are keyed by the closed issue number, or use a `pr-<number>-` prefix without a date.
- No report lives at the directory root. `scripts/ci/check_technical_reports_layout.py` fails CI when one does.

## Index

Each released line has a generated index by PR number (title, date, English and Korean links) at `<MAJOR>.<MINOR>/README.md`, for example [`0.8/README.md`](0.8/README.md). The line under development has none: every merged PR commits its own report (`.keep-reports`), so a committed index of that folder would be one file that every open PR rewrites at the same spot, and concurrent PRs would conflict on it in turn. lablup/backend.ai-go stopped tracking its report table for the same reason. Run `make reports-index-current` to print it instead.

A released line no longer receives reports, so its index is written once and then stays fixed. `scripts/ci/gen_reports_index.py --check` (run by the `python-tooling` CI job) fails when a released line's index is stale or missing, and when the current line has a committed one. Two moments need `make reports-index`:

- The PR that bumps the root `Cargo.toml` to a new minor line, which turns the previous line into a released one. `make bump-version` already runs it.
- A report backfilled into a released line.

## Which folder

A report belongs to the minor line of the first release that contains its PR.

For a new report, that is the `MAJOR.MINOR` of the workspace `version` in the root `Cargo.toml` on the PR's base branch:

```bash
grep -m1 '^version' Cargo.toml   # version = "0.9.0-beta.3"  ->  TECHNICAL_REPORTS/0.9/
```

This holds across the release cycle because the version only moves forward. While `0.9.0-beta.N` is the version, the PR ships in a 0.9 pre-release or in `0.9.0`. After `v0.9.0` is tagged and before the next bump, the PR ships in `0.9.1` at the latest, which is still `0.9/`. Once release preparation bumps the version to `0.10.0-beta.1`, new reports go to `0.10/`. Create the folder with the first report filed there.

The existing reports were placed by the first release tag containing each PR's merge commit (`git tag --contains`). Reports keyed by an issue number, which has no merge commit, were placed by the date in the filename against the tag dates.

## Filing new reports

The shared `report` and `chain-report` workflows default to the directory root. In this repository they must write both language files into the folder chosen above, and look for an existing report with `ls TECHNICAL_REPORTS/*/<PR>-*` rather than at the root. Commit only the two report files; nothing here is regenerated per report.

When one report links to another, link across folders with `../<MAJOR>.<MINOR>/<file>`, and to the rest of the repository with `../../<path>`.
