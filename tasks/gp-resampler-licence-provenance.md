---
id: "gp-resampler-licence-provenance"
title: "Document the licence provenance of src/media/resample.rs after the LGPL removal"
status: ready
priority: 1
severity: high
type: chore
owner: "unassigned"
due: "none"
labels:
  - "licence"
  - "media"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files:
  - "src/media/resample.rs"
  - "NOTICE"
  - "README.md"
acceptance_criteria:
  - "Provenance of the current src/media/resample.rs written down: clean-room (from which spec or paper, by whom, without reading the libswresample source) or derived under a licence compatible with tessl's"
  - "If it can't be shown to be clean-room, it is replaced with an implementation that can, or relicensed accordingly"
  - "NOTICE and README updated to match"
---

# Task brief v1

## Title
Document the licence provenance of src/media/resample.rs after the LGPL removal

Task: gp-resampler-licence-provenance
Type: chore
Status: ready
Priority: 1 (High)
Severity: high
Source: research session 2026-10-20 (article audit gaps)

## Description
The 2026-10-18 history rewrite removed an LGPL-derived `src/media/resample.rs`. The file now in the tree is documented as matching libswresample's output. Matching output is fine, but nothing records how the new file was written. Since the old file was LGPL-derived, the repo needs to show the new one isn't.

## Acceptance criteria
- [ ] Provenance of the current src/media/resample.rs written down: clean-room (from which spec or paper, by whom, without reading the libswresample source) or derived under a licence compatible with tessl's
- [ ] If it can't be shown to be clean-room, it is replaced with an implementation that can, or relicensed accordingly
- [ ] NOTICE and README updated to match
