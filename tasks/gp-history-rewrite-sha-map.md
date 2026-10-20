---
id: "gp-history-rewrite-sha-map"
title: "Publish the old-to-new SHA map from the 2026-10-18 history rewrite and prune refs/original"
status: ready
priority: 2
severity: medium
type: chore
owner: "unassigned"
due: "none"
labels:
  - "history"
  - "provenance"
  - "research-2026-10-20"
repositories:
  - "tessl"
  - "ojas"
planned_files:
  - "BUILD_REPORT.md"
acceptance_criteria:
  - "The old->new SHA map from the 2026-10-18 rewrite committed to tessl"
  - "Every pre-rewrite tessl SHA that ojas cites (e.g. 42f1d52) checked against the map and updated or annotated"
  - "refs/original pruned once the map is committed, and BUILD_REPORT updated to say so"
---

# Task brief v1

## Title
Publish the old-to-new SHA map from the 2026-10-18 history rewrite and prune refs/original

Task: gp-history-rewrite-sha-map
Type: chore
Status: ready
Priority: 2 (Normal)
Severity: medium
Source: research session 2026-10-20 (article audit gaps)

## Description
BUILD_REPORT says to keep `refs/original` until the old→new SHA map is published. The map hasn't been published yet, and ojas still cites pre-rewrite `42f1d52` as provenance. That SHA still resolves locally only because `refs/original` is still there.

## Acceptance criteria
- [ ] The old->new SHA map from the 2026-10-18 rewrite committed to tessl
- [ ] Every pre-rewrite tessl SHA that ojas cites (e.g. 42f1d52) checked against the map and updated or annotated
- [ ] refs/original pruned once the map is committed, and BUILD_REPORT updated to say so
