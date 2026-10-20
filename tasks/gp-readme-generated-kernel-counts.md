---
id: "gp-readme-generated-kernel-counts"
title: "Generate the README kernel counts from kernel_coverage and fail the docs check on drift"
status: ready
priority: 3
severity: low
type: chore
owner: "unassigned"
due: "none"
labels:
  - "docs"
  - "coverage"
  - "research-2026-10-20"
repositories:
  - "tessl"
planned_files:
  - "README.md"
  - "scripts/kernel_coverage.py"
acceptance_criteria:
  - "The README's kernel source and entry-point counts (diagram and text) come from kernel_coverage output, not hand edits"
  - "The docs check fails when the README counts differ from kernel_coverage"
  - "The current mismatch (diagram: 35 sources / 256 entry points; text: 39 / 303) is resolved"
---

# Task brief v1

## Title
Generate the README kernel counts from kernel_coverage and fail the docs check on drift

Task: gp-readme-generated-kernel-counts
Type: chore
Status: ready
Priority: 3 (Low)
Severity: low
Source: research session 2026-10-20 (article audit gaps)

## Description
The README contradicts itself. Its diagram says "35 kernel sources (256 entry points)", while its text says 39 sources and 303 entry points. Both numbers were hand-written. Have `kernel_coverage` produce them so they can't drift. The docs and dialect checks parse source text, so run them after the change.

## Acceptance criteria
- [ ] The README's kernel source and entry-point counts (diagram and text) come from kernel_coverage output, not hand edits
- [ ] The docs check fails when the README counts differ from kernel_coverage
- [ ] The current mismatch (diagram: 35 sources / 256 entry points; text: 39 / 303) is resolved
