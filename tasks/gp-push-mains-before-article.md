---
id: "gp-push-mains-before-article"
title: "Push tessl and ojas main so every SHA the article cites resolves on GitHub"
status: ready
priority: 0
severity: high
type: chore
owner: "unassigned"
due: "before the LinkedIn article is posted"
labels:
  - "release"
  - "article"
  - "research-2026-10-20"
repositories:
  - "tessl"
  - "ojas"
planned_files: []
acceptance_criteria:
  - "tessl main pushed to origin; `git rev-list --count origin/main..main` is 0"
  - "ojas main pushed to origin in the same pass; `git rev-list --count origin/main..main` is 0"
  - "Every SHA the article cites (including 5971091, the EmbeddingGemma 2 merge) opens on github.com; the wave 19 coverage (303 kernels, 301 with bench lanes) and ABI 13 are visible on the public main"
---

# Task brief v1

## Title
Push tessl and ojas main so every SHA the article cites resolves on GitHub

Task: gp-push-mains-before-article
Type: chore
Status: ready
Priority: 0 (Urgent)
Severity: high
Source: research session 2026-10-20 (article audit gaps)

## Description
The article links the public tessl repo. On 2026-10-20 local tessl main was 15 commits ahead of `origin/main`, and the public main stopped at `a8fc965` (2026-10-19, "BUILD_REPORT: history scrub and wave 18"). The article cites three things that exist only locally:
- the EmbeddingGemma 2 merge (`5971091`)
- the wave 19 kernel coverage: 303 kernels, 301 with bench lanes (`8737fa1`)
- ABI 13

According to the audit, ojas is 2 commits ahead. That doesn't affect the article, but push it in the same pass.

Follow the main fast-forward handshake with ojas: announce the sha, and wait for "rerun done" before moving main.

## Acceptance criteria
- [ ] tessl main pushed to origin; `git rev-list --count origin/main..main` is 0
- [ ] ojas main pushed to origin in the same pass; `git rev-list --count origin/main..main` is 0
- [ ] Every SHA the article cites (including 5971091, the EmbeddingGemma 2 merge) opens on github.com; the wave 19 coverage (303 kernels, 301 with bench lanes) and ABI 13 are visible on the public main
