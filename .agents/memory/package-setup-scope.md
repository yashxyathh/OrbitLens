---
name: Package setup scope
description: A setup note for imported projects with dependencies in existing subdirectories.
---

When installing runtime dependencies for an imported project, preserve the
repository's existing package boundaries. Package tooling may create a root
manifest or lockfile even when the app is organized under frontend/ and
backend/; those generated files are not automatically part of the project.

**Why:** The imported SatQuery project already had separate frontend and backend
requirements, and root scaffolding from environment setup would have added
unrelated files to the requested frontend-only change.

**How to apply:** Review git status after package setup and remove generated
root manifests that are not part of the repository's intended structure.