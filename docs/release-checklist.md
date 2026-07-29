# Public Release Checklist

## 日本語概要

本書は、公開リリース前に、実験根拠、入力の由来、機密情報の不在、公開インターフェース、再現性、配布物、文書を確認するためのチェックリストです。未検証の機能追加ではなく、既存の証拠と互換性の確認に焦点を当てています。

確認項目と実行コマンドは以下の英語本文を参照してください。

---

## English Summary

This checklist defines the review used for stable public releases. It keeps
release work focused on evidence quality, compatibility, and reproducibility
rather than adding unreviewed features at the end of a release cycle.

## Scope and evidence

- [ ] State whether experiment methods, parameters, metrics, or conclusions
  changed.
- [ ] Confirm that every input is generated in the repository or derived from
  a documented public source.
- [ ] Confirm that no confidential, identifying, environment-specific,
  authentication, personal-contact, or other non-public material is present.
- [ ] Keep controlled findings separate from production recommendations.

## Interface review

- [ ] Synchronize `pyproject.toml`, `pointcloud_playground.__version__`, CLI
  version output, documentation, and changelog entries.
- [ ] Review changes to top-level exports, function signatures, CLI commands,
  defaults, primary output filenames, and CSV schemas against the stability
  policy.
- [ ] Document every intentional compatibility change.
- [ ] Confirm that README commands and relative links are valid.

## Local verification

Run from a clean virtual environment:

```bash
python -m pip install -e ".[dev]"
python -m pip check
python -m pytest
python -m build
python -m pip install --force-reinstall --no-deps dist/*.whl
pointcloud-playground --version
pointcloud-playground --help
python experiments/verify_reference_results.py
git diff --check
git status --short
```

- [ ] Visually inspect any changed figures.
- [ ] Review `git diff` for accidental files, unsupported claims, and scope
  expansion.
- [ ] Record test counts and complete reproduction counts in the Release notes.

## Publication

- [ ] Commit with a concise English message and push `main`.
- [ ] Wait for every supported Python-version CI job to pass.
- [ ] Create and push an annotated semantic-version tag on the verified commit.
- [ ] Confirm the tag-triggered CI run.
- [ ] Publish English Release notes covering changes, verification, and scope.
- [ ] Update the GitHub profile README and verify the rendered result.
- [ ] Confirm that local worktrees are clean and synchronized with their
  remotes.
