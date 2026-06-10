# Developer notes

## How to add support for a new version

### Updating code
Normally you need to copy the `_search.py` file from `sklearn`. In there you need to set up the `RandomizedSearchCV` and `GridSearchCV` classes to accept the `spark` argument. 
SparkBaseSearchCV should also be created, and that needs to implement `fit()`.

### Setting up tests
There is a bash script for running all tests `scikit-spark/python/run-tests.sh`.

## Running tests

Tasks are defined in `Taskfile.yml` ([go-task](https://taskfile.dev)).

Prerequisites:
- [`task`](https://taskfile.dev) to run the tasks
- [`act`](https://github.com/nektos/act) and Docker to run the CI pipeline locally

Commands:
- `task test:fast` — run the suite against your local Python/Spark environment (fast).
- `task test:act` — run the full GitHub Actions pipeline locally in Docker via `act`.
- `task test:act:fast` — faster iterative `act` run that reuses containers and skips the image pull. Non-hermetic: do a plain `task test:act` before trusting a green result.

The `act` runner image is pinned in `.actrc`.

## Releasing

The version is single-sourced from the `version` field in `pyproject.toml`;
`skspark.__version__` reads it back at runtime via `importlib.metadata`. There is
no version string to edit anywhere else.

Releases go out of CI via PyPI Trusted Publishing (OIDC) — no tokens are stored.

### Cutting a release
1. In a PR: bump `version = "x.y.z"` in `pyproject.toml` and add a `# x.y.z`
   section to `changelog.md` describing the work.
2. Merge to `master`. The CI pipeline runs the test matrix, then publishes
   `x.y.z.dev<run_number>` to **TestPyPI** as a packaging smoke test (unique per push).
3. Go to **Actions → CI → Run workflow** (on `master`). The `publish-pypi` job:
   - re-runs the full test matrix,
   - validates `vx.y.z` is not already a git tag and `x.y.z` is not already on PyPI,
   - builds, then creates and pushes the annotated tag `vx.y.z`,
   - publishes to **PyPI**,
   - creates a GitHub Release with notes from the matching `changelog.md` section.

Everything reversible (build, tag) happens before the irreversible PyPI publish. If
publish fails, delete the dangling tag (`git push origin :vx.y.z`) and re-run.

After releasing, bump `pyproject.toml` to the next target version so master's
TestPyPI dev builds represent the upcoming release rather than the shipped one.

### Local packaging check
`task build` runs `python -m build && twine check dist/*` to validate packaging
before pushing.

### One-time setup (already done — for reference)
- **PyPI** → `scikit-spark` → Publishing → Trusted Publisher: owner `scikit-spark`,
  repo `scikit-spark`, workflow `pipeline.yml`, environment `pypi`.
- **TestPyPI** → pending publisher: same repo, workflow `pipeline.yml`, environment
  `testpypi` (creates the project on first publish).
- **GitHub** → Settings → Environments → `pypi` and `testpypi`.

These must exist before the first CI publish or OIDC auth fails.

See the `get_refactored_tests_to_skip` function. For each version it has a list of tests which need different handling to work correctly. Due to the way tests are added to `AllTests` it is not possible to use the `sklearn` method for parameterising the tests, so they have to be moved into `TestParameterisedTests`.  In most cases the original test can just be imported and called with the appropriate parameters.
