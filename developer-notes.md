# Developer notes

## How scikit-learn version support works

scikit-spark vendors a patched copy of scikit-learn's
`model_selection/_search.py` for each supported scikit-learn **minor** version,
under `python/skspark/sklearn_1_<minor>/_search.py` (e.g. `sklearn_1_4`).
`skspark/model_selection.py` inspects the installed `sklearn.__version__` and
imports the matching package; unsupported versions raise `NotImplementedError`.

Each vendored `_search.py` is that release's upstream `_search.py` reduced to
the search classes and re-patched for Spark. The patch ("the delta") is:

- Import `SparkSession` (pyspark) and `product` (itertools), and import the
  unchanged base pieces (`BaseSearchCV`, `ParameterGrid`, `ParameterSampler`,
  `_fit_and_score`, etc.) from `sklearn` rather than redefining them. **The
  internal import paths/names drift between sklearn versions — this is the main
  thing that breaks when adding a version** (e.g. `_check_fit_params` →
  `_check_method_params`; `joblib`/`utils.fixes` → `utils.parallel`).
- Add `SparkBaseSearchCV(BaseSearchCV)` carrying: the `spark` constructor kwarg
  (stored as `self.spark`); `_run_sklearn_fit` (the upstream evaluation loop);
  `_run_skspark_fit` (the same loop distributed via `parallelize`/`broadcast`);
  an overridden `fit()`; and `__getstate__` to drop the unpicklable
  `SparkSession`.
- Make `GridSearchCV`/`RandomizedSearchCV` subclass `SparkBaseSearchCV`, add
  `spark=True` to their `__init__`, and pass it through to `super()`.
- `fit()` must mirror **that version's** upstream `BaseSearchCV.fit()` body while
  routing candidate evaluation through `_run_skspark_fit`.

### How to add support for a new scikit-learn version

Work one version at a time. The example below adds **1.9** (substitute the real
target minor and its latest patch release, e.g. `1.9.0`):

1. **Copy the newest existing package** as the starting point:

   ```sh
   cp -r python/skspark/sklearn_1_8 python/skspark/sklearn_1_9
   ```

2. **Fetch both upstream `_search.py` files** — the new release and the release
   the package you copied was based on (here 1.8) — and diff them. Use the git
   tags on the scikit-learn repo:

   ```sh
   base=1.8.0; new=1.9.0
   for v in $base $new; do
     curl -sSL "https://raw.githubusercontent.com/scikit-learn/scikit-learn/$v/sklearn/model_selection/_search.py" -o "/tmp/u$v.py"
   done
   diff -u "/tmp/u$base.py" "/tmp/u$new.py"
   ```

3. **Re-apply only the deltas that touch the vendored regions.** Most of the
   diff is docstrings and code we don't vendor (it lives in the imported
   `BaseSearchCV`/`ParameterGrid`/etc.). The parts that matter:
   - the **import block** (internal module paths/names drift — this is the most
     common break);
   - the body of **`BaseSearchCV.fit()`** (mirror it in `SparkBaseSearchCV.fit()`,
     keeping the dispatch to `_run_skspark_fit`/`_run_sklearn_fit`);
   - any **helper methods `fit()` calls** (`_get_scorers`,
     `_get_routed_params_for_fit`, `get_metadata_routing`, etc.) — copy new ones
     and update changed signatures;
   - the **`GridSearchCV`/`RandomizedSearchCV` `__init__` and
     `_parameter_constraints`**.

   To confirm the class constructors/constraints didn't change you can diff just
   those regions, e.g.:

   ```sh
   for v in $base $new; do awk '/^class GridSearchCV/,0' "/tmp/u$v.py" \
     | grep -A40 "def __init__" | sed '/"""/q' > "/tmp/init_$v.txt"; done
   diff /tmp/init_$base.txt /tmp/init_$new.txt   # empty == unchanged
   ```

4. **Register the version**: add the minor to `_SUPPORTED_MINORS` in
   `python/skspark/model_selection.py`.

5. **Test it in isolation**: set the `sklearn-version` matrix list in
   `.github/workflows/pipeline.yml` to just the new version, run `task test:act`,
   and fix any per-version test handling in `python/test/conftest.py` (see
   below). New scikit-learn releases add tests to `test_search.py`; expect a few
   to need a version-gated skip or a fixture they rely on.

6. Once green in isolation, **add it back to the full matrix list** and run the
   whole matrix once to confirm nothing regressed.

7. On release, bump the scikit-spark version (`setup.py`,
   `python/skspark/__init__.py`) and add a `changelog.md` entry.

Notable historical break points (what changed in `fit()`/imports at each minor —
use as a checklist for what a new version is likely to touch):
`_parameter_constraints` / `_validate_params` arrived in **1.2** (1.1 predates
them and uses `_required_parameters`); `@_fit_context` in **1.3**; full metadata
routing (`process_routing`, `routed_params`, `get_metadata_routing`,
`_check_method_params`) in **1.4**; `_get_scorers()` signature change in **1.5**;
scorer `sample_weight` forwarding (`_check_scorers_accept_sample_weight`) in
**1.7**; `MetadataRouter(owner=self)` in **1.8**.

### Per-version test handling

The suite re-runs scikit-learn's own `test_search.py` against the Spark classes
(`python/test/test_sklearn.py` does `from sklearn.model_selection.tests.test_search
import *`, and `conftest.py` monkey-patches `GridSearchCV`/`RandomizedSearchCV`
to the skspark versions). Because the imported tests change between releases,
`conftest.py` maintains a `tests_to_skip` list, with version-gated entries for
tests that are genuinely incompatible with Spark (e.g. stateful scorers whose
call-counter resets per worker) or with the modern numpy/scipy stack on very old
sklearn (e.g. 1.1's NaN ranking). `conftest.py` also provides an
`enable_slep006` fixture (scikit-learn defines it in its own, unloaded,
`conftest.py`) so the 1.4+ metadata-routing tests can run.

## Pinning constraints

- **numpy `<2`**: scikit-learn 1.1-1.3 wheels are built against the numpy 1.x
  ABI and crash on numpy 2; numpy 1.26 is runtime-compatible across 1.1-1.8.
- **Python 3.11**: covers the whole 1.1-1.8 range. scikit-learn 1.0 has no 3.11
  wheel and 1.8 requires Python >= 3.11, so 1.0 is out of scope.

## Running tests

Tasks are defined in `Taskfile.yml` ([go-task](https://taskfile.dev)).

Prerequisites:
- [`task`](https://taskfile.dev) to run the tasks
- [`act`](https://github.com/nektos/act) and Docker to run the CI pipeline locally

Commands:
- `task test:fast` — run the suite against your local Python/Spark environment (fast).
- `task test:act` — run the full GitHub Actions pipeline locally in Docker via `act`.
- `task test:act:fast` — faster iterative `act` run that reuses containers and skips the image pull. Non-hermetic: do a plain `task test:act` before trusting a green result.

The `act` runner image is pinned in `.actrc`. To test a single scikit-learn
version locally, set the `sklearn-version` matrix list in
`.github/workflows/pipeline.yml` to just that version before running `act`.
