# NVFP4 Grouped GEMM Follow-up Tasks

This document is the actionable task list derived from the code review of branch
`zhitao/support-nvfp4-group-gemm` vs `origin/main`.

It is written to be directly consumable by a coding agent. Please follow the
instructions below as-is; do not re-plan the scope.

> **Reply language**: please reply to the user entirely in Chinese (中文).
> Code, filenames, and diagnostic messages may stay in English.

---

## 0. Context

- The branch restructured NVFP4 grouped GEMM into
  `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/` to mirror
  `alto/kernels/fp4/mxfp4/mxfp_grouped_gemm/`.
- Tests are now green (307 passed on MI355X / gfx950, torch 2.12a, ROCm 7.2).
- The remaining work is structural cleanup + convergence with MXFP4 conventions.
- **No blocking correctness bugs.** All tasks below are safe, scoped refactors.

**Hard rules**

- Do not change the numerical semantics of any QDQ path.
- Do not change the public `nvfp4_grouped_gemm(...)` or
  `_quantize_then_nvfp4_scaled_grouped_mm(...)` signatures unless the task
  explicitly says so.
- Do not introduce new dependencies.
- Every phase must keep all tests in `alto/kernels/fp4/nvfp4/tests/` green.

---

## 1. Phase 1 — Must land in this follow-up PR (Major)

### Task 1.1 — Remove the duplicated `_offs_to_indices` and its dead imports

**Files**

- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/autograd.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/utils.py`

**What to do**

1. Confirm `_offs_to_indices` is not called anywhere in the repo
   (`rg '_offs_to_indices\('` must return only the definitions themselves).
2. Delete the duplicate local definition in `autograd.py` around lines ~198-200.
3. Delete `from .utils import _offs_to_indices` in `autograd.py`
   (the `_resolve_expert_indices` import alone is enough).
4. Delete the unused definition in `utils.py` (around lines 80-82).
5. Ensure the routing logic still works via `_resolve_expert_indices`.

**Acceptance criteria**

- `rg '_offs_to_indices'` returns **zero** matches inside
  `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/`.
- All NVFP4 tests still pass.

---

### Task 1.2 — Move `_qdq` out of `nvfp_linear.py`

**Goal**

Break the `nvfp_grouped_gemm -> nvfp_linear` reverse dependency. `_qdq` is a
generic NVFP4 QDQ primitive, not a linear-specific helper, so it belongs in
`nvfp_quantization.py`.

**Files**

- `alto/kernels/fp4/nvfp4/nvfp_linear.py` (currently hosts `_qdq`)
- `alto/kernels/fp4/nvfp4/nvfp_quantization.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/cg_backward.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/autograd.py`
- Any other direct callers of `_qdq`

**What to do**

1. Move the `_qdq` function body from `nvfp_linear.py` into
   `nvfp_quantization.py`. Keep the signature, keyword args, and `return_raw`
   semantics **unchanged**.
2. In `nvfp_linear.py`, keep a backward-compat re-export so existing imports
   (`from .nvfp_linear import _qdq`, including test files) do not break:

   ```python
   from .nvfp_quantization import _qdq  # noqa: F401 (re-exported for BC)
   ```

3. Update internal imports in
   `nvfp_grouped_gemm/cg_backward.py` and
   `nvfp_grouped_gemm/autograd.py` to import `_qdq` from
   `alto.kernels.fp4.nvfp4.nvfp_quantization` directly.
4. Do **not** change call sites (same arguments, same return types).

**Acceptance criteria**

- `nvfp_grouped_gemm/` has no import of `alto.kernels.fp4.nvfp4.nvfp_linear`.
- `nvfp_linear.py` does not define `_qdq` anymore (only re-exports it).
- All NVFP4 tests still pass.

---

### Task 1.3 — Collapse the grouped-GEMM autograd alias zoo

**Goal**

The branch already unified two autograd classes into one
(`NVFP4GroupedGEMMFunction`) and kept `NVFP4GroupedGEMM` and
`NVFP4GroupedGEMMNative` as alias. Neither alias is used anywhere outside
this package.

**Files**

- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/autograd.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/__init__.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/functional.py`
- Tests under `alto/kernels/fp4/nvfp4/tests/` (only if any of them reference
  the old alias names)

**What to do**

1. Verify via `rg` that `NVFP4GroupedGEMMNative` is not referenced anywhere
   outside `nvfp_grouped_gemm/__init__.py` and `autograd.py`.
2. Rename `NVFP4GroupedGEMMFunction` to `NVFP4GroupedGEMM` (aligns with
   MXFP4's `MXFP4GroupedGEMM`).
3. Delete the two backward-compat aliases:

   ```python
   NVFP4GroupedGEMM = NVFP4GroupedGEMMFunction
   NVFP4GroupedGEMMNative = NVFP4GroupedGEMMFunction
   ```

4. Remove `NVFP4GroupedGEMMNative` from `__init__.py`'s imports and `__all__`.
5. Update `functional.py`'s import from `.autograd` accordingly (only one
   class is imported).

**Acceptance criteria**

- `rg 'NVFP4GroupedGEMMFunction|NVFP4GroupedGEMMNative'` returns zero hits.
- `__init__.py` exports only the public-facing class + helpers.
- All NVFP4 tests still pass.

---

## 2. Phase 2 — Same PR if cheap, or a small polish PR (Minor)

### Task 2.1 — Clean up `__all__` to stop exporting private helpers

**File**: `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/__init__.py`

**What to do**

- Remove the underscore-prefixed symbols from `__all__`
  (`_nvfp4_grouped_mm_2d_3d`, `_nvfp4_grouped_wgrad`,
  `_has_native_grouped_mm`, `_quantize_then_nvfp4_scaled_grouped_mm`).
- Keep `_quantize_then_nvfp4_scaled_grouped_mm` **as an importable symbol**
  (the dispatch layer imports it from the package), but do not advertise it
  in `__all__`. Everything else should be imported directly from its
  submodule when truly needed (tests, internal code).
- The resulting `__all__` should match the MXFP4 style, e.g.:

  ```python
  __all__ = (
      "ALIGN_SIZE_M",
      "NVFP4GroupedGEMM",
      "nvfp4_grouped_gemm",
      "_quantize_then_nvfp4_scaled_grouped_mm",
  )
  ```

- Update any test file that relied on importing private symbols via the
  package root to import them directly from the submodule.

**Acceptance criteria**

- `__all__` matches the shape used by MXFP4's `mxfp_grouped_gemm/__init__.py`.
- Tests still pass.

---

### Task 2.2 — Rename `_has_native_grouped_mm` to reflect what it actually tests

**File**: `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/utils.py`

**What to do**

- The current implementation is:

  ```python
  _GROUPED_MM_AVAILABLE = bool(torch.cuda.is_available() and is_cdna4())
  ```

  It has nothing to do with `torch._grouped_mm`. Rename the function to
  something semantically accurate, e.g. `_use_cdna4_grouped_backend()`.
  Update all internal call sites in
  `nvfp_grouped_gemm/cg_forward.py`, `cg_backward.py`, `autograd.py`, and the
  tests that call it directly.

- Add a small module-level docstring explaining that this flag selects the
  CDNA4 Triton grouped backend (via `blockwise_fp8.cg_grouped_gemm_*`), not
  `torch._grouped_mm`.

- Add a lightweight test helper
  `_reset_cdna4_grouped_backend_cache()` (still underscore, still not in
  `__all__`) so future tests that monkey-patch `is_cdna4` can invalidate the
  cached boolean.

**Acceptance criteria**

- Grep for `_has_native_grouped_mm` returns zero matches.
- Tests pass.

---

### Task 2.3 — Kill transpose + `.contiguous()` round-trips on the wgrad path

**Files**

- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/autograd.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/cg_forward.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/cg_backward.py`
- `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/functional.py`

**Background**

The dispatch path hard-codes `trans_weights=False`, so the following lines
trigger **one transpose + contiguous copy per step**:

- `autograd.py` - `w_for_dgrad = w_bwd if ctx.trans_weights else w_bwd.transpose(-2, -1).contiguous()`
- `cg_backward.py` - `if not trans_weights: dw = dw.transpose(-2, -1).contiguous()`
- `cg_forward.py` - `cg_grouped_gemm_forward(A.contiguous(), B.transpose(-2, -1).contiguous(), ...)`

**What to do**

1. Normalize weights once, at a single place (`functional._nvfp4_grouped_gemm_impl`
   is the natural choice). Decide on a single internal layout for the autograd
   function (e.g. `expert_weights` always `[E, K, N]`), and document it in the
   autograd class docstring.
2. Push the `trans_weights` handling entirely to the wrapper layer:
   - `nvfp4_grouped_gemm(..., trans_weights=True)` input transposes once
     into the internal layout before calling `.apply`.
   - `_quantize_then_nvfp4_scaled_grouped_mm(...)` already receives
     `[E, K, N]` and needs no transpose.
3. Remove the per-step transpose from `cg_forward.py` and `cg_backward.py`.
4. Remove the per-step `.contiguous()` calls inside `cg_forward.py` where the
   upstream tensor is guaranteed contiguous (e.g. produced by `_qdq`).
5. Keep defensive `.contiguous()` **only** at places where non-contiguous
   tensors can legitimately arrive (e.g. slice/view inputs from the user).

**Acceptance criteria**

- No `.transpose(-2, -1).contiguous()` is called on a per-step tensor inside
  `cg_forward.py`, `cg_backward.py`, or `autograd.backward`.
- All NVFP4 tests still pass, including
  `test_nvfp4_grouped_gemm_autograd` (which asserts dX / dW numerics vs.
  the BF16 reference).

---

### Task 2.4 — Clean up dead imports in `functional.py`

**File**: `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/functional.py`

**What to do**

- Remove unused imports from `.autograd`: at minimum `_nvfp4_grouped_wgrad`
  (and `NVFP4GroupedGEMMNative` once Task 1.3 is done).
- Import `ALIGN_SIZE_M` from `.autotune` directly rather than going through
  `.autograd`.

**Acceptance criteria**

- `functional.py`'s imports contain only symbols actually referenced in that
  file.

---

### Task 2.5 — Rename `autotune.py` **or** justify keeping the name

**File**: `alto/kernels/fp4/nvfp4/nvfp_grouped_gemm/autotune.py`

Today the file only hosts `ALIGN_SIZE_M` and `QDQ_AXIS0_BLOCK_SIZE`. The name
suggests Triton autotune configs (which it does not have).

**Choose one**

- **Option A** (preferred if you plan to keep the NVFP4 backend emulation-only
  for a while): rename to `constants.py`, update imports everywhere.
- **Option B** (preferred if a native NVFP4 Triton grouped kernel is coming
  soon): keep the name, add a docstring like:

  ```python
  """Grouped GEMM autotune / shared constants.

  Currently only hosts ``ALIGN_SIZE_M`` and ``QDQ_AXIS0_BLOCK_SIZE``. The
  file name is kept for structural parity with
  ``alto/kernels/fp4/mxfp4/mxfp_grouped_gemm/autotune.py``; Triton autotune
  configs will land here once a native NVFP4 grouped kernel is added.
  """
  ```

State in the PR description which option you picked and why.

---

## 3. Phase 3 — Nits (optional, batch into a separate polish PR if needed)

- Add `# forward inputs: ...` comment next to the long tuple of `None`s in
  `NVFP4GroupedGEMM.backward` so the arity alignment is obvious.
- Extend `trans_weights` parameterization in
  `tests/test_nvfp_grouped_gemm.py::test_nvfp4_grouped_gemm_forward_accuracy`
  to include `False`.
- Replace `torch.randint(...).item()` loops in
  `_make_contiguous_expert_indices` and `_bf16_grouped_ref_forward` with
  vectorized constructions.
- In `TrainingWeightWrapperBaseTensor`, consider a `_dispatch_grouped_mm`
  template method so MXFP4 and NVFP4 subclasses don't copy-paste the
  2D x 3D + offs assertion block.
- In `mxfp4/__init__.py`, consider adding `use_per_tensor_scale=False`
  (ignored) to `_quantize_then_mxfp_scaled_grouped_mm` so the MXFP4 / NVFP4
  dispatch helpers have identical signatures.

Do **not** start Phase 3 until Phase 1 is merged.

---

## 4. Verification plan

After each task, the agent must:

1. Run `rg` to prove the old symbol / import is gone.
2. Re-run the NVFP4 test suite. The expected baseline is:

   ```
   python3 -m pytest alto/kernels/fp4/nvfp4/tests/ --no-header -q
   # expected: 307 passed
   ```

   (Run inside the same environment used previously:
   MI355X / gfx950, torch 2.12a, ROCm 7.2.)
3. If any test changes behavior, explain in the report why it's an expected
   consequence of the refactor.

When all Phase 1 tasks are complete, the agent should additionally run:

```
python3 -m pytest alto/kernels/fp4/nvfp4/tests/test_nvfp_dispatch_guards.py \
                  alto/kernels/fp4/nvfp4/tests/test_nvfp_grouped_gemm.py \
                  alto/kernels/fp4/nvfp4/tests/test_nvfp_linear.py \
                  alto/kernels/fp4/nvfp4/tests/test_nvfp_quantization.py \
                  --no-header -v
```

and paste the summary line into the final report.

---

## 5. Commit / PR hygiene

- Split the work into one commit per Phase 1 task where possible. Do not
  squash the `_qdq` move and the alias cleanup into a single commit.
- Commit messages: title only, no body, no Cursor footer or any generated
  signature.
- Do **not** commit anything under `scripts/`.
- Do **not** commit artifacts under `.artifacts/` or any loss JSON / PNG
  file.
- If a refactor touches a test file, the commit title should say so
  explicitly, e.g. `tests: nvfp_grouped_gemm: drop NVFP4GroupedGEMMNative alias`.

---

## 6. Expected final output (for the agent to produce)

Reply in Chinese, structured as:

1. 任务完成清单（Phase 1 / 2 / 3 各完成了哪些）
2. 每个 task 实际改了哪些文件 + 核心改动说明
3. `rg` 结果摘要（证明死代码 / 老名字已清理干净）
4. `pytest` 运行结果摘要
5. 本次故意未做的 task（如有），以及原因
6. Commit 列表（title + 覆盖的 task 编号）

Do not echo this document back verbatim; only answer the items above.
