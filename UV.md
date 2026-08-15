# uv cheatsheet

## New clone (first time)

```powershell
git clone <repo-url> AMMORE
cd AMMORE
uv sync                      # creates .venv + installs from uv.lock
uv sync --extra dev          # add ruff / pytest / pyright
cp .env.example .env         # then add API keys
```

`uv sync` creates `.venv` itself — no `uv venv` needed.

## Returning to an existing repo

```powershell
cd AMMORE
uv sync                      # only if pyproject/uv.lock changed
```

Nothing changed? Nothing to do.

## Activating

```powershell
.\.venv\Scripts\activate     # Windows
source .venv/bin/activate    # Linux / macOS
deactivate                   # exit
```

## Skip activation (preferred)

`uv run` uses the project venv automatically:

```powershell
uv run python -m ammore.server
uv run ruff check ammore/
uv run pytest
```

## Dependencies

```powershell
uv add <pkg>                 # runtime dep -> pyproject + lock + install
uv add --optional dev <pkg>  # into the dev extra
uv remove <pkg>
uv lock                      # re-resolve only, venv untouched
uv lock --upgrade            # bump to newest allowed
uv sync --dry-run            # preview changes before applying
```

Never `pip install` into `.venv` — it drifts from `uv.lock` and the next
`uv sync` silently reverts it.

## Gotchas

| Symptom | Cause / fix |
|---|---|
| `No module named pip` | Normal — uv venvs omit pip. Use `uv add` / `uv run`. |
| `uv add X` fails to resolve | Whole graph must resolve, not just `X`. Check the named conflict. |
| `milvus-lite ... no wheel for win_amd64` | mmore ≥1.2.4 drops its platform marker; the `[tool.uv] override-dependencies` entry in `pyproject.toml` restores it. |
| Ruff/pytest missing after `uv sync` | Extras are skipped by default: `uv sync --extra dev`. |
| Code edits not taking effect | Long-running process holds old code — restart it. |
