# Vendored IFEval scorer

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval

Upstream commit: `09446d32b45c1d69e6bfff2ebce8638a6c05f886`

License: Apache 2.0 (see `LICENSE`).

## Files

- `evaluation_lib.py`, `evaluation_main.py` — scoring entry points
- `instructions.py`, `instructions_registry.py`, `instructions_util.py` — per-instruction verifiers

## Modifications

The only changes from upstream are four import-line fixes so the module
can be imported as a package (`third_party.ifeval.*`) rather than as a
Google-research-internal package:

- `evaluation_main.py`: `from instruction_following_eval import evaluation_lib` → `from . import evaluation_lib`
- `evaluation_lib.py`: `from instruction_following_eval import instructions_registry` → `from . import instructions_registry`
- `instructions_registry.py`: `from instruction_following_eval import instructions` → `from . import instructions`
- `instructions.py`: `from instruction_following_eval import instructions_util` → `from . import instructions_util`

## Dependencies

The upstream `requirements.txt` lists: `absl`, `langdetect`, `nltk`, `immutabledict`.
These are declared in the project's main `pyproject.toml` so a single
`uv sync` installs everything needed. One-time NLTK data download is also
required:

```bash
uv run python -c "import nltk; nltk.download('punkt_tab')"
```
