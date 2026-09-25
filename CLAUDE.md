# CLAUDE.md

Project-specific guidance for Claude Code sessions working in this repository.

## Project overview

- Read [`concept.md`](concept.md) first: it describes the big picture and is the reference for the rewrite.
- `legacy_code/` and `data/` are reference samples only. Don't refactor or extend the legacy code; the program is being rewritten from scratch.

## Communicating with the user

- Be concise: short, direct explanations; cut restated context and options you are not pursuing.
- Be explicit, not cryptic: define any short id, abbreviation or jargon on first use, and prefer a plain word over unexplained shorthand.

## Code style

- Lines over 120 characters are fine; don't wrap purely for line length.
- Prefer descriptive, self-explanatory names, even if long.
- CamelCase for class names, snake_case for functions and variables.
- Add short, free-form docstrings to all classes and functions (no mandated Args/Returns sections).
- `###` marks permanent comments (explanations, rationale); plain `#` is only for temporarily commented-out code.
- Prefer well-established libraries over custom code. Manage non-stdlib dependencies via Miniforge/conda-forge, not pip.
- Add type hints only where they aid readability or catch likely mistakes.
- Keep multi-argument signatures and calls on one line rather than one argument per line.
- Error handling: this is research code, so an uncaught exception with a traceback is acceptable. Don't wrap calls in defensive `try`/`except`; instead make exception messages helpful (include the offending value, filename or context). Only catch an exception when you can recover or add real context.
- Logging: use `rich`, configured once by a single setup function that installs a `rich` handler on the root logger (default level `info`). Its level and `file:line` output is enough; don't add the logger name to the format. Use `info` only for logging that doesn't slow the program down (e.g. the assembled config, worth recording); hot paths and inner loops log at `debug`. Log nested structures readably (e.g. block-style YAML).
- CLI sub-options: where reasonable, a CLI argument may accept repeated `X=Y` values (e.g. `-a/--my-arg X=Y`), collected into a dict. Parse `Y` with YAML so it can be any type, and split on the *first* `=` only so values may contain `=`:
```
some_arg_opt = dict([(lambda k,v: (k, yaml.safe_load(v)))(*elem.split("=", 1)) for elem in args.some_arg])
```
  - Dotted keys attach a parameter to another sub-option: `--vis out=result.html out.image_max_px=1100` gives `out` (the path) and a sibling key read via `vis.get("out.image_max_px")`. This falls out of the same parse. Keep such parameters optional with sensible defaults.
- Config: nested YAML plus `**`-splatting at every level, so each entry is documented (name + default) in the signature of the function that consumes it — the signature *is* the schema documentation:
  - Pass the whole config to the main class as `SomeClass(**config)`: each top-level section maps to one named `__init__` parameter receiving that section's sub-dict. Store each section as its own attribute rather than merging them.
  - Splat each section into the methods that use it via `method(**section)`, with individual entries as named keyword parameters.
  - At every level, give parameters sensible defaults and accept `**kwargs` so unknown entries are absorbed.
  - Assemble and mutate the config only in the CLI/config layer (load YAML, apply overrides); consumers just receive it.
  - Keep entry names unique across sections splatted into the same call (`f(**section_a, **section_b)`), since duplicate keywords raise `TypeError`.
  - The same applies to module-level functions:
```
### config = {"site": {...}, "weather": {"source": "tmy"}, "simulation": {...}}

### A free function can consume one or more sections splatted together (unique entry names):
def load_weather(latitude=None, longitude=None, source="tmy", **kwargs):   # each entry named + documented here
    ...
weather = load_weather(**config["site"], **config["weather"])   # sections -> function kwargs

### The main class receives already-produced inputs plus the sections it needs; extra sections are absorbed:
estimator = YieldEstimator(weather=weather, **config)   # top-level sections -> __init__ kwargs

def compute_irradiance(self, time_step_minutes=60, **kwargs):   # each entry named + documented here
    ...
self.compute_irradiance(**self.simulation)                     # section sub-dict -> method kwargs
```
