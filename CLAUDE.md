# CLAUDE.md

Project-specific guidance for Claude Code sessions working in this repository.

## Project overview

- Read [`concept.md`](concept.md) first: it describes the big picture (purpose, platforms, inputs, outputs, pipeline, technical decisions) and is the reference for the rewrite.
- `legacy_code/` and `data/` hold sample code and sample data for reference only. Don't refactor or extend the legacy code; the program is being rewritten from scratch.

## Communicating with the user

- Be concise. Prefer short, direct explanations over long ones; cut restated context and options you are not pursuing.
- Be explicit, not cryptic. When you introduce a short id or abbreviation (e.g. per-location labels, compass tags like "SE", jargon like "front-end"), define it the first time you use it, and prefer a plain word over an unexplained shorthand. The reader should never have to guess what a token stands for.

## Code style

- Lines longer than 120 characters are fine; don't wrap purely for line length.
- Prefer descriptive, self-explanatory names for variables/functions/classes over short/cryptic ones, even if that makes them long.
- Use camel case for class names and underscores for function/variable names
- Add docstrings to all classes and functions; keep them short and free-form (no mandated Args/Returns sections).
- Use `###` for comments meant to stay in the code permanently (explanations, rationale). Plain `#` is reserved for temporarily commented-out code.
- Favor using existing libraries over writing custom code when a well-established one exists. Manage non-stdlib dependencies via Miniforge/conda-forge rather than pip.
- Add type hints only where they meaningfully aid readability or catch likely mistakes — not required everywhere.
- Error handling: this is research code, so an uncaught exception with a helpful message (and a traceback) is a perfectly acceptable failure mode. Don't wrap calls in defensive `try`/`except` just to suppress or prettify tracebacks. Spend the effort on making the exception *message* helpful — include the offending value, filename, or context — and otherwise let exceptions propagate. Only catch an exception when you can genuinely act on it (recover, or add real context).
- Logging: use `rich` for log output (configure it once via a single logging setup function that installs a `rich` handler on the root logger; the default level is `info`). The rich handler shows each record's log level and source `file:line`, which is enough for debugging — don't add the logger name/namespace to the format. Reserve the `info` level for logging that does not slow the program down in any relevant way — one-time setup such as the initial/assembled configuration is a good fit for `info` (and worth recording), whereas anything in hot paths or inner loops belongs at `debug`. When logging a nested structure (e.g. the config), format it readably (e.g. block-style YAML) so the nesting is visible.
- Keep multi-argument function signatures/calls on a single line rather than one argument per line; avoid excessive vertical sprawl.
- CLI sub-options: CLI arguments may - where reasonable - accept repeated `X=Y` sub-option values (e.g. `-a/--my-arg X=Y`), collected into a dict for that argument. This is a general pattern, not tied to a specific flag naming scheme. Parse the value side (`Y`) with YAML so it can represent any type (numbers, bools, lists, nested structures), not just strings. Split on the *first* `=` only, so a value may itself contain `=`. Consider the following snippet for this:
```
some_arg_opt = dict([(lambda k,v: (k, yaml.safe_load(v)))(*elem.split("=", 1)) for elem in args.some_arg])
```
  - Dotted keys parameterize a base sub-option: to attach a parameter to another sub-option, prefix the key with that sub-option's name (`base.param=Y`), e.g. `--vis out=result.html out.image_max_px=1100` — `out` holds the value (the path) and `out.image_max_px` is a sibling key read alongside it (`vis.get("out.image_max_px")`). Splitting on the first `=` keeps the dotted key intact, so this falls out of the same parse with no extra work; there is no string-vs-dict conflict because base and parameter are separate flat keys. Keep such parameters optional with a sensible default (a missing key means "unset"), matching the config convention's defaults.
- Config handling uses nested YAML plus a `**`-splatting convention at every level, so each config entry is documented (name + default) right at the function that consumes it — the signature *is* the schema documentation:
  - Pass the whole config into the main class as `SomeClass(**config)`, so the config's top-level section keys map onto `__init__`'s named keyword parameters (one parameter per section, each receiving that section's sub-dict).
  - In turn, splat each section sub-dict into the method(s) that use it via `method(**section)`, where the individual entries are named keyword parameters.
  - At every level, give named parameters reasonable defaults so missing config entries don't break anything, and accept `**kwargs` so extra/unknown entries are absorbed rather than raising. Store each section as its own attribute rather than merging everything into one dict.
  - Assemble and mutate config *outside* the consuming class/functions — load the YAML and apply any on-the-fly overrides in the CLI/config layer, then pass the finished mapping in. Consumers just receive it; they don't build or mutate it.
  - Since sections may be splatted together into one call (`f(**section_a, **section_b)`), keep entry names unique across sections that feed the same function — duplicate keywords raise `TypeError`.
The same convention applies to module-level functions, not just methods — e.g. data
loading is a free function fed the relevant sections, splatted together:
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
