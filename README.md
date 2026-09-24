# Hermes Kev Router

Hermes plugin that asks a local [Kev](https://github.com/jaredpalmer/kev) System One endpoint for
the narrowest capability profile needed by each user turn, then filters recognized Hermes built-in
tool schemas through the documented `llm_request` middleware.

It does not patch Hermes core. `hermes update` and plugin updates remain independent.

## Safety and compatibility

- Classification runs once per user turn and is reused for retries and tool-loop requests.
- Any timeout, malformed response, uncertainty, or plugin error fails open to Hermes' original request.
- Unknown tools are preserved by default, including external plugin and MCP tools.
- Skill tools remain available, so Hermes' normal skills index and skill workflow are unchanged.
- Runtime authorization, approval prompts, execution controls, and tool registration are untouched.
- Tool filtering is a schema-disclosure optimization, not a security boundary.

Changing tool schemas between turns trades some cross-turn tool-prefix cache stability for smaller
requests. Hermes' system prompt remains byte-stable.

## Requirements

- Hermes Agent 0.21 or newer
- A reachable Kev System One endpoint
- No Python dependencies outside the standard library

Example CPU Kev endpoint:

```text
http://127.0.0.1:8009/v1/systemone
```

## Install

Until this repository is admitted to the Hermes Plugin Catalog, install it directly:

```bash
hermes plugins install nmavor/hermes-kev-router
hermes plugins enable hermes-kev-router
```

Restart a running Hermes gateway after initial installation so it loads the plugin.

## Configuration

Settings live under `plugins.entries.hermes-kev-router.settings`:

```yaml
plugins:
  entries:
    hermes-kev-router:
      settings:
        enabled: true
        endpoint: http://127.0.0.1:8009/v1/systemone
        model: kev-latest
        timeout: 1.25
        min_probability: 0.45
        min_confidence: 0.30
        min_margin: 0.15
        max_state_chars: 12000
        preserve_unknown_tools: true
```

`preserve_unknown_tools: true` is strongly recommended. Disabling it also hides schemas the plugin
does not recognize and can interfere with independently installed plugins.

## Profiles

- `chat`: clarification and skill tools
- `research`: web, reading, session search, vision, clarification, and skill tools
- `coding`: terminal, files, search, execution, delegation, planning, and skill tools
- `browser`: browser, web, vision, clarification, and skill tools
- `personal`: memory, connected services, media, Home Assistant, and skill tools
- `automation` and `full`: original Hermes tool list

Unknown external tools remain present in every profile by default.

## Development

```bash
python -m unittest discover -s tests -v
hermes plugins validate .
```
