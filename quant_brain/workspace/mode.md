# Mode Override

This file is optional. Agents should infer the working mode from the current
prompt by default.

## Current Mode

- Mode: `auto`
- Last Updated: `2026-05-21`
- Notes: `Use only as a strong hint when the prompt context is ambiguous.`

## Allowed Values

- `auto`
- `manual_trade_discussion`
- `trade_review`
- `systematization`
- `systematic_research`

## Behavior

- If this file is present and recently updated, agents may treat it as a strong override hint.
- If it is absent or stale, agents should infer mode from the prompt.
- Users may still use inline cues such as `Mode: manual idea`,
  `Mode: trade review`, `Mode: systematize`, or `Mode: systematic`.
