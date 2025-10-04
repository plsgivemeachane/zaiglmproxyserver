# GLM Timeout Propagation Fix

## Overview
Requests to the GLM API were timing out at 120s even when a 30s timeout was configured via environment variables or hardcoded in some places. Root causes:

- OpenAI wrapper `OpenAIToGLMWrapper` constructed `GLMAPIClient` incorrectly by passing a base URL as the first argument (interpreted as an authorization token), so the client's `timeout` fell back to its default of 120s.
- The API server only propagated `GLM_TIMEOUT` to the V2 wrapper. V1 and V3 wrappers were using their defaults (120s for underlying client), creating inconsistency.
- Default timeout comments in docs suggested 120s while the server defaulted to 15s if `GLM_TIMEOUT` was not set.

## Goals
- Ensure a single source of truth for request timeout via `GLM_TIMEOUT`.
- Propagate the configured timeout to all wrappers/clients (V1, V2, V3).
- Add startup logging for the effective timeout to aid diagnostics.

## Changes
- `openai_glm_wrapper.py`: Change `OpenAIToGLMWrapper.__init__` to accept `timeout` and pass it to `GLMAPIClient(timeout=timeout)`.
- `openai_glm_server.py`: Read `GLM_TIMEOUT` once and pass it to all wrappers: V1, V2, and V3. Log the effective timeout at startup.
- No API contract changes for endpoints; behavior is now consistent with configuration.

## Configuration
- Set the desired timeout in `.env`:

```env
GLM_TIMEOUT=30
```

- Restart the server to apply changes.

## Expected Results
- All GLM requests observe the configured timeout (e.g., 30s).
- Error messages will correctly reflect the configured timeout value.
- Startup logs will include: `GLM effective request timeout set to <value> seconds`.
