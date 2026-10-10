# gray-openai-compat

Shared OpenAI-Chat-Completions ↔ OpenAI-Responses translation for gray
provider sidecars — one `compat_turn` call and any OpenAI-compatible
upstream becomes a native gray provider.

Built for (and by) `gray-ollama-sub`, `gray-lmstudio-sub`,
`gray-deepseek-sub`, `gray-qwen-sub`, `gray-kimi-sub`, `gray-zai-sub`,
`gray-gemini-sub`, `gray-copilot-sub`.

## API surface

| Item | Role |
|---|---|
| `Compat` | Upstream config: `base` URL, optional `api_key`, extra headers. |
| `Compat::new(base, key)` / `.header(k, v)` | Builder. |
| `credential_key(params)` | Walks the provider `CredentialEnvelope`/`SecretMap` and pulls the API key without ever logging it. |
| `to_chat_request(turn, model, body)` | Responses request → chat-completions body: instructions → system, `input_text`/`input_image` parts → content arrays, tools → `tools`, `function_call_output` → `role: "tool"`. |
| `to_turn_result(v)` | Completion → `TurnResult` (text or folded `tool_calls`, usage). |
| `post_chat(cfg, body)` | Blocking POST to `{base}/chat/completions` (non-streaming — the relay synthesizes SSE). |
| `probe_models(cfg)` | `GET {base}/models` → catalog rows. |
| `compat_turn(...)` | The one-call integration: plug it into `acp_bridge::relay::mint_turn`'s `TurnFn` and the whole provider path is done. |

## The shape

```
host ── Responses POST ──▶ mint_turn relay ──▶ compat_turn
        ◀── Responses SSE ──      │      translates → upstream POST
                                  ▼
                          {base}/chat/completions
```

Streaming text, tool calls, and usage arrive on the host side as canonical
Responses events whatever the upstream speaks.

## License

MIT
