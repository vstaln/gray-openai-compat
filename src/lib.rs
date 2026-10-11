//! gray-openai-compat — shared translation between the host's OpenAI
//! Responses-API requests and upstream OpenAI Chat-Completions endpoints
//! (Ollama, LM Studio, DeepSeek, Qwen/DashScope, Kimi/Moonshot, Z.AI,
//! Gemini's /openai surface, GitHub Copilot).
//!
//! One [`compat_turn`] plugs into `acp_bridge::relay::mint_turn`: the
//! relay serves the host's POST, we translate, POST upstream
//! (non-streaming — the relay synthesizes the SSE stream), and fold the
//! completion back through `acp_bridge::chat::fold_result`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use acp_bridge::chat::{self, OutItem, PreparedTurn, TurnResult, Usage};
use acp_bridge::relay::TurnFn;
use serde_json::{json, Value};

/// One upstream endpoint's wiring.
#[derive(Clone)]
pub struct Compat {
    /// e.g. "https://api.deepseek.com/v1" — no trailing slash.
    pub upstream_base: String,
    /// Bearer for the upstream; `None` = no auth header (local servers).
    pub api_key: Option<String>,
    /// Extra headers on every upstream call (Copilot's `Editor-*`, …).
    pub headers: Vec<(String, String)>,
    /// Upstream request timeout.
    pub timeout: Duration,
}

impl Compat {
    pub fn new(base: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            upstream_base: base.into().trim_end_matches('/').to_string(),
            api_key,
            headers: Vec::new(),
            timeout: Duration::from_secs(300),
        }
    }

    pub fn header(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.headers.push((k.into(), v.into()));
        self
    }
}

/// Extract a usable API key from a `provider/*` request's credential
/// envelope (`params.credential.credential.secrets`). Secret names differ
/// per auth method — try the common ones, then any non-empty value.
pub fn credential_key(params: &Value) -> Option<String> {
    let secrets = params
        .pointer("/credential/credential/secrets")?
        .as_object()?;
    for name in ["api_key", "key", "token", "access_token", "secret"] {
        if let Some(v) = secrets.get(name).and_then(Value::as_str) {
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    secrets
        .values()
        .find_map(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Content parts → chat-completions content. Text collapses to a bare
/// string (widest upstream compat); anything with images stays a parts
/// array in OpenAI image_url shape.
fn content_to_chat(content: &Value) -> Value {
    match content {
        Value::String(_) => content.clone(),
        Value::Array(parts) => {
            let mut texts = String::new();
            let mut out: Vec<Value> = Vec::new();
            let mut has_media = false;
            for p in parts {
                match p["type"].as_str().unwrap_or("") {
                    "input_text" | "output_text" | "text" => {
                        let t = p["text"].as_str().unwrap_or_default();
                        texts.push_str(t);
                        out.push(json!({"type": "text", "text": t}));
                    }
                    "input_image" => {
                        has_media = true;
                        let url = p["image_url"].as_str().unwrap_or_default();
                        out.push(json!({"type": "image_url",
                            "image_url": {"url": url}}));
                    }
                    "refusal" => {
                        let t = p["refusal"].as_str().unwrap_or_default();
                        texts.push_str(t);
                        out.push(json!({"type": "text", "text": t}));
                    }
                    _ => {}
                }
            }
            if !has_media {
                return json!(texts);
            }
            json!(out)
        }
        _ => Value::Null,
    }
}

/// Responses `input[]` → chat-completions `messages[]`. `function_call`
/// items fold onto the preceding assistant message's `tool_calls`;
/// `function_call_output` becomes `role:"tool"`; reasoning items drop.
fn input_to_messages(input: &[Value]) -> Vec<Value> {
    let mut msgs: Vec<Value> = Vec::new();
    for item in input {
        let ty = item["type"].as_str().unwrap_or("");
        match ty {
            "function_call" => {
                let call = json!({
                    "id": item["call_id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item["arguments"],
                    },
                });
                let attach = msgs
                    .last_mut()
                    .filter(|m| m["role"] == "assistant");
                match attach {
                    Some(m) => {
                        if m.get("tool_calls").is_none() {
                            m["tool_calls"] = json!([]);
                            if m.get("content").is_none() {
                                m["content"] = Value::Null;
                            }
                        }
                        m["tool_calls"].as_array_mut().unwrap().push(call);
                    }
                    None => msgs.push(json!({
                        "role": "assistant", "content": Value::Null,
                        "tool_calls": [call],
                    })),
                }
            }
            "function_call_output" => {
                let out = match &item["output"] {
                    Value::String(s) => json!(s),
                    other => json!(other.to_string()),
                };
                msgs.push(json!({
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": out,
                }));
            }
            // message items and bare {role,content} shapes
            _ if item.get("role").and_then(Value::as_str).is_some() => {
                msgs.push(json!({
                    "role": item["role"],
                    "content": content_to_chat(&item["content"]),
                }));
            }
            _ => {}
        }
    }
    msgs
}

/// Responses `tools[]` → chat-completions `tools[]`.
fn tools_to_chat(tools: &[Value]) -> Vec<Value> {
    tools
        .iter()
        .filter(|t| t["type"].as_str() == Some("function"))
        .map(|t| {
            json!({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            })
        })
        .collect()
}

fn tool_choice_to_chat(v: &Value) -> Option<Value> {
    match v {
        Value::String(s) if ["auto", "required", "none"].contains(&s.as_str()) => {
            Some(json!(s))
        }
        Value::Object(_) if v["type"].as_str() == Some("function") => Some(json!({
            "type": "function",
            "function": {"name": v["name"]},
        })),
        _ => None,
    }
}

/// Build the upstream `/chat/completions` body.
pub fn to_chat_request(turn: &PreparedTurn, upstream_model: &str, body: &Value) -> Value {
    let mut messages = Vec::new();
    if !turn.system.is_empty() {
        messages.push(json!({"role": "system", "content": turn.system}));
    }
    messages.extend(input_to_messages(&turn.input));

    let mut req = json!({
        "model": upstream_model,
        "messages": messages,
        "stream": false,
    });
    let tools = tools_to_chat(&turn.tools);
    if !tools.is_empty() {
        req["tools"] = json!(tools);
    }
    if let Some(tc) = tool_choice_to_chat(&body["tool_choice"]) {
        req["tool_choice"] = tc;
    }
    for (k, v) in [("temperature", "temperature"), ("top_p", "top_p")] {
        if let Some(x) = body.get(v).filter(|x| !x.is_null()) {
            req[k] = x.clone();
        }
    }
    if let Some(n) = body["max_output_tokens"].as_u64() {
        req["max_tokens"] = json!(n);
    }
    if let Some(e) = &turn.effort {
        req["reasoning_effort"] = json!(e);
    }
    if let Some(p) = body["parallel_tool_calls"].as_bool() {
        req["parallel_tool_calls"] = json!(p);
    }
    req
}

/// A `chat/completions` JSON response → the relay's `TurnResult`.
pub fn to_turn_result(v: &Value) -> TurnResult {
    let choice = &v["choices"][0];
    let msg = &choice["message"];
    let mut items = Vec::new();

    if let Some(r) = msg["reasoning_content"].as_str() {
        if !r.is_empty() {
            items.push(OutItem::Reasoning { text: r.to_string() });
        }
    }
    let text = match &msg["content"] {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter(|p| p["type"].as_str() == Some("text"))
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    if !text.is_empty() {
        items.push(OutItem::Message { text });
    }
    if let Some(calls) = msg["tool_calls"].as_array() {
        for tc in calls {
            items.push(OutItem::FunctionCall {
                call_id: tc["id"].as_str().unwrap_or_default().to_string(),
                name: tc["function"]["name"].as_str().unwrap_or_default().to_string(),
                arguments: tc["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
            });
        }
    }

    let u = &v["usage"];
    let usage = if u.is_object() {
        Some(Usage {
            input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
            output_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
            total_tokens: u["total_tokens"].as_u64().unwrap_or(0),
            extra: json!({
                "prompt_cache_hit_tokens": u["prompt_cache_hit_tokens"],
                "prompt_cache_miss_tokens": u["prompt_cache_miss_tokens"],
            }),
        })
    } else {
        None
    };

    TurnResult {
        items,
        usage,
        stop_reason: choice["finish_reason"].as_str().unwrap_or("stop").to_string(),
        suspended: false,
    }
}

/// Blocking POST to `{base}/chat/completions`.
pub fn post_chat(cfg: &Compat, body: &Value) -> Result<Value, String> {
    let url = format!("{}/chat/completions", cfg.upstream_base);
    let mut req = ureq::post(&url)
        .timeout(cfg.timeout)
        .set("Content-Type", "application/json");
    if let Some(k) = &cfg.api_key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }
    for (k, v) in &cfg.headers {
        req = req.set(k, v);
    }
    match req.send_json(body) {
        Ok(r) => r.into_json().map_err(|e| format!("upstream JSON decode: {e}")),
        Err(ureq::Error::Status(code, resp)) => {
            let mut text = resp.into_string().unwrap_or_default();
            if text.len() > 600 {
                let mut end = 600;
                while !text.is_char_boundary(end) {
                    end -= 1;
                }
                text.truncate(end);
                text.push('…');
            }
            Err(format!("upstream HTTP {code}: {text}"))
        }
        Err(e) => Err(format!("upstream: {e}")),
    }
}

/// GET `{base}/models` → `[{"id","name"}]` rows for `provider/models`.
pub fn probe_models(cfg: &Compat) -> Result<Vec<Value>, String> {
    let url = format!("{}/models", cfg.upstream_base);
    let mut req = ureq::get(&url).timeout(Duration::from_secs(15));
    if let Some(k) = &cfg.api_key {
        req = req.set("Authorization", &format!("Bearer {k}"));
    }
    for (k, v) in &cfg.headers {
        req = req.set(k, v);
    }
    let resp = req
        .call()
        .map_err(|e| format!("models probe: {e}"))?
        .into_json::<Value>()
        .map_err(|e| format!("models JSON: {e}"))?;
    let rows = resp["data"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|m| {
            let id = m["id"].as_str()?;
            Some(json!({"id": id, "name": m["name"].as_str().unwrap_or(id)}))
        })
        .collect();
    Ok(rows)
}

/// The ready-made `TurnFn` for `relay::mint_turn`: Responses request in,
/// Responses SSE bytes out, upstream chat-completions in the middle.
///
/// `model_map` optionally rewrites the wire model id to the upstream id
/// (e.g. alias rows in a static catalog).
pub fn compat_turn(
    cfg: Arc<Compat>,
    model_map: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
) -> TurnFn {
    Arc::new(move |body: &[u8], cancel: &AtomicBool| {
        if cancel.load(Ordering::SeqCst) {
            return Err("turn cancelled".into());
        }
        let body: Value =
            serde_json::from_slice(body).map_err(|e| format!("request JSON: {e}"))?;
        let wire_model = body["model"].as_str().unwrap_or_default().to_string();
        let prepared = chat::prepare_turn(&body, &wire_model)?;
        let upstream_model = model_map
            .as_ref()
            .map(|f| f(&prepared.model))
            .unwrap_or_else(|| prepared.model.clone());
        let req = to_chat_request(&prepared, &upstream_model, &body);
        if cancel.load(Ordering::SeqCst) {
            return Err("turn cancelled".into());
        }
        let resp = post_chat(&cfg, &req)?;
        let result = to_turn_result(&resp);
        let response_id = format!("resp_{}", chat::rand_hex(12));
        Ok(chat::fold_result(&result, &upstream_model, &response_id))
    })
}

#[cfg(test)]
mod tests;
