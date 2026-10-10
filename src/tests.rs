use super::*;

#[test]
fn messages_and_tools_translate() {
    let body = json!({
        "model": "deepseek-chat[effort=high]",
        "instructions": "be terse",
        "input": [
            {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"yo"}]},
            {"type":"function_call","call_id":"c1","name":"bash","arguments":"{\"cmd\":\"ls\"}"},
            {"type":"function_call_output","call_id":"c1","output":"file.txt"},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"and?"}]}
        ],
        "tools": [{"type":"function","name":"bash","description":"run","parameters":{"type":"object"}}],
        "max_output_tokens": 512,
        "temperature": 0.2
    });
    let prepared = chat::prepare_turn(&body, "deepseek-chat[effort=high]").unwrap();
    assert_eq!(prepared.model, "deepseek-chat");
    assert_eq!(prepared.effort.as_deref(), Some("high"));
    let req = to_chat_request(&prepared, "deepseek-chat", &body);
    assert_eq!(req["model"], "deepseek-chat");
    assert_eq!(req["reasoning_effort"], "high");
    assert_eq!(req["max_tokens"], 512);
    let msgs = req["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs[0]["content"], "be terse");
    assert_eq!(msgs[1]["content"], "hi");
    assert_eq!(msgs[2]["role"], "assistant");
    // function_call folds onto the preceding assistant message
    assert_eq!(msgs[2]["tool_calls"][0]["id"], "c1");
    assert_eq!(msgs[3]["role"], "tool");
    assert_eq!(msgs[3]["tool_call_id"], "c1");
    assert_eq!(msgs[4]["content"], "and?");
    assert_eq!(req["tools"][0]["function"]["name"], "bash");
}

#[test]
fn images_stay_parts() {
    let v = content_to_chat(&json!([
        {"type":"input_text","text":"look"},
        {"type":"input_image","image_url":"data:image/png;base64,AA=="}
    ]));
    assert_eq!(v[1]["type"], "image_url");
}

#[test]
fn completion_folds_tool_calls() {
    let r = to_turn_result(&json!({
        "choices": [{"finish_reason": "tool_calls", "message": {
            "content": null,
            "tool_calls": [{"id":"call_9","type":"function",
                "function":{"name":"grep","arguments":"{\"p\":\"x\"}"}}]
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                  "prompt_cache_hit_tokens": 8}
    }));
    assert!(matches!(r.items[0], OutItem::FunctionCall { .. }));
    assert_eq!(r.usage.unwrap().input_tokens, 10);
    assert_eq!(r.stop_reason, "tool_calls");
}

#[test]
fn credential_key_walks_envelope() {
    let p = json!({"credential": {"credential": {"secrets": {"api_key": "sk-x"}}}});
    assert_eq!(credential_key(&p).as_deref(), Some("sk-x"));
    let p2 = json!({"credential": {"credential": {"secrets": {"weird": "tok"}}}});
    assert_eq!(credential_key(&p2).as_deref(), Some("tok"));
}
