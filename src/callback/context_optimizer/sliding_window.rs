use std::collections::HashSet;

use crate::{
    agent::{
        ContentItem, ExecutionContext,
        llm_request::{BeforeLlmCallback, LlmRequest},
    },
    callback::context_optimizer::count_tokens,
};

pub struct SlidingWindow {
    pub model: String,
    pub token_threshold: usize,
    pub window_size: usize,
}

#[async_trait::async_trait]
impl BeforeLlmCallback for SlidingWindow {
    async fn call(&self, _context: &mut ExecutionContext, request: &mut LlmRequest) {
        let before = count_tokens(&self.model, request);
        if before < self.token_threshold {
            return;
        }

        if request.contents.len() < self.window_size {
            return;
        }

        let user_idx = request
            .contents
            .iter()
            .position(|item| matches!(item,ContentItem::Message { role, .. } if role =="user"));

        let Some(user_idx) = user_idx else {
            return;
        };

        tracing::info!(
            "SlidingWindow: {before} tokens >= threshold {}, trimming to last {} items",
            self.token_threshold,
            self.window_size
        );

        let start = find_safe_start(&request.contents, request.contents.len() - self.window_size);
        let start = start.max(user_idx + 1);
        let remaining = request.contents[start..].to_vec();

        request.contents.drain(user_idx + 1..start);

        let after = count_tokens(&self.model, request);

        tracing::info!(
            "SlidingWindow: after trim, {after} tokens ({} items kept)",
            request.contents.len()
        );
    }
}
pub fn find_safe_start(contents: &[ContentItem], mut start: usize) -> usize {
    loop {
        let call_ids: HashSet<&str> = contents[start..]
            .iter()
            .filter_map(|item| match item {
                ContentItem::ToolCall {
                    tool_call_id,
                    name,
                    arguments,
                } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect();

        let missing_call = contents[start..].iter().find_map(|item| match item {
            ContentItem::ToolResult {
                tool_call_id,
                name,
                status,
                content,
            } if !call_ids.contains(tool_call_id.as_str()) => Some(tool_call_id),
            _ => None,
        });

        let Some(missing_call) = missing_call else {
            break;
        };

        let Some(call_idx) =contents[..start].iter().position(|item|{matches!(item,ContentItem::ToolCall { tool_call_id,.. } if tool_call_id==missing_call)})else {
            break;
        } ;
        start = call_idx;
    }
    start
}
