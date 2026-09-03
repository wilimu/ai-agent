use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentItem {
    #[serde(rename = "message")]
    Message { role: String, content: String },
    #[serde(rename = "tool_call")]
    ToolCall {
        tool_call_id: String,
        name: String,
        arguments: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_call_id: String,
        name: String,
        status: ToolResultStatus,
        content: String,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolResultStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    id: String,
    execution_id: String,
    time_stamp: i64,
    author: String,
    pub contents: Vec<ContentItem>,
}
impl Event {
    pub fn new(
        execution_id: impl Into<String>,
        author: impl Into<String>,
        contents: Vec<ContentItem>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            execution_id: execution_id.into(),
            time_stamp: Utc::now().timestamp(),
            author: author.into(),
            contents,
        }
    }
}
