use std::{collections::HashSet, io::Write};

use crate::agent::{
    ExecutionContext,
    callback::{BeforeToolCallBack, ToolCallView},
};

pub struct ApprovalCallback {
    dangerous_tools: HashSet<String>,
}

impl ApprovalCallback {
    pub fn new(dangerous_tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            dangerous_tools: dangerous_tools.into_iter().map(Into::into).collect(),
        }
    }
}

#[async_trait::async_trait]
impl BeforeToolCallBack for ApprovalCallback {
    async fn call(
        &self,
        _context: &ExecutionContext,
        tool_call: ToolCallView<'_>,
    ) -> Option<String> {
        if !self.dangerous_tools.contains(tool_call.name) {
            return None;
        }

        println!("\n Dangerous tool execution requested");
        println!("Tool: {}", tool_call.name);
        println!("Arguments: {}", tool_call.arguments);

        let approved = tokio::task::spawn_blocking(|| {
            print!("Execute? (Y/n): ");
            std::io::stdout().flush().ok();
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).ok();
            input.trim().eq_ignore_ascii_case("y")
        })
        .await
        .unwrap_or(false);

        if approved {
            println!("approved, continue to run.");
            None
        } else {
            println!("rejected, skip running.");
            Some(format!("User denied execution of {}", tool_call.name))
        }
    }
}
