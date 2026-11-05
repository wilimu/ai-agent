use async_openai::types::chat::{ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc, vec};

use crate::agent::ExecutionContext;

use crate::tools::file::delete::DeleteFileTool;
use crate::tools::file::list::ListFileTool;
use crate::tools::file::read::ReadFileTool;
use crate::tools::file::unzip::UnzipFileTool;
use crate::tools::{read_image::r#impl::ReadImageTool,
    calculator::calculator_impl::CalculatorTool,
    mcp::{client::McpClient, tool::McpTool},
    web_search::web_search_impl::WebSearchTool,
};

pub mod calculator;
pub mod file;
pub mod mcp;
pub mod web_search;
pub mod read_image;

pub type ToolBox = HashMap<String, Box<dyn Tool>>;

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters(&self) -> Value;
    //fn requires_confirmation(&self) -> bool;
    //fn get_confirmation_message(&self, arguments: &serde_json::Value) -> String;
    async fn execute(&self, args_json: &str, context: &ExecutionContext) -> anyhow::Result<String>;
    fn definition(&self) -> anyhow::Result<ChatCompletionTools> {
        let function = FunctionObjectArgs::default()
            .name(self.name())
            .description(self.description())
            .parameters(self.parameters())
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build definition for {}:{e}", self.name()))?;
        Ok(ChatCompletionTools::Function(ChatCompletionTool {
            function,
        }))
    }
}

pub async fn build_toolbox() -> anyhow::Result<ToolBox> {
    let mut tools: Vec<Box<dyn Tool>> = vec![Box::new(CalculatorTool), Box::new(WebSearchTool)];

    let mcp_client = Arc::new(McpClient::connect().await?);
    for tool in mcp_client.list_tools().await? {
        tools.push(Box::new(McpTool::new(mcp_client.clone(), tool)));
    }
    Ok(tools
        .into_iter()
        .map(|t| (t.name().to_string(), t))
        .collect())
}

pub fn build_file_explorer_toolbox(vision_model: impl Into<String>) -> ToolBox {
    let tools: Vec<Box<dyn Tool>> = vec![
        Box::new(UnzipFileTool),
        Box::new(ListFileTool),
        Box::new(ReadFileTool),
        Box::new(ReadImageTool::new(vision_model)),
        Box::new(DeleteFileTool)
    ];
    into_toolbox(tools)
}

fn into_toolbox(tools: Vec<Box<dyn Tool>>) -> ToolBox {
    tools.into_iter().map(|t| (t.name().to_string(), t)).collect()
}