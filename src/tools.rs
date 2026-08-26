use std::{collections::HashMap, vec};

use async_openai::types::chat::ChatCompletionTools;
use calculator::definition::calculator_tool_definition;
use web_search::definition::web_search_tool_definition;

use crate::tools::{
    calculator::calculator_impl::CalculatorTool, tool::Tool,
    web_search::web_search_impl::WebSearchTool,
};

pub mod calculator;
pub mod tool;
pub mod web_search;

pub type ToolBox = HashMap<String, Box<dyn Tool>>;

pub fn build_tool_box() -> ToolBox {
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(CalculatorTool), Box::new(WebSearchTool)];
    tools
        .into_iter()
        .map(|t| (t.name().to_string(), t))
        .collect()
}
