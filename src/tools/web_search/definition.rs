use async_openai::types::chat::{
    ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs,
};
use schemars::schema_for;

use crate::tools::web_search::execute::WebSearchArgs;

pub fn web_search_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObjectArgs::default()
            .name("web_search")
            .description("Search the web for current information on a given query.")
            .parameters(
                serde_json::to_value(schema_for!(WebSearchArgs))
                    .expect("Failed to serialize WebSearchArgs schema"),
            )
            .build()
            .expect("failed to build calculator definition."),
    })
}
