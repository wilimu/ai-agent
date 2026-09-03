use async_openai::types::chat::{
    ChatCompletionTool, ChatCompletionTools, FunctionObjectArgs,
};
use serde_json::json;

pub fn calculator_tool_definition() -> ChatCompletionTools {
    ChatCompletionTools::Function(ChatCompletionTool {
        function: FunctionObjectArgs::default()
            .name("calculator")
            .parameters(json!({
                            "type":"object",
                            "properties":{
                                "operator":{
                                "type":"string",
                                "description":"arithmetic operation to perform.",
                                "enum":["add","subtract","multiply","divide"],
                            },
                            "first_number":{
                                "type":"number",
                                "description":"first number for the calculation."
                            },
            "second_number":{
                "type":"number",
                "description":"second number for the calculation."
            }
                        },
                        "required":["operator","first_number","second_number"]
                    }))
            .build()
            .expect("failed to build calculator definition."),
    })
}
