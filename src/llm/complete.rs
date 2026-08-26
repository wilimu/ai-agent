use std::env::Args;

use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionTools,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
};

use crate::tools::{
    ToolBox,
    calculator::execute::{CalculatorArgs, calculate},
    tool,
    web_search::execute::{WebSearchArgs, search_web},
};

pub async fn chat_complete(
    model: &str,
    system: Option<&str>,
    prompt: &str,
    tool_box: ToolBox,
) -> anyhow::Result<String> {
    let client = async_openai::Client::new();

    let mut messages = vec![];

    if let Some(system) = system {
        messages.push(
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()?
                .into(),
        );
    }
    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(prompt)
            .build()?
            .into(),
    );

    let tool_definitions: Vec<ChatCompletionTools> = tool_box
        .values()
        .filter_map(|t| match t.definition() {
            Ok(def) => Some(def),
            Err(err) => None,
        })
        .collect();

    loop {
        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages(messages.clone())
            .tools(tool_definitions.clone())
            .max_tokens(2048u32)
            .build()?;

        let response = client.chat().create(request).await?;

        tracing::info!("response: {:#?}", response);

        let message = response
            .clone()
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no choicees"))?
            .message;

        tracing::info!("message: {:?}", message);
        if let Some(tool_calls) = message.tool_calls {
            messages.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()?
                    .into(),
            );
            for tool_call in tool_calls {
                match tool_call {
                    ChatCompletionMessageToolCalls::Function(call) => {
                        let function_name = call.function.name;
                        let arguments = call.function.arguments;
                        tracing::info!("Tool call:{function_name}({arguments})");

                        let function_result = match tool_box.get(&function_name) {
                            Some(tool) => match tool.execute(&arguments).await {
                                Ok(r) => r,
                                Err(err) => {
                                    let msg = format!("Tool execution error:{err}");
                                    tracing::error!(msg);
                                    msg
                                }
                            },
                            None => {
                                let msg = format!("Tool {function_name} not found");
                                tracing::error!(msg);
                                msg
                            }
                        };
                        messages.push(
                            ChatCompletionRequestToolMessageArgs::default()
                                .tool_call_id(call.id.clone())
                                .content(function_result.clone())
                                .build()?
                                .into(),
                        );
                        function_result
                    }
                    ChatCompletionMessageToolCalls::Custom(call) => {
                        let msg = "Custom tool not suppurted.";
                        messages.push(
                            ChatCompletionRequestToolMessageArgs::default()
                                .tool_call_id(call.id.clone())
                                .content(msg)
                                .build()?
                                .into(),
                        );
                        tracing::error!("{msg}");
                        msg.to_string()
                    }
                };
            }
        } else {
            let content = message
                .content
                .ok_or_else(|| anyhow::anyhow!("no content on final reponse."))?;
            return Ok(content);
        }
    }
}
