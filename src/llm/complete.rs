use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool, ChatCompletionTools,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
};

use crate::tools::{
    calculator::execute::{CalculatorArgs, calculate},
    web_search::execute::{WebSearchArgs, search_web},
};

pub async fn chat_complete(
    model: &str,
    system: Option<&str>,
    prompt: &str,
    tools: Vec<ChatCompletionTools>,
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
    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages.clone())
        .tools(tools.clone())
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
                ChatCompletionMessageToolCalls::Function(function_call) => {
                    let function_name = &function_call.function.name;
                    let arguments = &function_call.function.arguments;

                    tracing::info!("function: {function_name}");
                    tracing::info!("Arguments: {arguments}");

                    if function_name == "calculator" {
                        let args: CalculatorArgs = serde_json::from_str(&arguments)?;
                        let result =
                            calculate(&args.operator, args.first_number, args.second_number);

                        let tool_result = match result {
                            Ok(r) => r.to_string(),
                            Err(e) => e,
                        };

                        tracing::info!("calculator result: {tool_result}");
                        messages.push(
                            ChatCompletionRequestToolMessageArgs::default()
                                .tool_call_id(function_call.id)
                                .content(tool_result)
                                .build()?
                                .into(),
                        )
                    } else if function_name == "web_search" {
                        let args: WebSearchArgs = serde_json::from_str(&arguments)?;
                        let result = search_web(args).await;
                        let tool_result = match result {
                            Ok(r) => serde_json::to_string(&r)?,
                            Err(e) => e.to_string(),
                        };

                        tracing::info!("web search result: {tool_result}");
                        messages.push(
                            ChatCompletionRequestToolMessageArgs::default()
                                .tool_call_id(function_call.id)
                                .content(tool_result)
                                .build()?
                                .into(),
                        )
                    }
                }
                _ => {
                    tracing::error!("unsupportted tool type.")
                }
            }
        }

        let request = CreateChatCompletionRequestArgs::default()
            .model(model)
            .messages(messages.clone())
            .tools(tools.clone())
            .max_tokens(2048_u32)
            .build()?
            .into();

        let response = client.chat().create(request).await?;
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("no content in response."))?;

        return Ok(content);
    }

    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .ok_or_else(|| anyhow::anyhow!("no content"))?;
    Ok(content)
}
