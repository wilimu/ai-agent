use std::sync::Arc;

use anyhow::anyhow;
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool,
    ChatCompletionToolChoiceOption::{self},
    ChatCompletionTools, CreateChatCompletionRequestArgs, FunctionCall, FunctionObjectArgs,
    ToolChoiceOptions,
};
use backon::{ExponentialBuilder, Retryable};
use serde_json::Value;

use crate::{
    agent::{
        ContentItem::{self, Message, ToolCall, ToolResult},
        Event, ExecutionContext, ToolResultStatus,
        callback::{AfterToolCallback, BeforeToolCallBack, ToolCallView},
        llm_request::{self, BeforeLlmCallback, LlmRequest},
    },
    tools::ToolBox,
};

#[derive(Debug)]
pub struct AgentResult {
    pub output: String,
    pub context: ExecutionContext,
}

pub struct StructuredAgentResult<T> {
    pub output: T,
    pub context: ExecutionContext,
}
pub struct Agent {
    model: String,
    instruction: Option<String>,
    toolbox: Arc<ToolBox>,
    max_steps: u32,
    before_tool_callbacks: Vec<Arc<dyn BeforeToolCallBack>>,
    after_tool_callbacks: Vec<Arc<dyn AfterToolCallback>>,
    before_llm_callback: Vec<Arc<dyn BeforeLlmCallback>>,
}

impl Agent {
    pub fn new(
        model: impl Into<String>,
        instruction: Option<impl Into<String>>,
        toolbox: Arc<ToolBox>,
    ) -> Self {
        Agent {
            model: model.into(),
            instruction: instruction.map(Into::into),
            toolbox,
            max_steps: 10,
            before_tool_callbacks: Vec::new(),
            after_tool_callbacks: Vec::new(),
            before_llm_callback: Vec::new(),
        }
    }

    pub fn with_max_steps(mut self, steps: u32) -> Self {
        self.max_steps = steps;
        self
    }

    pub fn with_before_tool_callback(mut self, callback: Arc<dyn BeforeToolCallBack>) -> Self {
        self.before_tool_callbacks.push(callback);
        self
    }

    pub fn with_after_tool_callback(mut self, callback: Arc<dyn AfterToolCallback>) -> Self {
        self.after_tool_callbacks.push(callback);
        self
    }

    pub async fn run(&self, user_input: &str) -> anyhow::Result<AgentResult> {
        let mut context = ExecutionContext::default();
        let event = Event::new(
            context.execution_id.clone(),
            "user",
            vec![Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        );
        context.add_event(event);

        let client = async_openai::Client::new();

        let tool_definitions: Vec<ChatCompletionTools> = self
            .toolbox
            .values()
            .filter_map(|t| match t.definition() {
                Ok(def) => Some(def),
                Err(e) => {
                    tracing::warn!(
                        "skip tool {}, failed to get its definition: {:#?}",
                        t.name(),
                        e
                    );
                    None
                }
            })
            .collect();

        loop {
            if context.current_step > self.max_steps {
                anyhow::bail!(
                    "agent exceed max the maximum of {} steps without final answer.",
                    self.max_steps
                )
            }
            let llm_request = self.prepare_llm_request(&mut context).await;
            let messages = self.build_messages(&llm_request)?;

            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .max_tokens(2048u32)
                .build()?;

            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;
            if let Some(usage) = &response.usage {
                context.usage.add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                );
            }

            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no choice in response."))?
                .message;

            if let Some(tool_calls) = message.tool_calls {
                self.record_tool_calls(&mut context, &tool_calls);
                self.execute_tool_calls(&mut context, &tool_calls).await;
            } else {
                let content = message
                    .content
                    .ok_or_else(|| anyhow!("no content in final response"))?;

                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "agent",
                    vec![ContentItem::Message {
                        role: "assistant".to_string(),
                        content: content.clone(),
                    }],
                ));

                context.final_result = Some(content.clone());

                return Ok(AgentResult {
                    output: content,
                    context,
                });
            }
            context.increment_step();
        }
    }

    pub async fn run_structured<T>(
        &self,
        user_input: &str,
    ) -> anyhow::Result<StructuredAgentResult<T>>
    where
        T: schemars::JsonSchema + serde::de::DeserializeOwned,
    {
        let mut context = ExecutionContext::default();
        let event = Event::new(
            context.execution_id.clone(),
            "user",
            vec![Message {
                role: "user".to_string(),
                content: user_input.to_string(),
            }],
        );
        context.add_event(event);

        let client = async_openai::Client::new();

        let mut tool_definitions: Vec<ChatCompletionTools> = self
            .toolbox
            .values()
            .filter_map(|t| match t.definition() {
                Ok(def) => Some(def),
                Err(e) => {
                    tracing::warn!(
                        "skip tool {}, failed to get its definition: {:#?}",
                        t.name(),
                        e
                    );
                    None
                }
            })
            .collect();
        tool_definitions.push(final_answer_tool_definition::<T>()?);

        loop {
            if context.current_step > self.max_steps {
                anyhow::bail!(
                    "agent exceed max the maximum of {} steps without final answer.",
                    self.max_steps
                )
            }

            let llm_request = self.prepare_llm_request(&mut context).await;
            let messages = self.build_messages(&llm_request)?;

            let request = CreateChatCompletionRequestArgs::default()
                .model(self.model.clone())
                .messages(messages)
                .tools(tool_definitions.clone())
                .tool_choice(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::Required,
                ))
                .max_tokens(2048u32)
                .build()?;

            let response = (|| async { client.chat().create(request.clone()).await })
                .retry(ExponentialBuilder::default().with_max_times(3))
                .await?;
            if let Some(usage) = &response.usage {
                context.usage.add(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                );
            }

            let message = response
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("no choice in response."))?
                .message;

            let tool_calls = message.tool_calls.ok_or_else(|| {
                anyhow::anyhow!("nodel return no tool calls despite tool_choice=Required.")
            })?;

            self.record_tool_calls(&mut context, &tool_calls);

            let final_call = tool_calls.iter().find_map(|tool_call| match tool_call {
                ChatCompletionMessageToolCalls::Function(f)
                    if f.function.name == "final_answer" =>
                {
                    Some(f)
                }

                _ => None,
            });
            if let Some(final_call) = final_call {
                let raw_arguments = final_call.function.arguments.clone();
                let parsed = serde_json::from_str(&raw_arguments)?;

                context.add_event(Event::new(
                    context.execution_id.clone(),
                    "tool",
                    vec![ContentItem::ToolResult {
                        tool_call_id: final_call.id.clone(),
                        name: "final_answer".to_string(),
                        status: ToolResultStatus::Success,
                        content: raw_arguments.clone(),
                    }],
                ));

                context.final_result = Some(raw_arguments);

                return Ok(StructuredAgentResult {
                    output: parsed,
                    context,
                });
            }
            self.execute_tool_calls(&mut context, &tool_calls).await;
            context.increment_step();
        }
    }

    fn build_messages(
        &self,
        llm_request: &LlmRequest,
    ) -> anyhow::Result<Vec<ChatCompletionRequestMessage>> {
        let mut messages = Vec::new();
        if let Some(system_prompt) = &self.instruction {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(system_prompt.clone())
                    .build()?
                    .into(),
            );
        };

        for extra_intruction in &llm_request.instructions {
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(extra_intruction.as_str())
                    .build()?
                    .into(),
            );
        }

        for item in &llm_request.contents {
            match item {
                Message { role, content } => {
                    let message: ChatCompletionRequestMessage = if role == "user" {
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(content.clone())
                            .build()?
                            .into()
                    } else {
                        ChatCompletionRequestAssistantMessageArgs::default()
                            .content(content.clone())
                            .build()?
                            .into()
                    };
                    messages.push(message);
                }
                ToolCall {
                    tool_call_id,
                    name,
                    arguments,
                } => {
                    let tool_call =
                        ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                            id: tool_call_id.clone(),
                            function: FunctionCall {
                                name: name.clone(),
                                arguments: arguments.to_string(),
                            },
                        });

                    if let Some(ChatCompletionRequestMessage::Assistant(last)) = messages.last_mut()
                    {
                        last.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
                    } else {
                        messages.push(
                            ChatCompletionRequestAssistantMessageArgs::default()
                                .tool_calls(vec![tool_call])
                                .build()?
                                .into(),
                        );
                    }
                }
                ToolResult {
                    tool_call_id,
                    content,
                    ..
                } => messages.push(
                    ChatCompletionRequestToolMessageArgs::default()
                        .tool_call_id(tool_call_id.clone())
                        .content(content.clone())
                        .build()?
                        .into(),
                ),
            }
        }

        Ok(messages)
    }

    async fn prepare_llm_request(&self, context: &mut ExecutionContext) -> LlmRequest {
        let mut request = LlmRequest {
            instructions: Vec::new(),
            contents: context
                .events
                .iter()
                .flat_map(|ev| ev.contents.iter().cloned())
                .collect(),
        };

        for cb in &self.before_llm_callback {
            cb.call(context, &mut request).await
        }
        request
    }
    fn record_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        let mut call_items = Vec::new();
        for tool_call in tool_calls {
            if let ChatCompletionMessageToolCalls::Function(function_call) = tool_call {
                let arguments = serde_json::from_str(&function_call.function.arguments)
                    .unwrap_or(serde_json::Value::Null);
                call_items.push(ContentItem::ToolCall {
                    tool_call_id: function_call.id.clone(),
                    name: function_call.function.name.clone(),
                    arguments,
                });
            }
        }
        context.add_event(Event::new(
            context.execution_id.clone(),
            "agent",
            call_items,
        ));
    }

    async fn execute_tool_calls(
        &self,
        context: &mut ExecutionContext,
        tool_calls: &[ChatCompletionMessageToolCalls],
    ) {
        for tool_call in tool_calls {
            let mut result_items = Vec::new();
            let ChatCompletionMessageToolCalls::Function(function_call) = tool_call else {
                continue;
            };

            let function_name = &function_call.function.name;
            let arguments = &function_call.function.arguments;

            tracing::info!("tool call:{function_name}({arguments})");

            let arguments_value: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
            let view = ToolCallView {
                tool_call_id: &function_call.id,
                name: function_name,
                arguments: &arguments_value,
            };

            let mut short_circuited = None;

            for callback in &self.before_tool_callbacks {
                if let Some(result) = callback.call(context, view).await {
                    short_circuited = Some(result);
                    break;
                }
            }

            let (mut status, mut content) = match short_circuited {
                Some(result) => (ToolResultStatus::Success, result),
                None => match self.toolbox.get(function_name) {
                    Some(tool) => match tool.execute(arguments, context).await {
                        Ok(result) => {
                            tracing::info!("Tool result:{result}");
                            (ToolResultStatus::Success, result)
                        }
                        Err(e) => {
                            let msg = format!("tool execution error: {e}");
                            tracing::error!({ msg });
                            (ToolResultStatus::Error, msg)
                        }
                    },
                    None => {
                        let msg = format!("Tool execution error: unknown tool {function_name}");
                        tracing::error!("{msg}");
                        (ToolResultStatus::Error, msg)
                    }
                },
            };

            for callback in &self.after_tool_callbacks {
                if let Some((new_status, new_content)) = callback
                    .call(context, &function_call.id, function_name, status, &content)
                    .await
                {
                    status = new_status;
                    content = new_content;
                    break;
                }
            }
            result_items.push(ContentItem::ToolResult {
                tool_call_id: function_call.id.clone(),
                name: function_call.function.name.clone(),
                status,
                content,
            });
            context.add_event(Event::new(
                context.execution_id.clone(),
                "tool",
                result_items,
            ));
        }

        // let tool_calls = message.tool_calls.ok_or_else(|| {
        //     anyhow!("Model return no tool calls despite tool choice = Required.")
        // })?;

        // self.record_tool_calls(&mut context, &tool_calls);

        // let final_call = tool_calls.iter().find_map(|too_call| match too_call {
        //     ChatCompletionMessageToolCalls::Function(functoin_call)
        //         if functoin_call.function.name == "final answer" =>
        //     {
        //         Some(functoin_call)
        //     }
        //     _ => None,
        // });
        // if let Some(fc) = final_call {
        //     let raw_arguments = fc.function.arguments.clone();
        //     let parsed = serde_json::from_str(&raw_arguments);
    }

    pub fn with_before_llm_callback(mut self, callback: Arc<dyn BeforeLlmCallback>) -> Self {
        self.before_llm_callback.push(callback);
        self
    }
}

fn final_answer_tool_definition<T: schemars::JsonSchema>() -> anyhow::Result<ChatCompletionTools> {
    let schema = schemars::schema_for!(T);
    let schema_json = serde_json::to_value(schema)?;

    let function = FunctionObjectArgs::default()
        .name("final_answer")
        .description("return the final stuctured answer matching the require schema.")
        .parameters(schema_json)
        .build()
        .map_err(|e| anyhow!("failed to build final_answer tool definition: {e}."))?;
    Ok(ChatCompletionTools::Function(ChatCompletionTool {
        function,
    }))
}
