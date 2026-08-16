use async_openai::types::chat::{
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, ResponseFormat, ResponseFormatJsonSchema,
};

use crate::models::action_plan::{self, ActionPlan};

pub async fn chat_complete_structured(
    model: &str,
    system: Option<&str>,
    prompt: &str,
) -> anyhow::Result<ActionPlan> {
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

    let schema = schemars::schema_for!(ActionPlan);
    let schema_json = schema.as_value().clone();
    let format_setting = ResponseFormat::JsonSchema {
        json_schema: ResponseFormatJsonSchema {
            description: Some(
                "a step by step agent action plan with difficulty and time estimate".into(),
            ),
            name: "action plan".into(),
            schema: schema_json,
            strict: Some(true),
        },
    };

    let request = CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages(messages)
        .response_format(format_setting)
        .max_tokens(2048u32)
        .build()?;

    let response = client.chat().create(request).await?;

    tracing::info!("response: {:#?}", response);

    let plan: ActionPlan = response
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .ok_or_else(|| anyhow::anyhow!("no content"))
        .and_then(|s| serde_json::from_str(&s).map_err(Into::into))?;
    Ok(plan)
}
