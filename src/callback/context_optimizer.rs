use crate::{
    agent::{
        ContentItem::{Message, ToolCall, ToolResult},
        ExecutionContext,
        llm_request::{BeforeLlmCallback, LlmRequest},
    },
    callback::context_optimizer::{compaction::Compaction, summarization::Summarization},
};

pub mod compaction;
pub mod sliding_window;
pub mod summarization;

pub struct ContextOptimizer {
    pub model: String,
    pub token_threshold: usize,
    pub enable_compaction: bool,
    pub compaction_keep_recent: usize,
    pub enable_summarization: bool,
    pub summarization_keep_recent: usize,
}

impl ContextOptimizer {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            token_threshold: 20_000,
            enable_compaction: true,
            compaction_keep_recent: 4,
            enable_summarization: true,
            summarization_keep_recent: 5,
        }
    }
}

#[async_trait::async_trait]
impl BeforeLlmCallback for ContextOptimizer {
    async fn call(&self, context: &mut ExecutionContext, request: &mut LlmRequest) {
        let before: usize = count_tokens(&self.model, request);
        if before < self.token_threshold {
            return;
        }

        tracing::info!(
            "ContextOptimizer: {before} tokens >= threshold {}, optimizing ",
            self.token_threshold
        );

        if self.enable_compaction {
            Compaction {
                keep_recent: self.compaction_keep_recent,
            }
            .apply(request);
        }

        let after = count_tokens(&self.model, request);

        tracing::info!("ContextOptimizer: compaction brought it to {after} tokens.");

        if after < self.token_threshold {
            return;
        }

        if self.enable_summarization {
            let summarizer = Summarization {
                model: self.model.clone(),
                keep_recent: self.summarization_keep_recent,
            };

            match summarizer.apply(context, request).await {
                Ok(_) => {
                    let after = count_tokens(&self.model, request);
                    tracing::info!("ContextOptimizer: compaction brought it to {after} tokens.")
                }
                Err(e) => {
                    tracing::warn!("Summarization skipped: {e}.")
                }
            }
        }
    }
}
pub fn count_tokens(_model: &str, request: &LlmRequest) -> usize {
    let bpe = tiktoken_rs::cl100k_base()
        .expect("tiktoken should always resolve to the cl100k_base encoding.");
    let mut total = 0usize;

    for instruction in &request.instructions {
        total += bpe.encode_ordinary(instruction).len();
    }

    for item in &request.contents {
        total += 4;
        total += match item {
            Message { content, .. } => bpe.encode_ordinary(content).len(),
            ToolCall {
                name, arguments, ..
            } => {
                bpe.encode_ordinary(name).len() + bpe.encode_ordinary(&arguments.to_string()).len()
            }

            ToolResult { content, .. } => bpe.encode_ordinary(content).len(),
        }
    }
    total
}
