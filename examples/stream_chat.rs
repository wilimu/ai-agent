use std::{any, print, result};

use ai_agent::{
    constant,
    llm::{
        semaphore::get_semaphore,
        stream::{self, chat_stream_with_retry},
    },
};
use futures::StreamExt;
use tokio::task::JoinSet;
use tracing::{Instrument, Level, subscriber};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv()?;

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    let prompts = vec![
        "explain ownership of rust briefly",
        "what is asynconized programming, how does it differ to multithread",
    ];

    let mut set = JoinSet::new();
    for prompt in prompts {
        let span = tracing::info_span!("Chat", prompt = prompt);
        set.spawn(
            async move {
                let permit = get_semaphore().acquire().await?;
                let output = chat_stream_with_retry(
                    constant::GPT_OSS_20B,
                    Some("you are a versatile asistant"),
                    prompt,
                )
                .await?;
                drop(permit);
                Ok::<_, anyhow::Error>((prompt, output))
            }
            .instrument(span),
        );
    }

    while let Some(result) = set.join_next().await {
        match result {
            Ok(Ok((prompt, result))) => tracing::info!("\n{prompt}\n{result}"),
            Ok(Err(err)) => tracing::error!("task paniked: {err}"),
            Err(err) => tracing::error!("task paniked: {err}"),
        }
    }
    Ok(())
}
