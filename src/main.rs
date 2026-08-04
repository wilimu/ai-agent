use std::{any, print};

use crate::llm::complete;
use anyhow::Ok;
use tracing::{Level, subscriber};
use tracing_subscriber::FmtSubscriber;

mod constant;
mod llm;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv()?;

    let url = std::env::var("OPENAI_BASE_URL")?;
    print!("{url}");

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    let content = complete::chat_complete(
        constant::GEMMA_4_26B,
        Some("you are a versatile asistant"),
        "which team won champion title of china first soccer league?",
    )
    .await?;
    print!("Response: {content}.");
    Ok(())
}
