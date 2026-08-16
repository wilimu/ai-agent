use std::{any, print};

use ai_agent::{
    constant,
    llm::{complete, structured},
};
use anyhow::Ok;
use tracing::{Level, subscriber};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv()?;

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    let plan = structured::chat_complete_structured(
        constant::DEEPSEEK_V4_FLASH,
        Some("you are a versatile asistant"),
        "I'm going to watch World Cup of America Canada Mexio, give me a plan.",
    )
    .await?;
    print!("Response: {plan:#?}.");
    Ok(())
}
