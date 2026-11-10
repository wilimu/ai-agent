use ai_agent::{
    agent::Agent,
    constant::{DEEPSEEK_V4_FLASH, GPT_4O_MINI_MODEL},
    tools::build_toolbox,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let toolbox = Arc::new(build_toolbox().await?);

    let agent = Agent::new(
        DEEPSEEK_V4_FLASH,
        Some("你是一个全能的助手".to_string()),
        toolbox,
    );

    let result = agent
        .run("multi_turn", "我的名字叫张三，是一个软件开发工程师。")
        .await?;
    println!("\n回答1: {}", result.output);

    let result = agent
        .run("multi_turn", "我的叫什么名字，我的职业是什么？")
        .await?;
    println!("\n回答2: {}", result.output);

    Ok(())
}
