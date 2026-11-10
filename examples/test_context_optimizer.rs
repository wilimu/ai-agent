use std::sync::Arc;

use ai_agent::{
    agent::{Agent, llm_request::BeforeLlmCallback},
    callback::context_optimizer::{ContextOptimizer, sliding_window::SlidingWindow},
    constant::DEEPSEEK_V4_FLASH,
    tools::{ToolBox, build_toolbox},
};

use uuid::Uuid;

const SYSTEM_PROMPT: &str = "你是一个善用网页搜索的研究助手。请每次只调用一次 web_search，\
拿到结果、看完之后再决定下一步搜什么，不要一次性并行发起多个搜索。\
至少搜索 5 轮，每轮换一个不同角度或关键词，最后再给出总结。";
const QUERY: &str = "帮我搜索一下 2026 世界人工智能大会 WAIC 的三个亮点，不要扩大搜索范围，选择三个两点，并各自搜索一次细节";

async fn run_demo(
    title: &str,
    toolbox: Arc<ToolBox>,
    callback: Arc<dyn BeforeLlmCallback>,
) -> anyhow::Result<()> {
    println!("\n========== {title} ==========");

    let agent = Agent::new(DEEPSEEK_V4_FLASH, Some(SYSTEM_PROMPT.to_string()), toolbox)
        .with_max_steps(16)
        .with_before_llm_callback(callback);

    let result = agent.run(&Uuid::new_v4().to_string(), QUERY).await?;

    println!("\n最终回答: {}", result.output);
    println!("累计消耗 tokens: {}", result.context.usage.total_tokens);

    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // toolbox 只需要建一次，三个演示共用（Arc clone 很便宜）。
    let toolbox = Arc::new(build_toolbox().await?);

    // 三个演示互相独立，想单独看哪个效果，把其余两段注释掉即可。

    // ── 演示一：滑动窗口 ──────────────────────────────────────
    // 只按"最近 N 条"砍，完全不管每条内容有多大，最容易看出它的局限。
    {
        let sliding = SlidingWindow {
            model: DEEPSEEK_V4_FLASH.to_string(),
            token_threshold: 3_500,
            window_size: 8,
        };
        run_demo(
            "演示一：滑动窗口 SlidingWindow",
            toolbox.clone(),
            Arc::new(sliding),
        )
        .await?;
    }

    // ── 演示二：只用压缩 ──────────────────────────────────────
    // 关掉总结，看压缩单独能不能把上下文压住。
    {
        let compaction_only = ContextOptimizer {
            token_threshold: 3_500,
            compaction_keep_recent: 8,
            enable_summarization: false,
            ..ContextOptimizer::new(DEEPSEEK_V4_FLASH)
        };
        run_demo(
            "演示二：只用压缩 Compaction",
            toolbox.clone(),
            Arc::new(compaction_only),
        )
        .await?;
    }

    // ── 演示三：只用总结 ──────────────────────────────────────
    // 关掉压缩，逼总结自己扛下所有精简工作，最容易看到总结生效。
    {
        let summarization_only = ContextOptimizer {
            token_threshold: 3_500,
            enable_compaction: false,
            summarization_keep_recent: 4,
            ..ContextOptimizer::new(DEEPSEEK_V4_FLASH)
        };
        run_demo(
            "演示三：只用总结 Summarization",
            toolbox.clone(),
            Arc::new(summarization_only),
        )
        .await?;
    }

    Ok(())
}
