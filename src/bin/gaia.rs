use std::{collections::HashMap, print, sync::Arc};

use ai_agent::{
    constant,
    gaia::{
        dataset::load_gaia_level1,
        evaluator::{evaluate_gaia_single, evaluate_gaia_single_with_tool},
        models::GaiaEvalResult,
    },
    llm::semaphore,
    tools::build_toolbox,
};
use std::io::{self, Write};
use tokio::task::JoinSet;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    io::stdout().flush().unwrap();

    dotenvy::dotenv()?;

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;

    gaia_level1_experiment().await
}

pub async fn gaia_level1_experiment() -> anyhow::Result<()> {
    let problems = load_gaia_level1().await?;
    let toolbox = Arc::new(build_toolbox().await?);

    let mut set = JoinSet::new();
    let model = constant::DEEPSEEK_V4_FLASH;

    for problem in &problems {
        let problem = problem.clone();
        set.spawn(async move {
            let permit = semaphore::get_semaphore().acquire().await?;
            let eval = evaluate_gaia_single(problem, model).await;

            drop(permit);
            Ok::<_, anyhow::Error>(("without tool", eval))
        });
    }

    for problem in &problems {
        let problem = problem.clone();
        let toolbox = toolbox.clone();
        set.spawn(async move {
            let permit = semaphore::get_semaphore().acquire().await?;
            let eval = evaluate_gaia_single_with_tool(problem, model, toolbox).await;

            drop(permit);
            Ok::<_, anyhow::Error>(("with tool", eval))
        });
    }
    let mut results: HashMap<&str, Vec<GaiaEvalResult>> = HashMap::new();
    while let Some(Ok(result)) = set.join_next().await {
        match result {
            Ok((group, eval)) => {
                tracing::info!("[{group}]:{eval:#?}");
                results.entry(group).or_default().push(eval);
            }
            Err(e) => {
                tracing::error!("task paniked: {e}")
            }
        }
    }

    tracing::info!("======with tools vs without tools.=======");
    for (group, evals) in results {
        let correct = evals.iter().filter(|item| item.correct).count();
        let total = evals.len();

        tracing::info!(
            "test result:{group}: {correct}/{total} ({:.1}%)",
            correct as f64 / total as f64 * 100f64
        );
    }

    Ok(())
}
