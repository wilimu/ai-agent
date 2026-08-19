use std::{collections::HashMap, print, time::Duration};

use ai_agent::{
    constant,
    gaia::{dataset::load_gaia_level1, evaluator::evaluate_gaia_single, models::GaiaEvalResult},
    llm::semaphore,
};
use std::io::{self, Write};
use tokio::{task::JoinSet, time::timeout};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

const MODELS: &[&str] = &[
    constant::GEMMA_4_26B_FREE,
    constant::GPT_OSS_20B_FREE,
    constant::GPT_OSS_120B,
    constant::DEEPSEEK_V4_FLASH,
    constant::CLAUDE_HAIHU,
];

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    print!("starting");
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
    let mut set = JoinSet::new();
    for model in MODELS.iter() {
        for problem in &problems {
            let problem = problem.clone();
            set.spawn(async move {
                let permit = semaphore::get_semaphore().acquire().await?;
                let eval_result = timeout(
                    Duration::from_secs(10),
                    evaluate_gaia_single(problem, model),
                )
                .await;

                drop(permit);
                if let Ok(eval) = eval_result {
                    Ok::<_, anyhow::Error>((model, eval))
                } else {
                    Err(anyhow::anyhow!("timeout"))
                }
            });
        }
    }
    let mut results: HashMap<&str, Vec<GaiaEvalResult>> = HashMap::new();
    while let Some(Ok(result)) = set.join_next().await {
        match result {
            Ok((model, eval)) => {
                tracing::info!("[{model}]:{eval:#?}");
                results.entry(model).or_default().push(eval);
            }
            Err(e) => {
                tracing::error!("task paniked: {e}")
            }
        }
    }
    for (model, evals) in results {
        let correct = evals.iter().filter(|item| item.correct).count();
        let total = evals.len();

        tracing::info!(
            "test result:{model}: {correct}/{total} ({:.1}%)",
            correct as f64 / total as f64 * 100f64
        );
    }

    Ok(())
}
