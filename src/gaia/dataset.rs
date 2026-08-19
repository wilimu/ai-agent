use crate::gaia::models::{GaiaRow, HfResponse};

pub async fn load_gaia_level1() -> anyhow::Result<Vec<GaiaRow>> {
    let token = std::env::var("HF_TOKEN")?;
    let client = reqwest::Client::new();
    let url = "https://datasets-server.huggingface.co/rows";
    let response: HfResponse = client
        .get(url)
        .query(&[
            ("dataset", "gaia-benchmark/GAIA"),
            ("config", "2023_level1"),
            ("split", "validation"),
            ("offset", "20"),
            ("length", "5"),
        ])
        .bearer_auth(token)
        .send()
        .await?
        .json()
        .await?;
    Ok(response.rows.into_iter().map(|r| r.row).collect())
}
