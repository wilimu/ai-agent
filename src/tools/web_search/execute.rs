use anyhow::Context;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WebSearchArgs {
    pub query: String,

    #[schemars(range(min = 0, max = 20))]
    #[serde(default = "default_max_results")]
    pub max_results: u8,

    #[serde(default = "default_topic")]
    pub topic: String,

    #[serde(default)]
    pub time_range: Option<String>,
}

fn default_max_results() -> u8 {
    2
}

fn default_topic() -> String {
    "general".to_string()
}

#[derive(Debug, Serialize)]
struct TavilyRequest<'a> {
    query: &'a str,
    max_results: u8,
    topic: &'a str,

    #[serde(skip_serializing_if = "Option::is_none")]
    time_range: Option<&'a str>,

    search_depth: &'a str,
    include_answer: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub content: String,
    // pub score: Option<f64>
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TavilyResponse {
    results: Vec<SearchResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    answer: Option<String>,
}

pub async fn search_web(args: WebSearchArgs) -> anyhow::Result<TavilyResponse> {
    let api_key = std::env::var("TAVILY_API_KEY").expect("TAVILY_API_KEY not found.");

    let body = TavilyRequest {
        query: &args.query,
        max_results: args.max_results,
        topic: &args.topic,
        time_range: args.time_range.as_deref(),
        search_depth: "advanced",
        include_answer: true,
    };
    let url = "https://api.tavily.com/search".to_string();
    let response = reqwest::Client::new()
        .post(url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .context("request to tavily failed.")?;

    if !response.status().is_success() {
        anyhow::bail!("Tavily rreturned with status: {}", response.status())
    }

    let parsed: TavilyResponse = response.json().await.context("failed to parse response.")?;

    Ok(parsed)
}
