use std::collections::BinaryHeap;

use anyhow::Ok;

use crate::{
    constant,
    knowledge_base::embed::{embed_text, embed_texts},
};

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub text: String,
    pub similarity: f32,
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0f32 || norm_b == 0f32 {
        0f32
    } else {
        dot / (norm_a * norm_b)
    }
}

pub struct ScoredIndex {
    similarity: f32,
    index: usize,
}

impl PartialEq for ScoredIndex {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for ScoredIndex {}

impl Ord for ScoredIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.similarity.total_cmp(&self.similarity)
    }
}

impl PartialOrd for ScoredIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub async fn vector_search(
    query: &str,
    chunks: &[String],
    top_k: usize,
) -> anyhow::Result<Vec<SearchHit>> {
    if chunks.is_empty() || top_k == 0 {
        return Ok(Vec::new());
    }
    let model = constant::OPENAI_TEXT_EMBEDDING_SMALL;
    let query_embedding = embed_text(query, model).await?;
    let chunk_embeddings = embed_texts(chunks, model).await?;

    let mut heap: BinaryHeap<ScoredIndex> = BinaryHeap::with_capacity(top_k);

    for (index, embedding) in chunk_embeddings.iter().enumerate() {
        let similarity = cosine_similarity(&query_embedding, embedding);
        let item = ScoredIndex { similarity, index };
        if heap.len() < heap.capacity() {
            heap.push(item);
        } else if let Some(mut peek) = heap.peek_mut()
            && peek.gt(&item)
        {
            *peek = item;
        }
    }

    let mut ranked = heap.into_sorted_vec();
    ranked.reverse();
    Ok(ranked
        .into_iter()
        .map(|scored| SearchHit {
            text: chunks[scored.index].clone(),
            similarity: scored.similarity,
        })
        .collect())
}
