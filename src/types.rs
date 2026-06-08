use std::{cmp::Ordering, collections::HashMap};

use ollama_rs::error::OllamaError;
#[derive(Eq, Hash, PartialEq, Clone)]
pub struct DocModelKey {
    pub document: String,
    pub model: String,
}

#[derive(Debug)]
pub enum RBSError {
    Ollama(String),
    KMostSim(String),
}

impl From<OllamaError> for RBSError {
    fn from(err: OllamaError) -> RBSError {
        RBSError::Ollama(err.to_string())
    }
}

#[derive(Clone)]
pub struct Scores {
    pub document: String,
    pub score: f32,
}

impl PartialEq for Scores {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal
    }
}
impl Eq for Scores {}

impl PartialOrd for Scores {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scores {
    fn cmp(&self, other: &Self) -> Ordering {
        // use total_cmp to get a total ordering for f32 (handles NaN, -0.0, etc.)
        self.score.total_cmp(&other.score)
    }
}
