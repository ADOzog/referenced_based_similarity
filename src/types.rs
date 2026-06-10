use std::{cmp::Ordering, io::Error};

use hf_hub::api::sync::ApiError;
use ollama_rs::error::OllamaError;
use serde::{Deserialize, Serialize};
use serde_json::Error as JSError;
#[derive(Eq, Hash, PartialEq, Clone)]
pub struct DocModelKey {
    pub document: String,
    pub model: String,
}
pub struct EmbMaybeLabel {
    pub emb: Vec<f32>,
    pub label: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NewsDP {
    pub text: String,
    pub label: u8,
    pub label_text: String,
}

#[derive(Debug)]
pub enum RBSError {
    Ollama(String),
    KMostSim(String),
    HuggingFace(String),
    ReadFile(String),
    Json(String),
}

impl From<OllamaError> for RBSError {
    fn from(err: OllamaError) -> RBSError {
        RBSError::Ollama(err.to_string())
    }
}
impl From<ApiError> for RBSError {
    fn from(err: ApiError) -> RBSError {
        RBSError::HuggingFace(err.to_string())
    }
}
// make a macro for this for future projects
// this patter comes up a lot
impl From<Error> for RBSError {
    fn from(err: Error) -> RBSError {
        RBSError::ReadFile(err.to_string())
    }
}

impl From<JSError> for RBSError {
    fn from(err: JSError) -> RBSError {
        RBSError::ReadFile(err.to_string())
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
