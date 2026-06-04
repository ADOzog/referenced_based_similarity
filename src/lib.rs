mod types;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    vec,
};

use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use types::*;
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub async fn build_embeddings(
    ollama_cli: &Ollama,
    documents: &[String],
    embedding_model_list: &[String],
    // truncate: Option<bool>,
) -> Result<HashMap<DocModelKey, Vec<f32>>, RBSError> {
    // add code to test that ollama is running else ret error and tell user
    // add code validate models using ollama list
    // need to change in the future to allow use to input stuff
    let mut embs_of_doc: HashMap<DocModelKey, Vec<f32>> = HashMap::new();
    for m in embedding_model_list {
        let emb_request = GenerateEmbeddingsRequest::new(m.to_string(), documents.to_vec().into());
        let mut res = ollama_cli
            .generate_embeddings(emb_request)
            .await?
            .embeddings
            .into_iter();
        for i in 0..res.len() {
            embs_of_doc.insert(
                DocModelKey {
                    document: documents[i].to_string(),
                    model: m.to_string(),
                },
                res.next().unwrap(),
            );
        }
    }
    Ok(embs_of_doc)
}

pub async fn k_most_similar(
    ollama_cli: &Ollama,
    doc: &str,
    embs_set: &HashMap<DocModelKey, Vec<f32>>,
    avg_weights: Option<HashMap<&str, f32>>,
    k: usize,
) -> Result<Vec<String>, RBSError> {
    let (list_of_docs, list_of_models): (HashSet<&str>, HashSet<&str>) = embs_set
        .iter()
        .map(|(key, _value)| (key.document.as_str(), key.model.as_str()))
        .unzip();

    let mut new_sims: HashMap<DocModelKey, f32> = HashMap::new();

    let num_of_models: usize = list_of_models.len();
    let num_of_docs: usize = list_of_docs.len();
    let models: Vec<&str> = list_of_models.into_iter().collect();
    let docs: Vec<&str> = list_of_docs.into_iter().collect();

    let ws: HashMap<&str, f32> = match avg_weights {
        Some(ws) => {
            if num_of_models != ws.len() {
                return Err(RBSError::KMostSim(
                    "The number of weights does not match the number of models you provided"
                        .to_string(),
                ));
            } else {
                ws
            }
        }
        None => models
            .iter()
            .zip(vec![1.0_f32 / num_of_models as f32; num_of_models])
            .map(|(x, y)| (*x, y))
            .collect::<HashMap<&str, f32>>(),
    };
    for i in 0..num_of_models {
        let emb_request =
            GenerateEmbeddingsRequest::new(models[i].to_string(), doc.to_string().into());
        let new_emb: Vec<f32> = (*ollama_cli
            .generate_embeddings(emb_request)
            .await?
            .embeddings[0])
            .to_vec();
        for d in &docs {
            let dmkey = DocModelKey {
                document: d.to_string(),
                model: models[i].to_string(),
            };
            new_sims.insert(
                dmkey.clone(),
                dot(
                    &new_emb,
                    embs_set.get(&dmkey).expect("Embeddings not found"),
                ),
            );
        }
    }
    let mut w_avgs: Vec<Scores> = vec![
        Scores {
            document: "".to_string(),
            score: 0.0
        };
        num_of_docs
    ];
    let mut counter: usize = 0;

    for d in docs {
        let mut sum: f32 = 0.0;
        for m in &models {
            let dmkey = DocModelKey {
                document: d.to_string(),
                model: m.to_string(),
            };
            sum += ws.get(m).unwrap() * new_sims.get(&dmkey).unwrap();
        }
        // update w_avgs here
        w_avgs[counter] = Scores {
            document: d.to_string(),
            score: sum,
        };
        counter += 1;
    }
    // From here just get the top k

    let mut heap: BinaryHeap<Scores> = BinaryHeap::from(w_avgs);
    let mut top_k_docs: Vec<String> = vec!["".to_string(); k];
    for i in 0..k {
        top_k_docs[i] = heap.pop().unwrap().document;
    }
    Ok(top_k_docs)
}

fn dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum()
}

pub async fn optimize_average_weights(embedding_model_list: &Vec<String>) {}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
