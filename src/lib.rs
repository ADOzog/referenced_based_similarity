mod types;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    fs, vec,
};

use hf_hub::api::sync::Api;
use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde_json::Deserializer;
use types::*;

// Add an individual test for this
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

// Add a test for this
fn dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum()
}

pub async fn optimize_average_weights(
    embedding_model_list: &[String],
    data_set: Option<&[(String, String)]>, // Re-think this type to fit what-ever
) -> Result<HashMap<String, f32>, RBSError> {
    let (train_doc_w_labels, test_doc_w_lables): (Vec<(String, String)>, Vec<(String, String)>) =
        match data_set {
            Some(ds) => {
                let size = ds.len(); // Adjust the size as needed
                let true_count = (size as f64 * 0.8).round() as usize;
                let false_count = size - true_count;
                let mut split_locs: Vec<bool> = vec![true; true_count]
                    .into_iter()
                    .chain(vec![false; false_count])
                    .collect();

                let mut rng = SmallRng::seed_from_u64(42);

                split_locs.shuffle(&mut rng);

                let (rhs, lhs): (Vec<_>, Vec<_>) = ds
                    .to_vec()
                    .drain(..)
                    .zip(split_locs.into_iter())
                    .map(|(l, r)| (l.0, l.1, r))
                    .partition(|(_, _, b)| *b);
                (
                    rhs.into_iter().map(|(x, y, _)| (x, y)).collect(),
                    lhs.into_iter().map(|(x, y, _)| (x, y)).collect(),
                )
            }
            None => {
                let hf_api = Api::new()?;
                let repo = hf_api.dataset("SetFit/20_newsgroups".to_string());

                let train_path = repo.get("train.jsonl")?;
                let test_path = repo.get("test.jsonl")?;

                let train_data_raw = fs::read(train_path)?;
                let test_data_raw = fs::read(test_path)?;

                // the error is here
                let train_data: Vec<NewsDP> = Deserializer::from_slice(&train_data_raw)
                    .into_iter::<NewsDP>()
                    .map(|x| x.unwrap())
                    .collect();

                let test_data: Vec<NewsDP> = Deserializer::from_slice(&test_data_raw)
                    .into_iter::<NewsDP>()
                    .map(|x| x.unwrap())
                    .collect();
                //println!("{:#?}", train_data);
                //println!("{:#?}", test_data);
                (
                    train_data
                        .into_iter()
                        .map(|dp| (dp.text, dp.label_text))
                        .collect(),
                    test_data
                        .into_iter()
                        .map(|dp| (dp.text, dp.label_text))
                        .collect(),
                )
            }
        };
    // Now add pso stuff

    todo!()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simp_emb_and_top_one_test() {
        let ollama_cli = ollama_rs::Ollama::default();

        let doc_collection = vec![
            "Trees have leafs".to_string(),
            "Cats have tails".to_string(),
            "I am so glad I have had the time to write this library".to_string(),
            "Just one more sentance".to_string(),
        ];
        let models = vec![
            "nomic-embed-text:latest".to_string(),
            "bge-m3:latest".to_string(),
        ];
        let embs_set = build_embeddings(&ollama_cli, &doc_collection[..], &models[..])
            .await
            .unwrap();

        let new_doc = "This is my doc about trees";
        let ws = Option::None;
        let k = 1;

        let top_k: Vec<String> = k_most_similar(&ollama_cli, new_doc, &embs_set, ws, k)
            .await
            .unwrap();
        assert_eq!(top_k[0], doc_collection[0]);
    }
    #[tokio::test]
    async fn custom_weights_emb_and_top_k_test() {
        let ollama_cli = ollama_rs::Ollama::default();

        let doc_collection = vec![
            "Trees have leafs".to_string(),
            "Cats have tails".to_string(),
            "I am so glad I have had the time to write this library".to_string(),
            "Just one more sentance".to_string(),
            "Trees are good for the earth".to_string(),
        ];
        let models = vec![
            "nomic-embed-text:latest".to_string(),
            "bge-m3:latest".to_string(),
        ];
        let embs_set = build_embeddings(&ollama_cli, &doc_collection[..], &models[..])
            .await
            .unwrap();

        let new_doc = "This is my doc about trees";

        let mut ws: HashMap<&str, f32> = HashMap::new();
        ws.insert(&models[0], 0.6);
        ws.insert(&models[1], 0.4);
        let wrapped_ws = Option::Some(ws);
        let k = 2;

        let top_k: Vec<String> = k_most_similar(&ollama_cli, new_doc, &embs_set, wrapped_ws, k)
            .await
            .unwrap();
        let mut top_ks_unorderd = HashSet::new();
        top_ks_unorderd.insert(&top_k[0]);
        top_ks_unorderd.insert(&top_k[1]);
        let mut my_guesses = HashSet::new();
        my_guesses.insert(&doc_collection[0]);
        my_guesses.insert(&doc_collection[4]);
        assert_eq!(top_ks_unorderd, my_guesses);
    }

    #[tokio::test]
    async fn opt_weights_emb_and_top_k_test() {}

    #[tokio::test]
    async fn optimizer_test_20news() {
        let models = vec![
            "nomic-embed-text:latest".to_string(),
            "bge-m3:latest".to_string(),
        ];
        let ws: HashMap<String, f32> = optimize_average_weights(&models[..], Option::None)
            .await
            .unwrap();
    }
    /*
    #[tokio::test]
    async fn optimizer_test_custom_data() {
        let models = vec![
            "nomic-embed-text:latest".to_string(),
            "bge-m3:latest".to_string(),
        ];
        let the_opt_fn: fn(&[String], Option<&[String]>) -> HashMap<String, f32> =
            optimize_average_weights(embedding_model_list, data_set);
    }
    */
}
