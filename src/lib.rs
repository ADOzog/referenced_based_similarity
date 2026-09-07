mod types;
use core::num;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    fs,
    ops::Deref,
    vec,
};

use hf_hub::api::sync::Api;
use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde_json::Deserializer;
use types::*;

// Add an individual test for this
pub async fn build_embeddings<'a>(
    ollama_cli: &Ollama,
    documents: &'a [String],
    embedding_model_list: &'a [String],
    labels: Option<&'a [&'a str]>, // truncate: Option<bool>,
) -> Result<HashMap<DocModelKey<'a>, EmbMaybeLabel<'a>>, RBSError> {
    let mut embs_of_doc = HashMap::new();

    for m in embedding_model_list {
        let emb_request = GenerateEmbeddingsRequest::new(m.clone(), documents.to_vec().into());
        let embeddings = ollama_cli
            .generate_embeddings(emb_request)
            .await?
            .embeddings;

        for (i, (doc, emb)) in documents.iter().zip(embeddings.into_iter()).enumerate() {
            embs_of_doc.insert(
                DocModelKey {
                    document: doc,
                    model: m,
                },
                EmbMaybeLabel {
                    emb,
                    label: labels.and_then(|l| l.get(i).copied()),
                },
            );
        }
    }

    Ok(embs_of_doc)
}

pub async fn k_most_similar<'a>(
    ollama_cli: &Ollama,
    doc: &str,
    embs_set: &HashMap<DocModelKey<'a>, EmbMaybeLabel<'a>>,
    avg_weights: Option<HashMap<&str, f32>>,
    k: usize,
) -> Result<Vec<String>, RBSError> {
    let (list_of_docs, list_of_models): (HashSet<&str>, HashSet<&str>) = embs_set
        .iter()
        .map(|(key, _value)| (key.document, key.model))
        .unzip();

    let mut new_sims: HashMap<DocModelKey, f32> = HashMap::new();

    let num_of_models: usize = list_of_models.len();
    let num_of_docs: usize = list_of_docs.len();
    //println!("{:#?}", num_of_docs);
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
        let new_emb: Vec<f32> = match ollama_cli
            .generate_embeddings(emb_request)
            .await?
            .embeddings
            .get(0)
        {
            Some(emb) => emb.to_vec(),
            None => {
                return Err(RBSError::KMostSim(format!(
                    "No embedding returned for doc: {doc}"
                )));
            }
        };
        for d in &docs {
            let dmkey = DocModelKey {
                document: d,
                model: models[i],
            };
            new_sims.insert(
                dmkey.clone(),
                dot(
                    &new_emb,
                    &embs_set.get(&dmkey).expect("Embeddings not found").emb,
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
                document: d,
                model: m,
            };
            sum += ws.get(m).unwrap() * new_sims.get(&dmkey).unwrap();
            //println!("the sum was,{:#?}", sum)
        }
        // updantlnte w_avgs here
        w_avgs[counter] = Scores {
            document: d.to_string(),
            score: sum,
        };
        counter += 1;
    }
    // From here just get the top k

    let mut heap: BinaryHeap<Scores> = BinaryHeap::from(w_avgs);
    let mut top_k_docs = Vec::with_capacity(k);

    for _ in 0..k.min(num_of_docs) {
        if let Some(score) = heap.pop() {
            top_k_docs.push(score.document);
        }
    }
    Ok(top_k_docs)
}

// Add a test for this
fn dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.par_iter().zip(y.par_iter()).map(|(a, b)| a * b).sum()
}
fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.par_iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.par_iter().sum();
    exps.par_iter().map(|e| e / sum).collect()
}

async fn init_20news(
    embedding_model_list: &[String],
) -> Result<HashMap<DocModelKey, EmbMaybeLabel>, RBSError> {
    let hf_api = Api::new()?;
    let repo = hf_api.dataset("SetFit/20_newsgroups".to_string());

    let train_path = repo.get("train.jsonl")?;
    let test_path = repo.get("test.jsonl")?;

    let train_data_raw = fs::read(train_path)?;
    let test_data_raw = fs::read(test_path)?;

    let (documents, labels): (Vec<String>, Vec<String>) = Deserializer::from_slice(&train_data_raw)
        .into_iter::<NewsDP>()
        .chain(Deserializer::from_slice(&test_data_raw).into_iter::<NewsDP>())
        .collect::<Result<Vec<NewsDP>, _>>()?
        .into_iter()
        .filter(|dp| !dp.text.is_empty() && !dp.label_text.is_empty())
        .map(|dp| (dp.text, dp.label_text))
        .unzip();

    let n = documents.len().min(501);

    let docs = documents.into_iter().take(n).collect::<Vec<_>>();
    let labs = labels.into_iter().take(n).collect::<Vec<_>>();

    let ollama_cli = ollama_rs::Ollama::default();
    let label_refs: Vec<&str> = labs.iter().map(|s| s.as_str()).collect();

    build_embeddings(&ollama_cli, &docs, embedding_model_list, Some(&label_refs)).await
}

/* come back to here test rest first
async fn avg_score_for_k(
    ks: &Vec<usize>,
    doc_label_hash: &HashMap<String, String>,
    ollama_cli: &Ollama,
    embs_set: &HashMap<DocModelKey, EmbMaybeLabel>,
    weights: Option<HashMap<&str, f32>>,
) -> Result<f32, RBSError> {
    let mut score_at_k: Vec<f32> = vec![];
    for k in ks {
        let mut sum_at_k = 0;
        for (doc, label) in doc_label_hash.clone() {
            if doc.is_empty() {
                println!("An empty doc was found with label {:#?}", label)
            }
            let found_docs =
                k_most_similar(&ollama_cli, &doc, &embs_set, weights.clone(), *k).await?;
            for f_doc in found_docs {
                // small fix here maybe? nah
                if label == doc_label_hash.get(&f_doc).unwrap_or(&String::new()).deref() {
                    sum_at_k += 1;
                }
            }
            score_at_k.push((sum_at_k / k) as f32);
        }
    }
    Ok(score_at_k.iter().sum::<f32>() / score_at_k.len() as f32)
}

async fn fin_diff_grad(
    ks: &Vec<usize>,
    doc_label_hash: &HashMap<String, String>,
    ollama_cli: &Ollama,
    embs_set: &HashMap<DocModelKey, EmbMaybeLabel>,
    weights: Option<HashMap<&str, f32>>,
    theta: &[f32],
    eps: f32,
) -> Result<Vec<f32>, RBSError> {
    let mut grad = vec![0.0; theta.len()];
    let base = avg_score_for_k(ks, doc_label_hash, ollama_cli, embs_set, weights.clone()).await?;

    for i in 0..theta.len() {
        let mut t = theta.to_vec();
        t[i] += eps;
        let v = avg_score_for_k(ks, doc_label_hash, ollama_cli, embs_set, weights.clone()).await?;
        grad[i] = (v - base) / eps;
    }
    Ok(grad)
}

pub async fn optimize_average_weights(
    embedding_model_list: &[String],
    given_data_set: Option<HashMap<DocModelKey, EmbMaybeLabel>>, // Re-think this type to fit what-ever
    given_ks: Option<Vec<usize>>,
    given_runs: Option<usize>,
) -> Result<HashMap<String, f32>, RBSError> {
    // Cut the train and test split? just do opti for what-ever is given
    // Re-think the clones in this function, write paper first
    let ollama_cli = ollama_rs::Ollama::default();
    let data_set_w_labels: HashMap<DocModelKey, EmbMaybeLabel> =
        given_data_set.unwrap_or(init_20news(embedding_model_list).await?);
    println!(
        "The number of data points is {:#?}",
        data_set_w_labels.len(),
    );
    let ks = given_ks.unwrap_or(vec![1, 2, 4, 8, 16, 32, 64]);
    let doc_label_hash: HashMap<String, String> = data_set_w_labels
        .clone()
        .into_iter()
        .filter_map(|(k, v)| {
            v.label
                .as_ref()
                .map(|label| (k.document.clone(), label.clone()))
        })
        // should be the fix
        .filter(|(k, v)| !(k.is_empty() || v.is_empty()))
        .collect();
    let labeled_docs: Vec<(String, String)> = data_set_w_labels
        .into_iter()
        .filter_map(|(k, v)| v.label.map(|label| (k.document, label)))
        .collect();

    let docs: Vec<String> = labeled_docs.iter().map(|(d, _)| d.clone()).collect();
    let labels: Vec<&str> = labeled_docs.iter().map(|(_, l)| l.as_str()).collect();
    let embs_set =
        build_embeddings(&ollama_cli, &docs, embedding_model_list, Some(&labels)).await?;
    let runs = given_runs.unwrap_or(100);

    let number_of_models: usize = embedding_model_list.len();
    let mut theta: Vec<f32> = vec![0.0; number_of_models];

    let eps: f32 = 1e-3;
    let mut best_theta = theta.clone();
    let mut best_score = f32::NEG_INFINITY;

    for _ in 0..runs {
        let weights: HashMap<&str, f32> = softmax(&theta)
            .into_iter()
            .zip(embedding_model_list.iter())
            .map(|(f, s)| (s.as_str(), f))
            .collect();

        let score = avg_score_for_k(
            &ks,
            &doc_label_hash.clone(),
            &ollama_cli,
            &embs_set,
            Some(weights.clone()),
        )
        .await?;
        if score > best_score {
            best_score = score;
            best_theta = theta.clone();
        }

        let grad = fin_diff_grad(
            &ks,
            &doc_label_hash,
            &ollama_cli,
            &embs_set,
            Some(weights.clone()),
            &theta,
            eps,
        )
        .await?;
        let lr: f32 = 0.1;
        for i in 0..number_of_models {
            theta[i] += lr * grad[i];
        }
    }
    let final_weights = softmax(&best_theta);

    // res should be the list of weights

    Ok(embedding_model_list
        .to_vec()
        .into_iter()
        .zip(final_weights.into_iter().map(|x| x as f32))
        .collect())
}

/*
async fn precs_at_ks(
    data_set: HashMap<DocModelKey, EmbMaybeLabel>,
    do_label_hash: HashMap<String, String>,
    ks: Vec<usize>,
    Targets: HashMap<DocModelKey, EmbMaybeLabel>,
) -> Vec<usize> {
    todo!()
}
*/
*/

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
        let embs_set = build_embeddings(&ollama_cli, &doc_collection[..], &models[..], None)
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
        let embs_set = build_embeddings(&ollama_cli, &doc_collection[..], &models[..], None)
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

    /*
    /*
        #[tokio::test]
        async fn opt_weights_emb_and_top_k_test() {

        }
    */

    #[tokio::test]
    async fn optimizer_test_20news() {
        let models = vec![
            "nomic-embed-text:latest".to_string(),
            "bge-m3:latest".to_string(),
        ];
        let ws: HashMap<String, f32> =
            optimize_average_weights(&models[..], Option::None, Option::None, Option::None)
                .await
                .unwrap();
        println!("{:#?}", ws);
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
    */
}
