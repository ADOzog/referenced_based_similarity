mod types;
use std::{
    collections::{BinaryHeap, HashMap, HashSet},
    fs,
    ops::Deref,
    sync::Arc,
    vec,
};

use argmin::solver::particleswarm::ParticleSwarm;
use argmin::{
    core::{CostFunction, Error, Executor},
    solver,
};
use futures::{executor::block_on, future::try_join_all};
use hf_hub::api::sync::Api;
use ollama_rs::{Ollama, generation::embeddings::request::GenerateEmbeddingsRequest};
use rand::{
    SeedableRng,
    rngs::{SmallRng, StdRng},
    seq::SliceRandom,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde_json::Deserializer;
use types::*;

// Add an individual test for this
pub async fn build_embeddings(
    ollama_cli: &Ollama,
    documents: &[String],
    embedding_model_list: &[String],
    labels: Option<&[&str]>, // truncate: Option<bool>,
) -> Result<HashMap<DocModelKey, EmbMaybeLabel>, RBSError> {
    // add code to test that ollama is running else ret error and tell user
    // add code validate models using ollama list
    // need to change in the future to allow use to input stuff
    let mut embs_of_doc: HashMap<DocModelKey, EmbMaybeLabel> = HashMap::new();
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
                EmbMaybeLabel {
                    emb: res.next().unwrap(),
                    label: match labels {
                        Some(l) => Some(l[i].to_string()),
                        None => None,
                    },
                },
            );
        }
    }
    Ok(embs_of_doc)
}

pub async fn k_most_similar(
    ollama_cli: &Ollama,
    doc: &str,
    embs_set: &HashMap<DocModelKey, EmbMaybeLabel>,
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
fn average(vec: &Vec<f64>) -> f64 {
    let sum: f64 = vec.par_iter().sum();
    sum / vec.len() as f64
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

    // the error is here
    let train_data = Deserializer::from_slice(&train_data_raw)
        .into_iter::<NewsDP>()
        .map(|x| x.unwrap());

    let test_data = Deserializer::from_slice(&test_data_raw)
        .into_iter::<NewsDP>()
        .map(|x| x.unwrap());
    let (documents, labels): (Vec<String>, Vec<String>) = train_data
        .chain(test_data)
        .map(|dp| (dp.text, dp.label_text))
        .unzip();

    let ollama_cli = ollama_rs::Ollama::default();
    build_embeddings(
        &ollama_cli,
        &documents[..=50],
        embedding_model_list,
        Some(&labels.iter().map(|x| x.as_str()).collect::<Vec<&str>>()[..=50]),
    )
    .await
}

fn obj_fn_builder(
    given_weights: &[f64],
    data_set: HashMap<DocModelKey, EmbMaybeLabel>,
    ks: &Vec<usize>,
    given_runs: &Option<usize>,
    embedding_model_list: &[String],
    doc_label_hash: &HashMap<String, String>,
) -> f64 {
    let size = data_set.len(); // Adjust the size as needed
    let runs = given_runs.unwrap_or(30);
    let true_count = size - runs;
    let false_count = runs;
    let mut split_locs: Vec<bool> = vec![true; true_count]
        .into_iter()
        .chain(vec![false; false_count])
        .collect();

    let mut rng = SmallRng::seed_from_u64(42);

    split_locs.shuffle(&mut rng);

    let (train_w_bool, target_w_bool): (
        Vec<(DocModelKey, EmbMaybeLabel, bool)>,
        Vec<(DocModelKey, EmbMaybeLabel, bool)>,
    ) = data_set
        .into_iter()
        .zip(split_locs.drain(..))
        .map(|(l, r)| (l.0, l.1, r))
        .partition(|(_, _, b)| *b);
    let train: HashMap<DocModelKey, EmbMaybeLabel> =
        train_w_bool.into_iter().map(|a| (a.0, a.1)).collect();
    let (targets, labels): (Vec<String>, Vec<String>) = target_w_bool
        .into_iter()
        .map(|a| (a.0.document, a.1.label.unwrap_or_default()))
        .unzip();

    let k_max: &usize = ks.iter().max().unwrap();
    // Build the weights here

    let ws: HashMap<&str, f32> = given_weights
        .into_iter()
        .zip(embedding_model_list)
        .map(|(l, r)| (r.as_str(), *l as f32))
        .collect();
    // wrap train in a clone
    let arc_train = Arc::new(train);

    let ks_for_each_tar: Vec<Vec<String>> = block_on(try_join_all(
        targets
            .par_iter()
            .map(|target| {
                let sub_train = arc_train.clone();
                let value = ws.clone();
                let ollama_cli = ollama_rs::Ollama::default();
                async move {
                    k_most_similar(
                        &ollama_cli,
                        &target,
                        &sub_train,
                        Some(value.clone()),
                        *k_max,
                    )
                    .await
                }
            })
            .collect::<Vec<_>>(),
    ))
    .unwrap_or_default();

    let arc_doc_label_hash = Arc::new(doc_label_hash);

    // This needs to be fixed for each k next
    average(
        &ks_for_each_tar
            .into_iter()
            .zip(labels.into_iter())
            .map(|(data, label)| {
                data.into_iter()
                    .map({
                        let dict = arc_doc_label_hash.clone();
                        move |text| *dict.get(&text).unwrap_or(&String::new()) == label
                    })
                    .map(|tf| tf as usize as f64)
                    .collect::<Vec<f64>>()
            })
            .map(|sub_vec| average(&sub_vec))
            .collect(),
    )
}

struct PSOobj {
    given_weights: Vec<f64>,
    data_set: HashMap<DocModelKey, EmbMaybeLabel>,
    ks: Vec<usize>,
    given_runs: Option<usize>,
    embedding_model_list: Vec<String>,
    doc_label_hash: HashMap<String, String>,
}

impl CostFunction for PSOobj {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        // Negative
        Ok(obj_fn_builder(
            param.as_slice(),
            self.data_set.clone(),
            &self.ks,
            &self.given_runs,
            &self.embedding_model_list,
            &self.doc_label_hash,
        ))
    }
}

pub async fn optimize_average_weights(
    embedding_model_list: &[String],
    given_data_set: Option<HashMap<DocModelKey, EmbMaybeLabel>>, // Re-think this type to fit what-ever
    given_ks: Option<Vec<usize>>,
    given_runs: Option<usize>,
) -> Result<HashMap<String, f32>, RBSError> {
    // Cut the train and test split? just do opti for what-ever is given
    // Re-think the clones in this function, write paper first
    let data_set_w_labels: HashMap<DocModelKey, EmbMaybeLabel> =
        given_data_set.unwrap_or(init_20news(embedding_model_list).await?);
    let ks = given_ks.unwrap_or(vec![1, 2, 4, 8, 16, 32, 64]);
    let doc_label_hash: HashMap<String, String> = data_set_w_labels
        .clone()
        .into_iter()
        .filter(|(_k, v)| v.label.is_some())
        .map(|(k, v)| (k.document, v.label.unwrap()))
        .collect();

    let number_of_models: usize = embedding_model_list.len();
    let cost_function: PSOobj = PSOobj {
        given_weights: vec![0.0; number_of_models],
        data_set: data_set_w_labels.clone(),
        ks,
        given_runs,
        embedding_model_list: embedding_model_list.to_vec(),
        doc_label_hash,
    };
    let pso_solver = ParticleSwarm::new(
        (vec![0.0; number_of_models], vec![1.0; number_of_models]),
        40,
    );
    println!("right before exec");
    let res = Executor::new(cost_function, pso_solver)
        .configure(|state| state.max_iters(100))
        .run()?
        .state
        .best_individual
        .expect("Failed right here")
        .position;
    println!("After the opt");
    Ok(embedding_model_list
        .to_vec()
        .into_iter()
        .zip(res.into_iter().map(|x| x as f32))
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
}
