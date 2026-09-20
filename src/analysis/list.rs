//! Header-only tensor discovery, including checkpoints without a known model config.

use crate::storage::{TensorIndex, TensorInfo};
use std::error::Error;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListOptions {
    pub filter: Option<String>,
    pub limit: usize,
}

impl Default for ListOptions {
    fn default() -> Self {
        Self {
            filter: None,
            limit: 100,
        }
    }
}

impl ListOptions {
    pub fn validate(&self) -> Result<(), String> {
        if self.limit == 0 || self.limit > 100_000 {
            return Err("--limit must be in 1..=100000".into());
        }
        Ok(())
    }
}

pub struct TensorListing {
    pub model_path: PathBuf,
    pub options: ListOptions,
    pub total_matches: usize,
    pub tensors: Vec<TensorInfo>,
}

pub fn list_tensors(
    model_dir: &Path,
    options: ListOptions,
) -> Result<TensorListing, Box<dyn Error + Send + Sync>> {
    options.validate()?;
    let index = TensorIndex::open(model_dir)?;
    let mut tensors = Vec::new();
    let mut total_matches = 0;
    for tensor in index.tensors().filter(|tensor| {
        options
            .filter
            .as_deref()
            .is_none_or(|filter| tensor.name.contains(filter))
    }) {
        total_matches += 1;
        if tensors.len() < options.limit {
            tensors.push(tensor.clone());
        }
    }
    Ok(TensorListing {
        model_path: model_dir.to_owned(),
        options,
        total_matches,
        tensors,
    })
}
