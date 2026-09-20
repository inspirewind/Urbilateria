use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use urbilateria::analysis::{
    decode_tokens, explain_checkpoint, inspect_checkpoint, list_tensors, plan_checkpoint,
    preflight_checkpoint, probe_tensor, tokenize_text, Decoding, Explanation, Inspection,
    ListOptions, PlanOptions, Planning, Preflight, PreflightOptions, ProbeReport, TensorListing,
    Tokenization, TokenizeOptions,
};
use urbilateria::ModelFamily;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Task {
    Inspect,
    Plan(PlanOptions),
    Preflight(PreflightOptions),
    List(ListOptions),
    Explain,
    Probe {
        tensor: String,
        samples: usize,
    },
    Tokenize {
        text: String,
        options: TokenizeOptions,
    },
    Decode {
        ids: Vec<u32>,
        skip_special: bool,
    },
}

impl Task {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Inspect => "Inspecting",
            Self::Plan(_) => "Planning",
            Self::Preflight(_) => "Validating",
            Self::List(_) => "Listing",
            Self::Explain => "Explaining",
            Self::Probe { .. } => "Sampling",
            Self::Tokenize { .. } => "Tokenizing",
            Self::Decode { .. } => "Decoding",
        }
    }

    pub fn command(&self) -> &'static str {
        match self {
            Self::Inspect => "/inspect",
            Self::Plan(_) => "/plan",
            Self::Preflight(_) => "/preflight",
            Self::List(_) => "/list",
            Self::Explain => "/explain",
            Self::Probe { .. } => "/probe",
            Self::Tokenize { .. } => "/tokenize",
            Self::Decode { .. } => "/decode",
        }
    }
}

pub enum Report {
    Inspection(Box<Inspection>),
    Planning(Box<Planning>),
    Preflight(Box<Preflight>),
    Listing(Box<TensorListing>),
    Explanation(Box<Explanation>),
    Probe(Box<ProbeReport>),
    Tokenization(Box<Tokenization>),
    Decoding(Box<Decoding>),
}

impl Report {
    pub fn identity(&self) -> (&Path, Option<ModelFamily>) {
        match self {
            Self::Inspection(result) => (&result.model_path, Some(result.model.family)),
            Self::Planning(result) => (&result.model_path, Some(result.model.family)),
            Self::Preflight(result) => (&result.model_path, Some(result.model.family)),
            Self::Explanation(result) => (&result.model_path, Some(result.model.family)),
            Self::Tokenization(result) => (&result.model_path, Some(result.model.family)),
            Self::Decoding(result) => (&result.model_path, Some(result.model.family)),
            Self::Listing(result) => (&result.model_path, None),
            Self::Probe(result) => (&result.model_path, None),
        }
    }
}

pub struct Request {
    pub id: u64,
    pub path: PathBuf,
    pub task: Task,
}

pub struct Finished {
    pub id: u64,
    pub result: Result<Report, String>,
}

pub struct Worker {
    requests: SyncSender<Request>,
    pub results: Receiver<Finished>,
}

impl Worker {
    pub fn start() -> io::Result<Self> {
        let (requests, work) = mpsc::sync_channel::<Request>(1);
        let (done, results) = mpsc::sync_channel(1);
        // Detached on exit: a slow read-only filesystem operation must not hold the terminal open.
        thread::Builder::new()
            .name("urb-analysis".into())
            .spawn(move || {
                while let Ok(request) = work.recv() {
                    let result = match request.task {
                        Task::Inspect => inspect_checkpoint(&request.path)
                            .map(|report| Report::Inspection(Box::new(report))),
                        Task::Plan(options) => plan_checkpoint(&request.path, options)
                            .map(|report| Report::Planning(Box::new(report))),
                        Task::Preflight(options) => preflight_checkpoint(&request.path, options)
                            .map(|report| Report::Preflight(Box::new(report))),
                        Task::List(options) => list_tensors(&request.path, options)
                            .map(|report| Report::Listing(Box::new(report))),
                        Task::Explain => explain_checkpoint(&request.path)
                            .map(|report| Report::Explanation(Box::new(report))),
                        Task::Probe { tensor, samples } => {
                            probe_tensor(&request.path, &tensor, samples)
                                .map(|report| Report::Probe(Box::new(report)))
                                .map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
                                    Box::new(error)
                                })
                        }
                        Task::Tokenize { text, options } => {
                            tokenize_text(&request.path, text, options)
                                .map(|report| Report::Tokenization(Box::new(report)))
                        }
                        Task::Decode { ids, skip_special } => {
                            decode_tokens(&request.path, ids, skip_special)
                                .map(|report| Report::Decoding(Box::new(report)))
                        }
                    }
                    .map_err(|error| error.to_string());
                    if done
                        .send(Finished {
                            id: request.id,
                            result,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })?;
        Ok(Self { requests, results })
    }

    pub fn submit(&self, request: Request) -> Result<(), String> {
        self.requests
            .try_send(request)
            .map_err(|error| format!("Cannot start analysis: {error}"))
    }
}
