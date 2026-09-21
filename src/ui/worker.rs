use super::commands::GenerateOptions;
use super::generate::{self, Generation, Outcome};
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
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
    Generate(GenerateOptions),
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
            Self::Generate(_) => "Generating",
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
            Self::Generate(_) => "/generate",
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

pub enum Event {
    Analysis(Finished),
    Generation { id: u64, event: generate::Event },
}

pub struct Worker {
    requests: SyncSender<Request>,
    results: Receiver<Finished>,
    generation: Option<Generation>,
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
                        // submit() routes generation to a process owned by the UI thread.
                        Task::Generate(_) => unreachable!("generation is not an analysis request"),
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
        Ok(Self {
            requests,
            results,
            generation: None,
        })
    }

    pub fn submit(&mut self, request: Request) -> Result<(), String> {
        if let Task::Generate(options) = &request.task {
            if self.generation.is_some() {
                return Err("Generation is already running.".into());
            }
            self.generation = Some(
                Generation::start(request.id, &request.path, options)
                    .map_err(|error| format!("Cannot start generation: {error}"))?,
            );
            return Ok(());
        }
        self.requests
            .try_send(request)
            .map_err(|error| format!("Cannot start analysis: {error}"))
    }

    pub fn poll(&mut self) -> Result<Option<Event>, String> {
        if let Some(generation) = &mut self.generation {
            let id = generation.id;
            let event = match generation.poll() {
                Ok(event) => event,
                Err(error) => Some(generate::Event::Finished(Outcome::Failed(
                    error.to_string(),
                ))),
            };
            if matches!(event, Some(generate::Event::Finished(_))) {
                self.generation = None;
            }
            return Ok(event.map(|event| Event::Generation { id, event }));
        }
        match self.results.try_recv() {
            Ok(event) => Ok(Some(Event::Analysis(event))),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err("analysis worker stopped unexpectedly".into()),
        }
    }

    pub fn cancel_generation(&mut self) -> Result<(), String> {
        if let Some(generation) = &mut self.generation {
            generation
                .cancel()
                .map_err(|error| format!("Cannot stop generation: {error}"))?;
        }
        Ok(())
    }
}
