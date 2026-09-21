use super::commands::clean_text;
use super::generate::Stream;
use urbilateria::analysis::{
    Decoding, Explanation, Inspection, InspectionReport, NumericStats, Planning, Preflight,
    PreflightReport, ProbeReport, TensorListing, Tokenization,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Input,
    Info,
    Success,
    Warning,
    Error,
}

pub struct Detail {
    pub label: Option<&'static str>,
    pub text: String,
    pub warning: bool,
}

impl Detail {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            label: None,
            text: text.into(),
            warning: false,
        }
    }

    fn field(label: &'static str, value: impl ToString) -> Self {
        Self {
            label: Some(label),
            text: value.to_string(),
            warning: false,
        }
    }

    fn warning(text: impl Into<String>) -> Self {
        Self {
            label: None,
            text: text.into(),
            warning: true,
        }
    }
}

pub struct Entry {
    pub task_id: Option<u64>,
    pub kind: Kind,
    pub title: String,
    pub details: Vec<Detail>,
}

/// A compact, owned inspection snapshot, independent of scrollback and generation state.
pub struct ModelSummary {
    pub name: String,
    pub family: String,
    pub path: String,
    pub fields: Vec<Detail>,
}

impl ModelSummary {
    pub fn from_inspection(inspection: &Inspection) -> Self {
        let report = inspection_entry(inspection, std::time::Duration::ZERO);
        let quantization: Vec<_> = report
            .details
            .iter()
            .filter(|detail| detail.label == Some("Quantization"))
            .filter_map(|detail| detail.text.split(" · ").next().map(str::to_owned))
            .collect();
        let mut fields: Vec<_> = report
            .details
            .into_iter()
            .filter(|detail| {
                matches!(
                    detail.label,
                    Some(
                        "Architecture"
                            | "Max context"
                            | "Checkpoint"
                            | "Manifest"
                            | "Parameters"
                            | "Experts"
                            | "Base tensors"
                            | "Exact context ceiling"
                    )
                )
            })
            .collect();
        if !quantization.is_empty() {
            fields.push(Detail::field("Quantization", quantization.join(" / ")));
        }
        Self {
            name: clean_text(
                &inspection
                    .model_path
                    .file_name()
                    .unwrap_or(inspection.model_path.as_os_str())
                    .to_string_lossy(),
                256,
            )
            .replace('\n', " "),
            family: inspection.model.family.to_string(),
            path: clean_text(&inspection.model_path.display().to_string(), 1024).replace('\n', " "),
            fields,
        }
    }
}

impl Entry {
    pub fn new(kind: Kind, title: impl Into<String>, details: Vec<Detail>) -> Self {
        // Bound stored output so long sessions and pathological error messages stay responsive.
        let mut remaining = 8192;
        let mut remaining_lines = 128;
        let mut clipped = false;
        let mut bounded = Vec::new();
        for mut detail in details {
            if bounded.len() == 63 || remaining == 0 || remaining_lines == 0 {
                clipped = true;
                break;
            }
            let clean = clean_text(&detail.text, remaining);
            let lines: Vec<_> = clean.lines().take(remaining_lines).collect();
            remaining_lines -= lines.len();
            let text = lines.join("\n");
            clipped |=
                text.len() < clean.len() || clean.chars().count() < detail.text.chars().count();
            remaining = remaining.saturating_sub(text.chars().count());
            detail.text = text;
            bounded.push(detail);
        }
        if clipped {
            bounded.push(Detail::text(
                "Display shortened; run the corresponding urb command in your shell for the full report.",
            ));
        }
        Self {
            task_id: None,
            kind,
            title: clean_text(&title.into(), 1024).replace('\n', " "),
            details: bounded,
        }
    }

    pub fn message(kind: Kind, title: &str, text: impl Into<String>) -> Self {
        Self::new(kind, title, vec![Detail::text(text)])
    }
}

/// Keep generated text separate from runtime diagnostics; only this text enters the transcript.
#[derive(Default)]
pub struct GenerationTranscript {
    text: Tail,
}

impl GenerationTranscript {
    pub fn append(&mut self, stream: Stream, text: &str) {
        match stream {
            Stream::Text => self.text.append(text, 32_768, 256),
            Stream::Log => {}
        }
    }

    pub fn entry(&self, id: u64, kind: Kind, title: String, error: Option<&str>) -> Entry {
        let mut details = vec![Detail::field(
            "Text",
            if self.text.text.is_empty() {
                "(no generated text yet)".into()
            } else {
                self.text.display()
            },
        )];
        if let Some(error) = error {
            details.push(Detail::text(clean_text(error, 1024)));
        }
        // Already sanitized and bounded per stream; bypass the smaller static-report budget.
        Entry {
            task_id: Some(id),
            kind,
            title,
            details,
        }
    }
}

/// Latest generation's diagnostics are independent of chat scrollback and remain visible
/// after completion, failure or cancellation until the next generation/model change.
pub struct RuntimeInfo {
    pub path: String,
    pub status: &'static str,
    pub metrics: crate::progress::Snapshot,
    log: Tail,
}

impl RuntimeInfo {
    pub fn new(path: &std::path::Path) -> Self {
        Self {
            path: clean_text(&path.display().to_string(), 1024).replace('\n', " "),
            status: "Generating",
            metrics: crate::progress::Snapshot::default(),
            log: Tail::default(),
        }
    }

    pub fn append(&mut self, text: &str) {
        self.log.append(text, 8_192, 64);
    }

    pub fn text(&self) -> String {
        if self.log.text.is_empty() {
            "Waiting for runtime diagnostics…".into()
        } else {
            self.log.display()
        }
    }

    pub fn clear_log(&mut self) {
        self.log = Tail::default();
    }
}

#[derive(Default)]
struct Tail {
    text: String,
    clipped: bool,
}

impl Tail {
    fn append(&mut self, text: &str, bytes: usize, lines: usize) {
        self.text.push_str(&clean_text(text, usize::MAX));
        let mut start = self.text.len().saturating_sub(bytes);
        while !self.text.is_char_boundary(start) {
            start += 1;
        }
        if let Some((index, _)) = self.text.match_indices('\n').rev().nth(lines) {
            start = start.max(index + 1);
        }
        self.clipped |= start != 0;
        self.text.drain(..start);
    }

    fn display(&self) -> String {
        if self.clipped {
            format!(
                "[Earlier output omitted; showing recent output]\n{}",
                self.text
            )
        } else {
            self.text.clone()
        }
    }
}

pub fn inspection_entry(inspection: &Inspection, elapsed: std::time::Duration) -> Entry {
    use crate::{human_bytes, human_count};
    let mut details = vec![
        Detail::field("Path", inspection.model_path.display()),
        Detail::field("Family", inspection.model.family),
        Detail::field(
            "Architecture",
            format!(
                "{} layers · {} hidden · {} vocabulary",
                inspection.model.num_hidden_layers,
                inspection.model.hidden_size,
                inspection.model.vocab_size
            ),
        ),
        Detail::field("Max context", inspection.model.max_position_embeddings),
    ];
    match &inspection.report {
        InspectionReport::Checkpoint(report) => {
            details.extend([
                Detail::field(
                    "Checkpoint",
                    format!(
                        "{} shards · {} tensors · {} payload",
                        report.shard_count,
                        report.tensor_count,
                        human_bytes(report.tensor_payload_bytes)
                    ),
                ),
                Detail::field(
                    "Parameters",
                    format!(
                        "{} known · {} estimated active/token",
                        human_count(report.known_logical_parameters),
                        human_count(report.estimated_active_parameters_per_token)
                    ),
                ),
                Detail::field(
                    "Experts",
                    format!(
                        "{} per layer · {} active/token",
                        report.model.routed_experts_per_layer,
                        report.model.active_experts_per_token
                    ),
                ),
                Detail::field(
                    "Base tensors",
                    format!(
                        "{} required · {} missing",
                        report.required_base_tensor_count, report.missing_base_tensor_count
                    ),
                ),
            ]);
            for quant in &report.quantization {
                details.push(Detail::field(
                    "Quantization",
                    format!(
                        "{}-bit · {} matrices · {} weights",
                        quant.bits_per_weight,
                        quant.matrix_count,
                        human_count(quant.logical_parameters)
                    ),
                ));
            }
            details.extend(report.format_notes.iter().cloned().map(Detail::text));
            details.extend(report.warnings.iter().cloned().map(Detail::warning));
        }
        InspectionReport::Qwen38(report) => {
            details.extend([
                Detail::field(
                    "Manifest",
                    format!(
                        "{} shards · {} tensors · {} payload",
                        report.checkpoint_shard_count,
                        report.checkpoint_tensor_count,
                        human_bytes(report.checkpoint_payload_bytes)
                    ),
                ),
                Detail::field("Parameters", human_count(report.logical_parameter_count)),
                Detail::field(
                    "Attention",
                    format!(
                        "{} linear · {} full · {} MTP layers",
                        report.linear_attention_layer_count,
                        report.full_attention_layer_count,
                        report.mtp_layer_count
                    ),
                ),
                Detail::field(
                    "Experts",
                    format!(
                        "{} per layer · {} selected/token",
                        report.experts_per_layer, report.selected_experts_per_token
                    ),
                ),
                Detail::text(
                    "Manifest inspection only; shard headers and tensor payloads were not read.",
                ),
            ]);
        }
        InspectionReport::Hy4(report) => {
            details.extend([
                Detail::field(
                    "Checkpoint",
                    format!(
                        "{} shards · {} tensors · {} payload",
                        report.checkpoint_shard_count,
                        report.checkpoint_tensor_count,
                        human_bytes(report.checkpoint_payload_bytes)
                    ),
                ),
                Detail::field("Parameters", human_count(report.logical_parameter_count)),
                Detail::field(
                    "Layers",
                    format!(
                        "{} dense · {} MoE · {} full indexer",
                        report.dense_layer_count,
                        report.moe_layer_count,
                        report.full_indexer_layer_count
                    ),
                ),
                Detail::field(
                    "Experts",
                    format!(
                        "{} per layer · {} selected/token",
                        report.routed_experts_per_layer, report.selected_experts_per_token
                    ),
                ),
                Detail::field("Exact context ceiling", report.exact_context_ceiling),
                Detail::text("Schema checked from headers; tensor payloads were not loaded."),
            ]);
        }
        InspectionReport::DeepseekV41(report) => {
            details.extend([
                Detail::field(
                    "Checkpoint",
                    format!(
                        "{} shards · {} tensors · {} payload",
                        report.checkpoint_shard_count,
                        report.checkpoint_tensor_count,
                        human_bytes(report.checkpoint_payload_bytes)
                    ),
                ),
                Detail::field(
                    "Parameters",
                    format!(
                        "{} backbone · {} Engram · {} vision · {} DSpark",
                        human_count(report.backbone_logical_parameters),
                        human_count(report.engram_logical_parameters),
                        human_count(report.vision_logical_parameters),
                        human_count(report.dspark_logical_parameters)
                    ),
                ),
                Detail::field(
                    "Layers",
                    format!(
                        "{} encoder · {} decoder · {} KV sources",
                        report.encoder_layer_count,
                        report.decoder_layer_count,
                        report.kv_source_layer_count
                    ),
                ),
                Detail::field("Exact context ceiling", report.exact_context_ceiling),
                Detail::text("Schema checked from headers; tensor payloads were not loaded."),
            ]);
        }
    }
    Entry::new(
        Kind::Success,
        format!("Inspection complete · {:.2}s", elapsed.as_secs_f64()),
        details,
    )
}

pub fn plan_entry(planning: &Planning, elapsed: std::time::Duration) -> Entry {
    use crate::human_bytes;
    let report = &planning.report;
    let mut details = vec![
        Detail::field("Path", planning.model_path.display()),
        Detail::field("Family", planning.model.family),
        Detail::field("Budget result", if report.feasible { "Feasible for these memory estimates" } else { "NOT FEASIBLE" }),
        Detail::field("RAM budget", human_bytes(report.ram_budget_bytes)),
        Detail::field("Context", format!("{} requested · {} conservative ceiling", report.context_tokens, report.maximum_context_under_budget)),
        Detail::field("Resident core", human_bytes(report.resident_core_bytes)),
        Detail::field("KV and state", format!("{} · {} · indexer {}", human_bytes(report.kv_cache_bytes),
            if report.kv_state_bytes == 0 { "native mixed precision".into() } else { format!("{}-byte state", report.kv_state_bytes) },
            if report.kv_includes_configured_indexer { "included" } else { "unavailable" })),
        Detail::field("Scratch", human_bytes(report.scratch_bytes)),
        Detail::field("Safety reserve", human_bytes(report.safety_reserve_bytes)),
        Detail::field("Expert working set", format!("{} (cache plus transient load)", human_bytes(report.expert_cache_budget_bytes))),
        Detail::field("Largest / transient expert", format!("{} / {}", human_bytes(report.maximum_expert_bytes), human_bytes(report.transient_expert_bytes))),
        Detail::field("Expert slots", format!("{} total · {} per sparse layer · {:.1}% capacity coverage", report.expert_slots_total, report.expert_slots_per_sparse_layer, report.expert_capacity_fraction * 100.0)),
        Detail::field("Cold expert traffic", format!("{} per token before cache reuse", human_bytes(report.cold_routed_bytes_per_token))),
        Detail::text("Memory estimates only. Run /preflight with your intended context and expert slots to validate the checkpoint."),
    ];
    if report.unused_ram_after_full_expert_residency > 0 {
        details.push(Detail::field(
            "RAM beyond full expert residency",
            human_bytes(report.unused_ram_after_full_expert_residency),
        ));
    }
    details.extend(report.notes.iter().cloned().map(Detail::text));
    details.extend(planning.warnings.iter().cloned().map(Detail::warning));
    Entry::new(
        if report.feasible && planning.warnings.is_empty() {
            Kind::Success
        } else {
            Kind::Warning
        },
        format!("Plan complete · {:.2}s", elapsed.as_secs_f64()),
        details,
    )
}

pub fn preflight_entry(preflight: &Preflight, elapsed: std::time::Duration) -> Entry {
    use crate::{human_bytes, human_count};
    let mut details = vec![
        Detail::field("Path", preflight.model_path.display()),
        Detail::field("Family", preflight.model.family),
    ];
    match &preflight.report {
        PreflightReport::Glm(report) => details.extend(runtime_details(
            report.context_limit,
            report.exact_context_ceiling,
            report.resident_bytes,
            report.kv_cache_bytes,
            report.maximum_expert_bytes,
            report.transient_expert_bytes,
            report.expert_cache_bytes,
            report.expert_slots_per_layer,
        )),
        PreflightReport::DeepseekV4(report) => {
            details.extend(runtime_details(
                report.context_limit,
                report.exact_context_ceiling,
                report.resident_bytes,
                report.kv_cache_bytes,
                report.maximum_expert_bytes,
                report.transient_expert_bytes,
                report.expert_cache_bytes,
                report.expert_slots_per_layer,
            ));
            details.extend([
                Detail::field(
                    "Schema tensors",
                    format!(
                        "{} required · {} present · {} unexpected",
                        report.required_tensor_count,
                        report.checkpoint_tensor_count,
                        report.unexpected_tensor_count
                    ),
                ),
                Detail::field(
                    "Streamed rows",
                    format!(
                        "embedding {} · LM head {}",
                        human_bytes(report.streamed_embedding_bytes),
                        human_bytes(report.streamed_lm_head_bytes)
                    ),
                ),
                Detail::field(
                    "Architecture",
                    format!(
                        "{} indexed layers · {} DSpark stages",
                        report.indexed_layer_count, report.dspark_stage_count
                    ),
                ),
            ]);
        }
        PreflightReport::DeepseekV41(report) => {
            details.extend(runtime_details(
                report.context_limit,
                report.exact_context_ceiling,
                report.streamed_layer_bytes,
                report.kv_cache_bytes,
                report.maximum_expert_bytes,
                report.transient_expert_bytes,
                report.expert_cache_bytes,
                report.expert_slots_per_layer,
            ));
            details.extend([
                Detail::field(
                    "Schema tensors",
                    format!(
                        "{} required · {} present across {} shards",
                        report.required_tensor_count,
                        report.checkpoint_tensor_count,
                        report.checkpoint_shard_count
                    ),
                ),
                Detail::field(
                    "KV layout",
                    format!(
                        "{} global · {} sliding window",
                        human_bytes(report.global_kv_cache_bytes),
                        human_bytes(report.sliding_window_cache_bytes)
                    ),
                ),
                Detail::field(
                    "Engram",
                    format!(
                        "{} table on storage · {} lookup/token",
                        human_bytes(report.engram_table_bytes),
                        human_bytes(report.engram_lookup_bytes_per_token)
                    ),
                ),
                Detail::text(
                    "Base-text runtime; vision and DSpark tensors are schema-validated only.",
                ),
            ]);
        }
        PreflightReport::Hy4(report) => {
            details.extend(runtime_details(
                report.context_limit,
                report.exact_context_ceiling,
                report.resident_bytes,
                report.kv_cache_bytes,
                report.maximum_expert_bytes,
                report.transient_expert_bytes,
                report.expert_cache_bytes,
                report.expert_slots_per_layer,
            ));
            details.extend([
                Detail::field(
                    "Schema tensors",
                    format!(
                        "{} required · {} present across {} shards",
                        report.required_tensor_count,
                        report.checkpoint_tensor_count,
                        report.checkpoint_shard_count
                    ),
                ),
                Detail::field(
                    "Architecture",
                    format!(
                        "{} full IndexCache layers · {} MTP payload (excluded from base forward)",
                        report.full_indexer_layer_count,
                        human_bytes(report.mtp_payload_bytes)
                    ),
                ),
            ]);
        }
        PreflightReport::Qwen38(report) => {
            details.extend([
                Detail::field("Scope", "Complete text-runtime schema"),
                Detail::field(
                    "Checkpoint",
                    format!(
                        "{} tensors across {} shards · {} payload",
                        report.schema.checkpoint_tensor_count,
                        report.schema.checkpoint_shard_count,
                        human_bytes(report.schema.checkpoint_payload_bytes)
                    ),
                ),
                Detail::field("Context", report.context_limit),
                Detail::field(
                    "Streamed layer peak",
                    human_bytes(report.streamed_layer_bytes),
                ),
                Detail::field(
                    "Recurrent / convolution state",
                    format!(
                        "{} / {}",
                        human_bytes(report.recurrent_state_bytes),
                        human_bytes(report.convolution_state_bytes)
                    ),
                ),
                Detail::field("Full-GQA KV", human_bytes(report.kv_cache_bytes)),
                Detail::field(
                    "Scratch",
                    format!(
                        "{} (includes layer-wise prompt snapshots)",
                        human_bytes(report.scratch_bytes)
                    ),
                ),
                Detail::field(
                    "Experts",
                    format!(
                        "{} each · {} cache slots/layer · {} cache",
                        human_bytes(report.expert_bytes),
                        report.expert_slots_per_layer,
                        human_bytes(report.expert_cache_bytes)
                    ),
                ),
                Detail::field(
                    "Persistent / miss peak",
                    format!(
                        "{} / {} (causal state separate)",
                        human_bytes(report.resident_bytes),
                        human_bytes(report.peak_resident_bytes)
                    ),
                ),
            ]);
        }
        PreflightReport::KimiK3(report) => {
            details.extend([
                Detail::field("Scope", "Complete multimodal checkpoint schema"),
                Detail::field("Schema tensors", format!("{} required · {} present", report.required_tensor_count, report.checkpoint_tensor_count)),
                Detail::field("Decoder", format!("{} tensors · {} KDA / {} MLA · {} dense / {} MoE layers", report.decoder_tensor_count, report.kda_layer_count, report.mla_layer_count, report.dense_layer_count, report.moe_layer_count)),
                Detail::field("Multimodal", format!("{} projector · {} vision tensors", report.projector_tensor_count, report.vision_tensor_count)),
                Detail::field("Parameters", human_count(report.logical_parameter_count)),
                Detail::field("Checkpoint", format!("{} payload across {} shards", human_bytes(report.checkpoint_payload_bytes), report.checkpoint_shard_count)),
                Detail::text("Kimi preflight checks schema only; --context and --expert-slots do not affect it. Use /plan for memory estimates."),
            ]);
        }
        PreflightReport::KimiK3Partial(report) => {
            details.extend([
                Detail::field("Scope", "Partial Kimi-K3 transfer; visible decoder layers only"),
                Detail::field("Validated layer IDs", format!("{:?}", report.validated_layers)),
                Detail::field("Validated tensors", format!("{} decoder · {} visible total", report.validated_tensor_count, report.checkpoint_tensor_count)),
                Detail::field("Non-layer tensors", format!("{} (not validated in partial mode)", report.non_layer_tensor_count)),
                Detail::field("Complete shard files", report.checkpoint_shard_count),
                Detail::warning("Partial validation does not establish checkpoint completeness or runtime readiness. Context and expert slots are not checked."),
            ]);
        }
    }
    details.push(Detail::text("Headers only; tensor payloads were not loaded. No RAM budget or inference execution was tested."));
    let partial = matches!(preflight.report, PreflightReport::KimiK3Partial(_));
    Entry::new(
        if partial {
            Kind::Warning
        } else {
            Kind::Success
        },
        format!(
            "{} · {:.2}s",
            if partial {
                "Partial preflight complete"
            } else {
                "Preflight complete"
            },
            elapsed.as_secs_f64()
        ),
        details,
    )
}

#[allow(clippy::too_many_arguments)]
fn runtime_details(
    context: usize,
    ceiling: usize,
    resident: u64,
    kv: u64,
    maximum_expert: u64,
    transient: u64,
    cache: u64,
    slots: usize,
) -> Vec<Detail> {
    use crate::human_bytes;
    vec![
        Detail::field("Scope", "Runtime schema is load-compatible"),
        Detail::field(
            "Context",
            format!("{context} requested · {ceiling} exact ceiling"),
        ),
        Detail::field("Resident / peak core", human_bytes(resident)),
        Detail::field("KV and state", human_bytes(kv)),
        Detail::field(
            "Largest / transient expert",
            format!(
                "{} / {}",
                human_bytes(maximum_expert),
                human_bytes(transient)
            ),
        ),
        Detail::field(
            "Expert working set",
            format!(
                "{} · {slots} persistent slots per sparse layer",
                human_bytes(cache)
            ),
        ),
    ]
}

pub fn list_entry(listing: &TensorListing, elapsed: std::time::Duration) -> Entry {
    let mut details = vec![
        Detail::field("Path", listing.model_path.display()),
        Detail::field(
            "Filter",
            listing.options.filter.as_deref().unwrap_or("(all tensors)"),
        ),
        Detail::field(
            "Matches",
            format!(
                "{} total · {} returned · limit {}",
                listing.total_matches,
                listing.tensors.len(),
                listing.options.limit
            ),
        ),
    ];
    if listing.tensors.is_empty() {
        details.push(Detail::text(
            "No matching tensors. Try a shorter name fragment or /list without a filter.",
        ));
    } else {
        details.push(Detail::text("Size        Dtype       Shape / tensor name"));
        details.push(Detail::text(
            listing
                .tensors
                .iter()
                .take(128)
                .map(|tensor| {
                    format!(
                        "{:>10}  {:<10} {:?}  {}",
                        crate::human_bytes(tensor.data_len),
                        tensor.dtype,
                        tensor.shape,
                        tensor.name
                    )
                })
                .collect::<Vec<_>>()
                .join("\n"),
        ));
        if listing.tensors.len() > 128 {
            details.push(Detail::text("Display shortened; refine the filter or use urb list in your shell for the full result."));
        }
    }
    Entry::new(
        Kind::Success,
        format!("Tensor list · {:.2}s", elapsed.as_secs_f64()),
        details,
    )
}

pub fn explain_entry(explanation: &Explanation, elapsed: std::time::Duration) -> Entry {
    let mut details = vec![
        Detail::field("Path", explanation.model_path.display()),
        Detail::field("Family", explanation.model.family),
    ];
    details.extend(
        explanation
            .lines
            .iter()
            .map(|line| Detail::text(line.trim_start())),
    );
    Entry::new(
        Kind::Success,
        format!("Model explanation · {:.2}s", elapsed.as_secs_f64()),
        details,
    )
}

pub fn probe_entry(report: &ProbeReport, elapsed: std::time::Duration) -> Entry {
    let mut details = vec![
        Detail::field("Path", report.model_path.display()),
        Detail::field("Tensor", &report.tensor),
        Detail::field(
            "Dtype / shape",
            format!("{} {:?}", report.dtype, report.declared_shape),
        ),
        Detail::field(
            "Sampled storage",
            format!(
                "{} of {} including sidecars",
                crate::human_bytes(report.sampled_storage_bytes),
                crate::human_bytes(
                    report
                        .storage_bytes
                        .saturating_add(report.sidecar_storage_bytes)
                )
            ),
        ),
    ];
    if let Some(shape) = report.logical_matrix_shape {
        details.push(Detail::field("Logical [O,I]", format!("{shape:?}")));
    }
    details.push(Detail::field("Values", stats_text(&report.sampled_values)));
    let mut non_finite = report.sampled_values.non_finite;
    if let Some(quant) = &report.quantization {
        details.push(Detail::field(
            "Quantization",
            format!(
                "{}-bit · {} · group {:?}",
                quant.bits_per_weight, quant.scale_layout, quant.group_size
            ),
        ));
        details.push(Detail::field(
            "Saturation",
            format!("{:.4}%", quant.saturation_fraction * 100.0),
        ));
        if quant.bits_per_weight == 4 {
            details.push(Detail::field(
                "Code histogram",
                format!("{:?}", quant.code_histogram),
            ));
        }
        if let Some(scales) = &quant.scales {
            details.push(Detail::field("Scales", stats_text(scales)));
            non_finite += scales.non_finite;
        }
    }
    details.push(Detail::text(&report.sampling_note));
    if non_finite > 0 {
        details.push(Detail::warning(format!(
            "Found {non_finite} non-finite sampled values/scales."
        )));
    }
    Entry::new(
        if non_finite > 0 {
            Kind::Warning
        } else {
            Kind::Success
        },
        format!("Tensor probe · {:.2}s", elapsed.as_secs_f64()),
        details,
    )
}

fn stats_text(stats: &NumericStats) -> String {
    let number = |value: Option<f64>| {
        value
            .map(|value| format!("{value:.6e}"))
            .unwrap_or_else(|| "n/a".into())
    };
    format!(
        "n={} · range {}..{} · mean {} · std {} · zero {} · nonfinite={}",
        stats.count,
        number(stats.minimum),
        number(stats.maximum),
        number(stats.mean),
        number(stats.standard_deviation),
        stats
            .zero_fraction
            .map(|value| format!("{:.2}%", value * 100.0))
            .unwrap_or_else(|| "n/a".into()),
        stats.non_finite
    )
}

pub fn tokenize_entry(tokenization: &Tokenization, elapsed: std::time::Duration) -> Entry {
    Entry::new(
        Kind::Success,
        format!("Tokenization complete · {:.2}s", elapsed.as_secs_f64()),
        vec![
            Detail::field("Path", tokenization.model_path.display()),
            Detail::field("Family", tokenization.model.family),
            Detail::field("Token count", tokenization.report.token_count),
            Detail::field("Token IDs", token_ids_text(&tokenization.report.token_ids)),
            Detail::field(
                "Prompt",
                if tokenization.report.prompt.is_empty() {
                    "(empty)"
                } else {
                    &tokenization.report.prompt
                },
            ),
        ],
    )
}

pub fn decode_entry(decoding: &Decoding, elapsed: std::time::Duration) -> Entry {
    Entry::new(
        Kind::Success,
        format!("Decoding complete · {:.2}s", elapsed.as_secs_f64()),
        vec![
            Detail::field("Path", decoding.model_path.display()),
            Detail::field("Family", decoding.model.family),
            Detail::field("Token count", decoding.report.token_ids.len()),
            Detail::field(
                "Text",
                if decoding.report.text.is_empty() {
                    "(empty)"
                } else {
                    &decoding.report.text
                },
            ),
            Detail::field("Token IDs", token_ids_text(&decoding.report.token_ids)),
        ],
    )
}

fn token_ids_text(ids: &[u32]) -> String {
    if ids.is_empty() {
        return "(none)".into();
    }
    ids.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn help_entry() -> Entry {
    Entry::new(Kind::Info, "Commands & keys", vec![
        Detail::field("Message", "Plain text generates a reply with previous successful turns. Default: 512 new tokens, automatic RAM on Linux. macOS: set /settings --ram-gib N first."),
        Detail::field("/inspect [MODEL_DIR]", "Inspect and pin model metadata at the upper right. Omit the path to reuse the current model; quote paths with spaces."),
        Detail::field("/plan [MODEL_DIR]", "Memory estimates: --ram-gib N --context N --kv-bytes 2|4. Defaults: detected RAM, 2048 tokens, 4-byte state. macOS requires --ram-gib."),
        Detail::field("/preflight [MODEL_DIR]", "Schema/runtime check: --context N --expert-slots N [--partial]. Defaults: 1 token, 0 cache slots. --partial is Kimi-K3 only."),
        Detail::field("/list [FILTER]", "Find tensors by name fragment: --limit N (default 100; max 100000)."),
        Detail::field("/explain [MODEL_DIR]", "Explain the model architecture and token path using its config."),
        Detail::field("/probe TENSOR_NAME", "Sample one exact tensor: --samples N (default 8192; max 10000000). Reads bounded weight data."),
        Detail::field("/tokenize \"TEXT\"", "Encode raw text, or add --chat [--no-thinking] for a model-native user turn."),
        Detail::field("/decode TOKEN_IDS", "Decode comma-separated IDs; --skip-special removes special tokens."),
        Detail::field("/generate \"TEXT\"", "Reply with explicit options: --ram-gib N --allow-large-model required. --max-new-tokens N (default 1), --threads N, --raw-prompt or --no-thinking. --prompt TEXT also works."),
        Detail::field("/settings", "Show/change chat defaults: --ram-gib N|auto --max-new-tokens N --threads N|auto --thinking | --no-thinking. /generate also remembers its RAM/token/thread/thinking settings."),
        Detail::text("Generation accepts --profile, --profile-json PATH and --profile-trace PATH (not remembered). Esc stops generation. Cancelled/failed replies and raw prompts stay out of chat history. Oldest turns are dropped when context is full. Weights reload for each request."),
        Detail::text("Omit paths to reuse the model. /list, /probe, /tokenize, /decode and /generate accept --model MODEL_DIR. Quote command arguments with spaces; plain messages need no quotes. Use -- before a literal command argument starting with '-'."),
        Detail::text("Example: /plan --ram-gib 32 --context 2048; then /preflight --context 2048 --expert-slots 8 (run separately)."),
        Detail::text("Qwen3.8 uses /preflight for hybrid memory requirements; /plan is unsupported. Kimi preflight validates schema only."),
        Detail::field("/help", "Show this guide"),
        Detail::field("/version", "Show the program version (no model needed)"),
        Detail::field("/clear", "Clear transcript and chat context; keep model and settings. Running output can reappear but will not enter the new context."),
        Detail::field("/quit", "Return to your shell"),
        Detail::field("Enter / Ctrl+J", "Send message or run command / insert newline (Alt+Enter also works)"),
        Detail::field("Tab / Esc", "Complete command / dismiss suggestions; Esc with no suggestions stops generation"),
        Detail::field("F2", "Expand/close Runtime logs (also in narrow terminals). PgUp/PgDn scroll this view; Esc closes it before stopping generation."),
        Detail::field("Up / Down", "History at the first/last input line; Ctrl+P / Ctrl+N always browse history"),
        Detail::field("PgUp / PgDn", "Scroll output; Ctrl+Home / Ctrl+End jump to top / follow latest"),
        Detail::field("Ctrl+C", "Quit and stop generation; Ctrl+D also quits when input is empty"),
        Detail::text("Runtime logs stay in the right panel. The footer shows output/total token counts, decode tok/s and request TTFT (includes loading and prefill). EOS counts as a token; rates need at least two tokens. Metrics remain after completion."),
        Detail::text("Model commands run in the background. /generate loads weights after RAM/runtime checks; /probe reads bounded samples. Use the plain CLI for --json or full output; generation supports --progress and --progress-json on stderr."),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use urbilateria::models::kimi_k3::schema::KimiK3PartialLayers;
    use urbilateria::{CommonModelConfig, ModelFamily};

    #[test]
    fn generation_limits_streams_independently_and_keeps_final_diagnostics() {
        let mut output = GenerationTranscript::default();
        output.append(Stream::Text, &"中文🙂".repeat(10_000));
        output.append(Stream::Text, "\u{1b}\0tail");
        let mut runtime = RuntimeInfo::new(std::path::Path::new("/model"));
        runtime.append(&"verbose profile\n".repeat(1000));
        runtime.append("error: failed");
        let entry = output.entry(1, Kind::Error, "Generation failed".into(), None);
        assert!(entry.details[0].text.contains("Earlier output omitted"));
        assert!(entry.details[0].text.ends_with("tail"));
        assert!(entry.details[0].text.len() < 33_000);
        assert_eq!(entry.details.len(), 1);
        assert!(runtime.text().lines().count() < 67);
        assert!(runtime.text().ends_with("error: failed"));
        assert!(entry
            .details
            .iter()
            .all(|detail| !detail.text.contains(['\u{1b}', '\0'])));
    }

    #[test]
    fn probe_marks_nonfinite_samples_and_sanitizes_tensor_names() {
        let report = ProbeReport {
            model_path: "/model".into(),
            tensor: "tensor\u{1b}[2J".into(),
            dtype: "F32".into(),
            declared_shape: vec![8],
            logical_matrix_shape: None,
            storage_bytes: 32,
            sidecar_storage_bytes: 0,
            sampled_storage_bytes: 8,
            sampled_values: NumericStats {
                count: 2,
                non_finite: 1,
                minimum: Some(0.0),
                maximum: Some(0.0),
                mean: Some(0.0),
                standard_deviation: Some(0.0),
                zero_fraction: Some(1.0),
            },
            quantization: None,
            sampling_note: "Bounded sample, not a full scan".into(),
        };
        let entry = probe_entry(&report, Duration::ZERO);
        assert!(matches!(entry.kind, Kind::Warning));
        assert!(entry
            .details
            .iter()
            .any(|detail| detail.warning && detail.text.contains("non-finite")));
        assert!(entry
            .details
            .iter()
            .all(|detail| !detail.text.contains('\u{1b}')));
        assert!(entry
            .details
            .iter()
            .any(|detail| detail.text.contains("not a full scan")));
    }

    #[test]
    fn partial_preflight_does_not_claim_runtime_readiness_or_validate_unseen_layers() {
        let preflight = Preflight {
            model_path: "/models/kimi".into(),
            model: CommonModelConfig {
                family: ModelFamily::KimiK3,
                model_type: "kimi_k3".into(),
                vocab_size: 1,
                hidden_size: 1,
                num_hidden_layers: 4,
                max_position_embeddings: 1024,
                eos_token_ids: vec![],
            },
            report: PreflightReport::KimiK3Partial(Box::new(KimiK3PartialLayers {
                validated_layers: vec![0, 2],
                validated_tensor_count: 20,
                checkpoint_tensor_count: 21,
                non_layer_tensor_count: 1,
                checkpoint_shard_count: 2,
            })),
        };
        let entry = preflight_entry(&preflight, Duration::from_secs(1));
        assert!(matches!(entry.kind, Kind::Warning));
        assert!(entry.title.starts_with("Partial preflight"));
        assert!(entry.details.iter().any(|detail| detail.text == "[0, 2]"));
        assert!(entry.details.iter().any(|detail| detail.warning
            && detail
                .text
                .contains("does not establish checkpoint completeness")));
        assert!(!entry
            .details
            .iter()
            .any(|detail| detail.text.contains("is load-compatible")));
    }
}
