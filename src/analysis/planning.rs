//! Model-family planning shared by the CLI and interactive frontends.

use super::{
    analyze_checkpoint, build_deepseek_resource_plan, build_deepseek_v41_resource_plan,
    build_hy4_resource_plan, build_kimi_k3_resource_plan, build_resource_plan, ResourcePlan,
};
use crate::models::{deepseek_v4, deepseek_v41, hy4};
use crate::storage::TensorIndex;
use crate::{CommonModelConfig, ModelConfig};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanOptions {
    /// None uses available RAM detection; unsupported hosts must provide an explicit budget.
    pub ram_bytes: Option<u64>,
    pub context: u64,
    pub kv_bytes: u8,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            ram_bytes: None,
            context: 2048,
            kv_bytes: 4,
        }
    }
}

impl PlanOptions {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.context == 0 {
            return Err("--context must be greater than zero");
        }
        if !matches!(self.kv_bytes, 2 | 4) {
            return Err("--kv-bytes must be 2 or 4");
        }
        if self.ram_bytes == Some(0) {
            return Err("RAM budget must be at least one byte; pass --ram-gib N");
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Planning {
    pub model_path: PathBuf,
    pub model: CommonModelConfig,
    pub report: ResourcePlan,
    /// Inspection warnings matter even when the estimated working set fits the budget.
    pub warnings: Vec<String>,
}

pub fn plan_checkpoint(
    model_dir: &Path,
    options: PlanOptions,
) -> Result<Planning, Box<dyn Error + Send + Sync>> {
    options.validate()?;
    let config = ModelConfig::load(model_dir)?;
    let model = config.common();
    if matches!(config, ModelConfig::Qwen38(_)) {
        return Err("the generic `urb plan` report does not represent Qwen3.8 hybrid state; use `urb preflight MODEL_DIR --context N --expert-slots N` for its exact streamed-layer, recurrent, convolution, GQA-KV, scratch, and expert-cache budgets".into());
    }
    let ram = options
        .ram_bytes
        .or_else(detect_available_ram)
        .ok_or("could not detect RAM; pass --ram-gib N (required on macOS)")?;
    let context = options.context;
    let kv_bytes = options.kv_bytes;
    let mut warnings = Vec::new();
    let report = match &config {
        ModelConfig::Glm52(config) => {
            let checkpoint = analyze_checkpoint(model_dir)?;
            let plan = build_resource_plan(config, &checkpoint, ram, context, kv_bytes);
            warnings = checkpoint.warnings;
            plan
        }
        ModelConfig::DeepseekV4(config) => {
            let checkpoint = analyze_checkpoint(model_dir)?;
            let context_usize =
                usize::try_from(context).map_err(|_| "--context does not fit this platform")?;
            let index = TensorIndex::open(model_dir)?;
            let requirements =
                deepseek_v4::schema::inspect_requirements(config, &index, context_usize, 0)?;
            let plan = build_deepseek_resource_plan(
                config,
                &checkpoint,
                &requirements,
                ram,
                context,
                kv_bytes,
            );
            warnings = checkpoint.warnings;
            plan
        }
        ModelConfig::DeepseekV41(config) => {
            let context_usize =
                usize::try_from(context).map_err(|_| "--context does not fit this platform")?;
            let index = TensorIndex::open(model_dir)?;
            let requirements =
                deepseek_v41::schema::inspect_requirements(config, &index, context_usize, 0)?;
            build_deepseek_v41_resource_plan(config, &requirements, ram, context)
        }
        ModelConfig::KimiK3(config) => {
            let checkpoint = analyze_checkpoint(model_dir)?;
            let plan = build_kimi_k3_resource_plan(config, &checkpoint, ram, context, kv_bytes);
            warnings = checkpoint.warnings;
            plan
        }
        ModelConfig::Hy4(config) => {
            let context_usize =
                usize::try_from(context).map_err(|_| "--context does not fit this platform")?;
            let index = TensorIndex::open(model_dir)?;
            let requirements = hy4::schema::inspect_requirements(config, &index, context_usize, 0)?;
            build_hy4_resource_plan(config, &requirements, ram, context, kv_bytes)
        }
        ModelConfig::Qwen38(_) => unreachable!("Qwen3.8 returned before generic analysis"),
    };
    Ok(Planning {
        model_path: model_dir.to_owned(),
        model,
        report,
        warnings,
    })
}

/// Linux host availability capped by an applicable cgroup limit, as used by the CLI runtime.
/// Returns None on platforms without /proc; callers can provide an explicit budget.
pub fn detect_available_ram() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    let host_available = kib.checked_mul(1024)?;
    Some(
        cgroup_memory_remaining()
            .map(|remaining| remaining.min(host_available))
            .unwrap_or(host_available),
    )
}

fn cgroup_memory_remaining() -> Option<u64> {
    let pairs = [
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        (
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        ),
    ];
    for (limit_path, usage_path) in pairs {
        let Ok(limit_text) = fs::read_to_string(limit_path) else {
            continue;
        };
        if limit_text.trim() == "max" {
            continue;
        }
        let Ok(limit) = limit_text.trim().parse::<u64>() else {
            continue;
        };
        let Ok(usage_text) = fs::read_to_string(usage_path) else {
            continue;
        };
        let Ok(usage) = usage_text.trim().parse::<u64>() else {
            continue;
        };
        return Some(limit.saturating_sub(usage));
    }
    None
}
