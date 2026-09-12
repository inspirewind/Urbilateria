//! Bounded slice loading for Hy4's consolidated native-MXFP8 routed experts.
//!
//! The release stores 256 experts in two rank-3 tensors instead of giving every expert its own
//! tensor name. Loading either complete tensor would defeat constrained-memory execution, so this
//! adapter reads exactly one expert's contiguous value and scale ranges.

use super::Hy4Config;
use crate::execution::{install, spawn_compute};
use crate::math::{simulate_e4m3_activation_in_place, MxError, MxFp8Matrix};
use crate::model::{WeightError, WeightMatrix};
use crate::models::deepseek_v4::math::{
    bounded_swiglu, bounded_swiglu_in_place, round_to_bf16_in_place, DeepseekMathError,
};
use crate::profiling::{capture_context, span, ProfileSpan, ProfileStage};
use crate::storage::{DType, ReadBuffer, SafetensorError, TensorIndex, TensorInfo};
use std::fmt;

pub const MX_BLOCK: usize = 32;

#[derive(Debug)]
pub enum Hy4ExpertError {
    Checkpoint(SafetensorError),
    Mx(MxError),
    Weight(WeightError),
    Activation(DeepseekMathError),
    Invalid(String),
    Budget { required: u64, maximum: u64 },
}

impl fmt::Display for Hy4ExpertError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Checkpoint(error) => error.fmt(formatter),
            Self::Mx(error) => error.fmt(formatter),
            Self::Weight(error) => error.fmt(formatter),
            Self::Activation(error) => error.fmt(formatter),
            Self::Invalid(reason) => write!(formatter, "invalid Hy4 routed expert: {reason}"),
            Self::Budget { required, maximum } => write!(
                formatter,
                "Hy4 routed expert needs {required} resident bytes, caller budget is {maximum}"
            ),
        }
    }
}

impl std::error::Error for Hy4ExpertError {}

impl From<SafetensorError> for Hy4ExpertError {
    fn from(value: SafetensorError) -> Self {
        Self::Checkpoint(value)
    }
}

impl From<MxError> for Hy4ExpertError {
    fn from(value: MxError) -> Self {
        Self::Mx(value)
    }
}

impl From<WeightError> for Hy4ExpertError {
    fn from(value: WeightError) -> Self {
        Self::Weight(value)
    }
}

impl From<DeepseekMathError> for Hy4ExpertError {
    fn from(value: DeepseekMathError) -> Self {
        Self::Activation(value)
    }
}

type ExpertForward = Result<Vec<f32>, Hy4ExpertError>;
type ExpertBatchForward = Result<Vec<Vec<f32>>, Hy4ExpertError>;

/// One routed expert extracted from the release's consolidated `[expert, row, column]` tensors.
#[derive(Debug, Clone)]
pub struct Hy4Expert {
    gate_up: WeightMatrix,
    down: WeightMatrix,
    hidden_size: usize,
    intermediate_size: usize,
}

struct ExpertLayout {
    gate_up_name: String,
    gate_up_scale_name: String,
    down_name: String,
    down_scale_name: String,
    hidden: usize,
    intermediate: usize,
    gate_up_rows: usize,
    gate_up_scale_columns: usize,
    down_scale_columns: usize,
}

/// Same-shaped owned storage recovered from an evicted routed expert.
#[derive(Debug)]
pub(crate) struct Hy4ExpertBuffers {
    gate_up_values: ReadBuffer,
    gate_up_scales: ReadBuffer,
    down_values: ReadBuffer,
    down_scales: ReadBuffer,
}

impl Hy4Expert {
    /// Validates all four consolidated tensors and reads only one expert's contiguous slices.
    pub fn load(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
        expert: usize,
        maximum_resident_bytes: u64,
    ) -> Result<Self, Hy4ExpertError> {
        Self::load_with_payload_validation(
            index,
            config,
            layer,
            expert,
            maximum_resident_bytes,
            true,
            false,
        )
    }

    pub(crate) fn load_with_payload_validation(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
        expert: usize,
        maximum_resident_bytes: u64,
        validate_payload: bool,
        direct_io: bool,
    ) -> Result<Self, Hy4ExpertError> {
        let layout = prepare_layout(index, config, layer, expert, maximum_resident_bytes)?;
        let gate_up = load_gate_up(index, expert, &layout, validate_payload, direct_io)?;
        let down = load_down(index, expert, &layout, validate_payload, direct_io)?;
        Ok(finish_expert(layout, gate_up, down))
    }

    /// Loads a missing expert as a two-stage pipeline: once gate/up is resident, its projection
    /// runs while the smaller down projection continues reading from storage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_and_forward_quantized(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
        expert: usize,
        maximum_resident_bytes: u64,
        quantized: &[f32],
        swiglu_limit: f32,
        reuse: Option<Hy4ExpertBuffers>,
        load_profile: ProfileSpan,
        validate_payload: bool,
        direct_io: bool,
    ) -> Result<(Self, ExpertForward), Hy4ExpertError> {
        validate_quantized_input(quantized, config.hidden_size)?;
        let layout = prepare_layout(index, config, layer, expert, maximum_resident_bytes)?;
        let (gate_up_reuse, down_reuse) = match reuse {
            Some(reuse) => (
                Some((reuse.gate_up_values, reuse.gate_up_scales)),
                Some((reuse.down_values, reuse.down_scales)),
            ),
            None => (None, None),
        };
        let gate_up = load_gate_up_reusing(
            index,
            expert,
            &layout,
            gate_up_reuse,
            validate_payload,
            direct_io,
        )?;
        let profile_context = capture_context();
        let _compute_profile = span(ProfileStage::Hy4ExpertCompute);
        if direct_io {
            let gate_up = std::sync::Arc::new(gate_up);
            let compute_gate_up = std::sync::Arc::clone(&gate_up);
            let quantized = quantized.to_vec();
            let compute_context = profile_context.clone();
            let activated = spawn_compute(move || {
                compute_context.enter(|| {
                    activate_gate_up(
                        &compute_gate_up,
                        layout.intermediate,
                        &quantized,
                        swiglu_limit,
                    )
                })
            });
            let down =
                load_down_reusing(index, expert, &layout, down_reuse, validate_payload, true);
            drop(load_profile);
            let activated = activated.join();
            // The compute task has completed and dropped its only cloned owner.
            let gate_up = std::sync::Arc::try_unwrap(gate_up)
                .expect("completed gate/up computation releases its matrix owner");
            // Preserve the original load-before-compute error precedence.
            let down = down?;
            let computed = activated.and_then(|activated| {
                let mut output = down.matvec(&activated)?;
                round_to_bf16_in_place(&mut output)?;
                Ok(output)
            });
            return Ok((finish_expert(layout, gate_up, down), computed));
        }
        let (down, activated) = install(|| {
            rayon::join(
                || {
                    profile_context.enter(|| {
                        let result = load_down_reusing(
                            index,
                            expert,
                            &layout,
                            down_reuse,
                            validate_payload,
                            direct_io,
                        );
                        drop(load_profile);
                        result
                    })
                },
                || {
                    profile_context.enter(|| {
                        activate_gate_up(&gate_up, layout.intermediate, quantized, swiglu_limit)
                    })
                },
            )
        });
        // Preserve the original load-before-compute error precedence.
        let down = down?;
        let computed = activated.and_then(|activated| {
            let mut output = down.matvec(&activated)?;
            round_to_bf16_in_place(&mut output)?;
            Ok(output)
        });
        Ok((finish_expert(layout, gate_up, down), computed))
    }

    /// Loads one expert and applies it to several token activations while its weights are hot.
    /// The gate/up batch runs concurrently with the down-projection read, preserving the existing
    /// storage/compute overlap while decoding each MXFP8 weight block only once per token group.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_and_forward_quantized_batch(
        index: &TensorIndex,
        config: &Hy4Config,
        layer: usize,
        expert: usize,
        maximum_resident_bytes: u64,
        quantized: &[f32],
        batch: usize,
        swiglu_limit: f32,
        reuse: Option<Hy4ExpertBuffers>,
        load_profile: ProfileSpan,
        validate_payload: bool,
        direct_io: bool,
    ) -> Result<(Self, ExpertBatchForward), Hy4ExpertError> {
        validate_quantized_batch(quantized, batch, config.hidden_size)?;
        let layout = prepare_layout(index, config, layer, expert, maximum_resident_bytes)?;
        let (gate_up_reuse, down_reuse) = match reuse {
            Some(reuse) => (
                Some((reuse.gate_up_values, reuse.gate_up_scales)),
                Some((reuse.down_values, reuse.down_scales)),
            ),
            None => (None, None),
        };
        let gate_up = load_gate_up_reusing(
            index,
            expert,
            &layout,
            gate_up_reuse,
            validate_payload,
            direct_io,
        )?;
        let profile_context = capture_context();
        let _compute_profile = span(ProfileStage::Hy4ExpertCompute);
        let (down, activated) = install(|| {
            rayon::join(
                || {
                    profile_context.enter(|| {
                        let result = load_down_reusing(
                            index,
                            expert,
                            &layout,
                            down_reuse,
                            validate_payload,
                            direct_io,
                        );
                        drop(load_profile);
                        result
                    })
                },
                || {
                    profile_context.enter(|| {
                        activate_gate_up_batch(
                            &gate_up,
                            layout.intermediate,
                            quantized,
                            batch,
                            swiglu_limit,
                        )
                    })
                },
            )
        });
        let down = down?;
        let computed = activated.and_then(|activated| {
            let mut output = down.matmul_rows(&activated, batch)?;
            round_to_bf16_in_place(&mut output)?;
            Ok(output
                .chunks_exact(layout.hidden)
                .map(<[f32]>::to_vec)
                .collect())
        });
        Ok((finish_expert(layout, gate_up, down), computed))
    }

    /// Recovers the four native checkpoint buffers once an evicted cache entry has no readers.
    pub(crate) fn into_reusable_buffers(self) -> Option<Hy4ExpertBuffers> {
        let WeightMatrix::MxFp8(gate_up) = self.gate_up else {
            return None;
        };
        let WeightMatrix::MxFp8(down) = self.down else {
            return None;
        };
        let (gate_up_values, gate_up_scales) = gate_up.into_e8m0_buffers()?;
        let (down_values, down_scales) = down.into_e8m0_buffers()?;
        Some(Hy4ExpertBuffers {
            gate_up_values,
            gate_up_scales,
            down_values,
            down_scales,
        })
    }

    pub fn resident_bytes(&self) -> usize {
        self.gate_up
            .resident_bytes()
            .saturating_add(self.down.resident_bytes())
    }

    pub fn gate_row(&self, row: usize) -> Result<Vec<f32>, Hy4ExpertError> {
        if row >= self.intermediate_size {
            return Err(Hy4ExpertError::Invalid(format!(
                "gate row {row} is outside 0..{}",
                self.intermediate_size
            )));
        }
        Ok(self.gate_up.row(row)?)
    }

    pub fn up_row(&self, row: usize) -> Result<Vec<f32>, Hy4ExpertError> {
        if row >= self.intermediate_size {
            return Err(Hy4ExpertError::Invalid(format!(
                "up row {row} is outside 0..{}",
                self.intermediate_size
            )));
        }
        Ok(self.gate_up.row(self.intermediate_size + row)?)
    }

    pub fn down_row(&self, row: usize) -> Result<Vec<f32>, Hy4ExpertError> {
        if row >= self.hidden_size {
            return Err(Hy4ExpertError::Invalid(format!(
                "down row {row} is outside 0..{}",
                self.hidden_size
            )));
        }
        Ok(self.down.row(row)?)
    }

    /// Executes a native ModelOpt-MXFP8 expert from a dynamic E4M3/UE8M0 activation shared by all
    /// routed experts. Both GEMMs materialize BF16 outputs like the release kernels.
    pub(crate) fn forward_quantized(
        &self,
        quantized: &[f32],
        swiglu_limit: f32,
    ) -> Result<Vec<f32>, Hy4ExpertError> {
        validate_quantized_input(quantized, self.hidden_size)?;
        // Gate and up are consecutive halves of one checkpoint matrix. Computing all rows in one
        // invocation preserves every row's reduction order while avoiding a second allocation,
        // Rayon installation, and traversal setup for every routed expert.
        let activated = activate_gate_up(
            &self.gate_up,
            self.intermediate_size,
            quantized,
            swiglu_limit,
        )?;
        let mut output = self.down.matvec(&activated)?;
        round_to_bf16_in_place(&mut output)?;
        Ok(output)
    }

    /// Applies a resident expert to consecutive quantized token activations `[batch, hidden]`.
    pub(crate) fn forward_quantized_batch(
        &self,
        quantized: &[f32],
        batch: usize,
        swiglu_limit: f32,
    ) -> Result<Vec<Vec<f32>>, Hy4ExpertError> {
        validate_quantized_batch(quantized, batch, self.hidden_size)?;
        let activated = activate_gate_up_batch(
            &self.gate_up,
            self.intermediate_size,
            quantized,
            batch,
            swiglu_limit,
        )?;
        let mut output = self.down.matmul_rows(&activated, batch)?;
        round_to_bf16_in_place(&mut output)?;
        Ok(output
            .chunks_exact(self.hidden_size)
            .map(<[f32]>::to_vec)
            .collect())
    }

    /// Pure dequantized-weight probe retained for checkpoint inspection tests.
    pub fn forward_dequantized_reference(
        &self,
        input: &[f32],
        swiglu_limit: f32,
    ) -> Result<Vec<f32>, Hy4ExpertError> {
        if input.len() != self.hidden_size {
            return Err(Hy4ExpertError::Invalid(format!(
                "expert expects {} input values, got {}",
                self.hidden_size,
                input.len()
            )));
        }
        let mut gate = self.gate_up.matvec(input)?;
        let up = gate.split_off(self.intermediate_size);
        let activated = bounded_swiglu(&gate, &up, swiglu_limit)?;
        Ok(self.down.matvec(&activated)?)
    }
}

fn prepare_layout(
    index: &TensorIndex,
    config: &Hy4Config,
    layer: usize,
    expert: usize,
    maximum_resident_bytes: u64,
) -> Result<ExpertLayout, Hy4ExpertError> {
    if layer >= config.num_hidden_layers || !config.layer_is_sparse(layer) {
        return Err(Hy4ExpertError::Invalid(format!(
            "layer {layer} is not one of the {} sparse base layers",
            config.sparse_layer_count()
        )));
    }
    if expert >= config.n_routed_experts {
        return Err(Hy4ExpertError::Invalid(format!(
            "expert {expert} is outside 0..{}",
            config.n_routed_experts
        )));
    }
    let required = resident_bytes(config)?;
    if required > maximum_resident_bytes {
        return Err(Hy4ExpertError::Budget {
            required,
            maximum: maximum_resident_bytes,
        });
    }

    let stem = format!("model.layers.{layer}.mlp.experts");
    let layout = ExpertLayout {
        gate_up_name: format!("{stem}.gate_up_proj"),
        gate_up_scale_name: format!("{stem}.gate_up_proj_scale"),
        down_name: format!("{stem}.down_proj"),
        down_scale_name: format!("{stem}.down_proj_scale"),
        hidden: config.hidden_size,
        intermediate: config.moe_intermediate_size,
        gate_up_rows: config
            .moe_intermediate_size
            .checked_mul(2)
            .ok_or_else(|| Hy4ExpertError::Invalid("gate/up rows overflow".to_owned()))?,
        gate_up_scale_columns: config.hidden_size.div_ceil(MX_BLOCK),
        down_scale_columns: config.moe_intermediate_size.div_ceil(MX_BLOCK),
    };
    validate_rank3(
        index.require(&layout.gate_up_name)?,
        DType::F8E4M3,
        config.n_routed_experts,
        layout.gate_up_rows,
        layout.hidden,
    )?;
    validate_rank3(
        index.require(&layout.gate_up_scale_name)?,
        DType::U8,
        config.n_routed_experts,
        layout.gate_up_rows,
        layout.gate_up_scale_columns,
    )?;
    validate_rank3(
        index.require(&layout.down_name)?,
        DType::F8E4M3,
        config.n_routed_experts,
        layout.hidden,
        layout.intermediate,
    )?;
    validate_rank3(
        index.require(&layout.down_scale_name)?,
        DType::U8,
        config.n_routed_experts,
        layout.hidden,
        layout.down_scale_columns,
    )?;
    Ok(layout)
}

fn load_gate_up(
    index: &TensorIndex,
    expert: usize,
    layout: &ExpertLayout,
    validate_payload: bool,
    direct_io: bool,
) -> Result<WeightMatrix, Hy4ExpertError> {
    load_gate_up_reusing(index, expert, layout, None, validate_payload, direct_io)
}

fn load_gate_up_reusing(
    index: &TensorIndex,
    expert: usize,
    layout: &ExpertLayout,
    reuse: Option<(ReadBuffer, ReadBuffer)>,
    validate_payload: bool,
    direct_io: bool,
) -> Result<WeightMatrix, Hy4ExpertError> {
    let (value_reuse, scale_reuse) = reuse.unzip();
    let values = read_expert_slice_reusing(
        index,
        &layout.gate_up_name,
        expert,
        layout.gate_up_rows,
        layout.hidden,
        value_reuse,
        direct_io,
    )?;
    let scales = read_expert_slice_reusing(
        index,
        &layout.gate_up_scale_name,
        expert,
        layout.gate_up_rows,
        layout.gate_up_scale_columns,
        scale_reuse,
        direct_io,
    )?;
    let matrix = if validate_payload {
        MxFp8Matrix::from_read_buffers(
            layout.gate_up_rows,
            layout.hidden,
            1,
            MX_BLOCK,
            values,
            scales,
        )?
    } else {
        MxFp8Matrix::from_read_buffers_prevalidated(
            layout.gate_up_rows,
            layout.hidden,
            1,
            MX_BLOCK,
            values,
            scales,
        )?
    };
    Ok(WeightMatrix::MxFp8(matrix))
}

fn load_down(
    index: &TensorIndex,
    expert: usize,
    layout: &ExpertLayout,
    validate_payload: bool,
    direct_io: bool,
) -> Result<WeightMatrix, Hy4ExpertError> {
    load_down_reusing(index, expert, layout, None, validate_payload, direct_io)
}

fn load_down_reusing(
    index: &TensorIndex,
    expert: usize,
    layout: &ExpertLayout,
    reuse: Option<(ReadBuffer, ReadBuffer)>,
    validate_payload: bool,
    direct_io: bool,
) -> Result<WeightMatrix, Hy4ExpertError> {
    let (value_reuse, scale_reuse) = reuse.unzip();
    let values = read_expert_slice_reusing(
        index,
        &layout.down_name,
        expert,
        layout.hidden,
        layout.intermediate,
        value_reuse,
        direct_io,
    )?;
    let scales = read_expert_slice_reusing(
        index,
        &layout.down_scale_name,
        expert,
        layout.hidden,
        layout.down_scale_columns,
        scale_reuse,
        direct_io,
    )?;
    let matrix = if validate_payload {
        MxFp8Matrix::from_read_buffers(
            layout.hidden,
            layout.intermediate,
            1,
            MX_BLOCK,
            values,
            scales,
        )?
    } else {
        MxFp8Matrix::from_read_buffers_prevalidated(
            layout.hidden,
            layout.intermediate,
            1,
            MX_BLOCK,
            values,
            scales,
        )?
    };
    Ok(WeightMatrix::MxFp8(matrix))
}

fn finish_expert(layout: ExpertLayout, gate_up: WeightMatrix, down: WeightMatrix) -> Hy4Expert {
    Hy4Expert {
        gate_up,
        down,
        hidden_size: layout.hidden,
        intermediate_size: layout.intermediate,
    }
}

fn validate_quantized_input(quantized: &[f32], hidden_size: usize) -> Result<(), Hy4ExpertError> {
    if quantized.len() != hidden_size || quantized.iter().any(|value| !value.is_finite()) {
        return Err(Hy4ExpertError::Invalid(format!(
            "expert expects {hidden_size} finite quantized input values, got {}",
            quantized.len()
        )));
    }
    Ok(())
}

fn validate_quantized_batch(
    quantized: &[f32],
    batch: usize,
    hidden_size: usize,
) -> Result<(), Hy4ExpertError> {
    let expected = batch.checked_mul(hidden_size).ok_or_else(|| {
        Hy4ExpertError::Invalid("expert batch dimensions overflow usize".to_owned())
    })?;
    if batch == 0 || quantized.len() != expected || quantized.iter().any(|value| !value.is_finite())
    {
        return Err(Hy4ExpertError::Invalid(format!(
            "expert batch expects a non-zero batch and {expected} finite quantized values, got {}",
            quantized.len()
        )));
    }
    Ok(())
}

fn activate_gate_up(
    gate_up: &WeightMatrix,
    intermediate_size: usize,
    quantized: &[f32],
    swiglu_limit: f32,
) -> Result<Vec<f32>, Hy4ExpertError> {
    let mut activated = gate_up.matvec(quantized)?;
    round_to_bf16_in_place(&mut activated)?;
    {
        let (gate, up) = activated.split_at_mut(intermediate_size);
        bounded_swiglu_in_place(gate, up, swiglu_limit)?;
    }
    activated.truncate(intermediate_size);
    simulate_e4m3_activation_in_place(&mut activated, MX_BLOCK)
        .map_err(|error| Hy4ExpertError::Invalid(error.to_string()))?;
    Ok(activated)
}

fn activate_gate_up_batch(
    gate_up: &WeightMatrix,
    intermediate_size: usize,
    quantized: &[f32],
    batch: usize,
    swiglu_limit: f32,
) -> Result<Vec<f32>, Hy4ExpertError> {
    let mut projected = gate_up.matmul_rows(quantized, batch)?;
    round_to_bf16_in_place(&mut projected)?;
    let gate_up_size = intermediate_size.checked_mul(2).ok_or_else(|| {
        Hy4ExpertError::Invalid("gate/up batch dimensions overflow usize".to_owned())
    })?;
    let mut activated = Vec::with_capacity(batch.saturating_mul(intermediate_size));
    for gate_up in projected.chunks_exact_mut(gate_up_size) {
        let (gate, up) = gate_up.split_at_mut(intermediate_size);
        bounded_swiglu_in_place(gate, up, swiglu_limit)?;
        activated.extend_from_slice(gate);
    }
    for token in activated.chunks_exact_mut(intermediate_size) {
        simulate_e4m3_activation_in_place(token, MX_BLOCK)
            .map_err(|error| Hy4ExpertError::Invalid(error.to_string()))?;
    }
    Ok(activated)
}

fn validate_rank3(
    tensor: &TensorInfo,
    dtype: DType,
    experts: usize,
    rows: usize,
    columns: usize,
) -> Result<(), Hy4ExpertError> {
    let expected = [experts as u64, rows as u64, columns as u64];
    let elements = experts
        .checked_mul(rows)
        .and_then(|value| value.checked_mul(columns))
        .ok_or_else(|| Hy4ExpertError::Invalid("consolidated tensor shape overflows".to_owned()))?;
    if tensor.dtype != dtype
        || tensor.shape != expected
        || tensor.declared_elements != elements as u64
        || tensor.data_len != elements as u64
    {
        return Err(Hy4ExpertError::Invalid(format!(
            "tensor {:?} is {} {:?}; expected {dtype} {expected:?}",
            tensor.name, tensor.dtype, tensor.shape
        )));
    }
    Ok(())
}

fn read_expert_slice_reusing(
    index: &TensorIndex,
    name: &str,
    expert: usize,
    rows: usize,
    columns: usize,
    reuse: Option<ReadBuffer>,
    direct_io: bool,
) -> Result<ReadBuffer, Hy4ExpertError> {
    let bytes = rows
        .checked_mul(columns)
        .ok_or_else(|| Hy4ExpertError::Invalid("expert slice size overflows".to_owned()))?;
    let offset = expert
        .checked_mul(bytes)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| Hy4ExpertError::Invalid("expert slice offset overflows".to_owned()))?;
    if direct_io {
        Ok(index.read_range_direct_reusing(name, offset, bytes, reuse)?)
    } else {
        Ok(index.read_range_aligned_reusing(name, offset, bytes, reuse)?)
    }
}

fn resident_bytes(config: &Hy4Config) -> Result<u64, Hy4ExpertError> {
    let hidden = config.hidden_size as u64;
    let intermediate = config.moe_intermediate_size as u64;
    let gate_up_rows = intermediate
        .checked_mul(2)
        .ok_or_else(|| Hy4ExpertError::Invalid("gate/up rows overflow".to_owned()))?;
    let values = gate_up_rows
        .checked_mul(hidden)
        .and_then(|value| value.checked_add(hidden.checked_mul(intermediate)?))
        .ok_or_else(|| Hy4ExpertError::Invalid("expert value count overflows".to_owned()))?;
    let scales = gate_up_rows
        .checked_mul(hidden.div_ceil(MX_BLOCK as u64))
        .and_then(|value| {
            value.checked_add(hidden.checked_mul(intermediate.div_ceil(MX_BLOCK as u64))?)
        })
        .ok_or_else(|| Hy4ExpertError::Invalid("expert scale bytes overflow".to_owned()))?;
    values
        .checked_add(scales)
        .ok_or_else(|| Hy4ExpertError::Invalid("expert resident bytes overflow".to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modelopt_matrix(rows: usize, cols: usize, seed: usize) -> WeightMatrix {
        let codes = [0x00, 0x18, 0x28, 0x38, 0xb0, 0x40, 0x07, 0x87];
        let values = (0..rows * cols)
            .map(|index| codes[(index.wrapping_mul(13) + seed) % codes.len()])
            .collect();
        let scale_cols = cols.div_ceil(MX_BLOCK);
        let scales = (0..rows * scale_cols)
            .map(|index| 124 + ((index + seed) % 7) as u8)
            .collect();
        WeightMatrix::MxFp8(
            MxFp8Matrix::from_packed(rows, cols, 1, MX_BLOCK, values, scales).unwrap(),
        )
    }

    #[test]
    fn batched_expert_is_bit_exact_with_independent_tokens() {
        let hidden = 64;
        let intermediate = 32;
        let expert = Hy4Expert {
            gate_up: modelopt_matrix(intermediate * 2, hidden, 3),
            down: modelopt_matrix(hidden, intermediate, 11),
            hidden_size: hidden,
            intermediate_size: intermediate,
        };
        for batch in 2..=18 {
            let input = (0..batch * hidden)
                .map(|index| ((index.wrapping_mul(7) % 23) as f32 - 11.0) / 16.0)
                .collect::<Vec<_>>();
            let expected = input
                .chunks_exact(hidden)
                .map(|token| expert.forward_quantized(token, 7.0).unwrap())
                .collect::<Vec<_>>();
            let actual = expert.forward_quantized_batch(&input, batch, 7.0).unwrap();
            assert_eq!(
                actual
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}
