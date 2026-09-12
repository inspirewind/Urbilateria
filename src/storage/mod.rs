//! Checkpoint storage and bounded random-access readers.

mod load;
mod safetensors;
mod weight;

pub use load::{
    load_i64_matrix_row, load_reference_matrix, load_reference_matrix_row, load_reference_values,
    load_reference_vector, streamed_reference_matvec, TensorLoadError,
};
pub(crate) use safetensors::ReadBuffer;
pub use safetensors::{DType, SafetensorError, ShardInfo, TensorIndex, TensorInfo};
pub(crate) use weight::{
    infer_int4_group_size, infer_unique_packed_bits, validate_quantized_scale_layout,
};
pub use weight::{
    inspect_compact_bf16_matrix, inspect_weight_matrix, load_compact_bf16_matrices,
    load_compact_bf16_matrix, load_weight_matrices, load_weight_matrix, WeightFormat, WeightLayout,
    WeightLoadError,
};
