//! Small, scalar reference implementations of GLM-5.2's important numerical building blocks.
//!
//! These functions favor readability and testability. Optimized SIMD kernels will live beside
//! them later and must remain numerically checked against these references.

mod mxfp;
mod ops;
mod quant;
mod routing;

pub use mxfp::{
    decode_e2m1, decode_e4m3fn, decode_e8m0, encode_e2m1_scalar, encode_e4m3fn_scalar,
    encode_e8m0_scalar, normalized_hadamard, simulate_e2m1_activation, simulate_e4m3_activation,
    simulate_e4m3_activation_in_place, simulate_finegrained_e4m3_activation, MxError, MxFp4Matrix,
    MxFp8Matrix,
};
pub use ops::{interleaved_rope, rms_norm, silu, softmax_in_place, MathError};
pub use quant::{Int4Matrix, Int8Matrix, QuantError};
pub use routing::{route_noaux_tc, RouteChoice, RouteError};
