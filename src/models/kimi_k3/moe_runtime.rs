//! Runtime-backed [`LatentMoeProjector`](super::moe::LatentMoeProjector) for Kimi-K3.
//!
//! Resident dense matrices are borrowed from the decoder layer. Routed expert matrices are
//! acquired from [`KimiK3ExpertStore`] and the current expert is held across its consecutive
//! `w1`, `w3`, and `w2` projections. This matters for a zero-slot expert cache: one routed expert
//! is read once for the complete SiTU MLP instead of once per projection.

use super::expert::{KimiK3Expert, KimiK3ExpertError, KimiK3ExpertStore};
use super::moe::{LatentMoeProjector, MoeProjection};
use crate::model::{WeightError, WeightMatrix};
use std::fmt;
use std::sync::Arc;

#[derive(Debug)]
pub enum KimiK3MoeProjectorError {
    OutputLength {
        layer: usize,
        projection: MoeProjection,
        expected: usize,
        got: usize,
    },
    Matrix {
        layer: usize,
        projection: MoeProjection,
        source: WeightError,
    },
    ExpertAcquire {
        layer: usize,
        expert: usize,
        projection: MoeProjection,
        source: Box<KimiK3ExpertError>,
    },
    ExpertProjection {
        layer: usize,
        expert: usize,
        projection: MoeProjection,
        source: Box<KimiK3ExpertError>,
    },
}

impl fmt::Display for KimiK3MoeProjectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputLength {
                layer,
                projection,
                expected,
                got,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} {projection} produced {expected} values but projector output has length {got}"
            ),
            Self::Matrix {
                layer,
                projection,
                source,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} {projection} matrix projection failed: {source}"
            ),
            Self::ExpertAcquire {
                layer,
                expert,
                projection,
                source,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} {projection} could not acquire expert {expert}: {source}"
            ),
            Self::ExpertProjection {
                layer,
                expert,
                projection,
                source,
            } => write!(
                formatter,
                "Kimi-K3 layer {layer} {projection} failed for expert {expert}: {source}"
            ),
        }
    }
}

impl std::error::Error for KimiK3MoeProjectorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Matrix { source, .. } => Some(source),
            Self::ExpertAcquire { source, .. } | Self::ExpertProjection { source, .. } => {
                Some(source.as_ref())
            }
            Self::OutputLength { .. } => None,
        }
    }
}

/// Borrowed runtime adapter for one Kimi-K3 LatentMoE layer.
#[derive(Debug)]
pub struct KimiK3MoeProjector<'weights, 'store> {
    layer: usize,
    router: &'weights WeightMatrix,
    routed_down: &'weights WeightMatrix,
    routed_up: &'weights WeightMatrix,
    shared_w1: &'weights WeightMatrix,
    shared_w3: &'weights WeightMatrix,
    shared_w2: &'weights WeightMatrix,
    experts: &'store mut KimiK3ExpertStore,
    held_expert: Option<Arc<KimiK3Expert>>,
}

impl<'weights, 'store> KimiK3MoeProjector<'weights, 'store> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer: usize,
        router: &'weights WeightMatrix,
        routed_down: &'weights WeightMatrix,
        routed_up: &'weights WeightMatrix,
        shared_w1: &'weights WeightMatrix,
        shared_w3: &'weights WeightMatrix,
        shared_w2: &'weights WeightMatrix,
        experts: &'store mut KimiK3ExpertStore,
    ) -> Self {
        Self {
            layer,
            router,
            routed_down,
            routed_up,
            shared_w1,
            shared_w3,
            shared_w2,
            experts,
            held_expert: None,
        }
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    /// The transient expert retained for a contiguous routed-expert projection sequence.
    pub fn held_expert(&self) -> Option<usize> {
        self.held_expert.as_ref().map(|expert| expert.expert())
    }

    /// Releases a zero-slot transient early when the caller has completed an expert sequence.
    pub fn release_held_expert(&mut self) {
        self.held_expert = None;
    }

    pub fn expert_store(&self) -> &KimiK3ExpertStore {
        self.experts
    }

    pub fn expert_store_mut(&mut self) -> &mut KimiK3ExpertStore {
        self.experts
    }

    fn acquire_expert(
        &mut self,
        expert: usize,
        projection: MoeProjection,
    ) -> Result<Arc<KimiK3Expert>, KimiK3MoeProjectorError> {
        let matches = self
            .held_expert
            .as_ref()
            .is_some_and(|held| held.layer() == self.layer && held.expert() == expert);
        if !matches {
            // Drop a zero-slot transient before loading the next expert, bounding the live expert
            // payload to one acquisition even when the next checkpoint read fails.
            self.held_expert = None;
            let loaded = self.experts.acquire(self.layer, expert).map_err(|source| {
                KimiK3MoeProjectorError::ExpertAcquire {
                    layer: self.layer,
                    expert,
                    projection,
                    source: Box::new(source),
                }
            })?;
            self.held_expert = Some(loaded);
        }
        Ok(Arc::clone(self.held_expert.as_ref().expect(
            "successful Kimi-K3 expert acquisition is retained",
        )))
    }

    fn project_dense(
        &self,
        projection: MoeProjection,
        matrix: &WeightMatrix,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), KimiK3MoeProjectorError> {
        if output.len() != matrix.rows() {
            return Err(KimiK3MoeProjectorError::OutputLength {
                layer: self.layer,
                projection,
                expected: matrix.rows(),
                got: output.len(),
            });
        }
        let projected = matrix
            .matvec(input)
            .map_err(|source| KimiK3MoeProjectorError::Matrix {
                layer: self.layer,
                projection,
                source,
            })?;
        copy_projection(self.layer, projection, projected, output)
    }

    fn project_expert(
        &mut self,
        projection: MoeProjection,
        expert_id: usize,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), KimiK3MoeProjectorError> {
        let expert = self.acquire_expert(expert_id, projection)?;
        let projected = match projection {
            MoeProjection::RoutedExpertW1 { .. } => expert.project_w1(input),
            MoeProjection::RoutedExpertW3 { .. } => expert.project_w3(input),
            MoeProjection::RoutedExpertW2 { .. } => expert.project_w2(input),
            _ => unreachable!("project_expert receives only routed expert projections"),
        }
        .map_err(|source| KimiK3MoeProjectorError::ExpertProjection {
            layer: self.layer,
            expert: expert_id,
            projection,
            source: Box::new(source),
        })?;
        copy_projection(self.layer, projection, projected, output)
    }
}

impl LatentMoeProjector for KimiK3MoeProjector<'_, '_> {
    type Error = KimiK3MoeProjectorError;

    fn project(
        &mut self,
        projection: MoeProjection,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), Self::Error> {
        match projection {
            MoeProjection::RoutedExpertW1 { expert }
            | MoeProjection::RoutedExpertW3 { expert }
            | MoeProjection::RoutedExpertW2 { expert } => {
                self.project_expert(projection, expert, input, output)
            }
            projection => {
                // A non-expert projection terminates the contiguous w1/w3/w2 sequence.
                self.release_held_expert();
                let matrix = match projection {
                    MoeProjection::Router => self.router,
                    MoeProjection::RoutedDown => self.routed_down,
                    MoeProjection::RoutedUp => self.routed_up,
                    MoeProjection::SharedW1 => self.shared_w1,
                    MoeProjection::SharedW3 => self.shared_w3,
                    MoeProjection::SharedW2 => self.shared_w2,
                    MoeProjection::RoutedExpertW1 { .. }
                    | MoeProjection::RoutedExpertW3 { .. }
                    | MoeProjection::RoutedExpertW2 { .. } => {
                        unreachable!("expert projections were handled above")
                    }
                };
                self.project_dense(projection, matrix, input, output)
            }
        }
    }
}

fn copy_projection(
    layer: usize,
    projection: MoeProjection,
    projected: Vec<f32>,
    output: &mut [f32],
) -> Result<(), KimiK3MoeProjectorError> {
    if output.len() != projected.len() {
        return Err(KimiK3MoeProjectorError::OutputLength {
            layer,
            projection,
            expected: projected.len(),
            got: output.len(),
        });
    }
    output.copy_from_slice(&projected);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DenseMatrix;
    use crate::runtime::ExpertTelemetry;
    use crate::storage::TensorIndex;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    const WIDTH: usize = 32;
    const TINY_EXPERT_BYTES: u64 = 1_632;

    fn dense(multiplier: f32) -> WeightMatrix {
        WeightMatrix::F32(DenseMatrix::new(1, 1, vec![multiplier]).unwrap())
    }

    fn fixture_dir() -> PathBuf {
        crate::test_support::temp_dir("urbilateria_kimi_k3_moe_runtime")
    }

    fn append_tensor(
        tensors: &mut BTreeMap<String, serde_json::Value>,
        payload: &mut Vec<u8>,
        name: String,
        shape: [usize; 2],
        byte: u8,
        count: usize,
    ) {
        let start = payload.len();
        payload.extend(std::iter::repeat_n(byte, count));
        tensors.insert(
            name,
            serde_json::json!({
                "dtype": "U8",
                "shape": shape,
                "data_offsets": [start, payload.len()],
            }),
        );
    }

    fn write_experts(path: &Path, experts: usize) {
        let mut tensors = BTreeMap::new();
        let mut payload = Vec::new();
        for expert in 0..experts {
            let names = super::super::expert::KimiK3ExpertTensorNames::new(0, expert);
            let code = if expert == 0 { 0x11 } else { 0x22 };
            for (packed, scale) in [
                (names.w1_packed, names.w1_scale),
                (names.w2_packed, names.w2_scale),
                (names.w3_packed, names.w3_scale),
            ] {
                append_tensor(
                    &mut tensors,
                    &mut payload,
                    packed,
                    [WIDTH, WIDTH / 2],
                    code,
                    WIDTH * WIDTH / 2,
                );
                append_tensor(
                    &mut tensors,
                    &mut payload,
                    scale,
                    [WIDTH, WIDTH / 32],
                    127,
                    WIDTH * WIDTH / 32,
                );
            }
        }
        let mut header = serde_json::to_vec(&tensors).expect("serialize fixture header");
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(payload);
        fs::write(path, output).expect("write fixture shard");
    }

    struct Harness {
        router: WeightMatrix,
        routed_down: WeightMatrix,
        routed_up: WeightMatrix,
        shared_w1: WeightMatrix,
        shared_w3: WeightMatrix,
        shared_w2: WeightMatrix,
        store: KimiK3ExpertStore,
        dir: PathBuf,
    }

    impl Harness {
        fn new(slots: usize) -> Self {
            let dir = fixture_dir();
            fs::create_dir_all(&dir).unwrap();
            write_experts(&dir.join("tiny.safetensors"), 2);
            let index = TensorIndex::open(&dir).unwrap();
            Self {
                router: dense(2.0),
                routed_down: dense(3.0),
                routed_up: dense(4.0),
                shared_w1: dense(5.0),
                shared_w3: dense(6.0),
                shared_w2: dense(7.0),
                store: KimiK3ExpertStore::new(index, 1, 2, WIDTH, WIDTH, slots, TINY_EXPERT_BYTES)
                    .unwrap(),
                dir,
            }
        }

        fn projector(&mut self) -> KimiK3MoeProjector<'_, '_> {
            KimiK3MoeProjector::new(
                0,
                &self.router,
                &self.routed_down,
                &self.routed_up,
                &self.shared_w1,
                &self.shared_w3,
                &self.shared_w2,
                &mut self.store,
            )
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.dir).expect("remove fixture directory");
        }
    }

    #[test]
    fn every_dense_projection_maps_to_its_declared_matrix() {
        let mut harness = Harness::new(0);
        let mut projector = harness.projector();
        for (projection, expected) in [
            (MoeProjection::Router, 4.0),
            (MoeProjection::RoutedDown, 6.0),
            (MoeProjection::RoutedUp, 8.0),
            (MoeProjection::SharedW1, 10.0),
            (MoeProjection::SharedW3, 12.0),
            (MoeProjection::SharedW2, 14.0),
        ] {
            let mut output = [f32::NAN];
            projector.project(projection, &[2.0], &mut output).unwrap();
            assert_eq!(output, [expected], "wrong mapping for {projection}");
        }
        assert_eq!(projector.held_expert(), None);
    }

    #[test]
    fn zero_slot_w1_w3_w2_sequence_reads_each_expert_only_once() {
        let mut harness = Harness::new(0);
        let mut projector = harness.projector();
        let input = [1.0; WIDTH];
        for (projection, expected) in [
            (MoeProjection::RoutedExpertW1 { expert: 0 }, 16.0),
            (MoeProjection::RoutedExpertW3 { expert: 0 }, 16.0),
            (MoeProjection::RoutedExpertW2 { expert: 0 }, 16.0),
        ] {
            let mut output = [f32::NAN; WIDTH];
            projector.project(projection, &input, &mut output).unwrap();
            assert_eq!(output, [expected; WIDTH]);
        }
        assert_eq!(projector.held_expert(), Some(0));
        assert_eq!(
            projector.expert_store().telemetry(),
            &ExpertTelemetry {
                hits: 0,
                misses: 1,
                evictions: 0,
                bytes_read: TINY_EXPERT_BYTES,
                resident_experts: 0,
                resident_bytes: 0,
            }
        );

        // Switching IDs drops/replaces the held Arc. Expert 1 uses E2M1 code 2 (weight 1.0).
        for projection in [
            MoeProjection::RoutedExpertW1 { expert: 1 },
            MoeProjection::RoutedExpertW3 { expert: 1 },
            MoeProjection::RoutedExpertW2 { expert: 1 },
        ] {
            let mut output = [f32::NAN; WIDTH];
            projector.project(projection, &input, &mut output).unwrap();
            assert_eq!(output, [32.0; WIDTH]);
        }
        assert_eq!(projector.held_expert(), Some(1));
        assert_eq!(projector.expert_store().telemetry().misses, 2);
        assert_eq!(
            projector.expert_store().telemetry().bytes_read,
            2 * TINY_EXPERT_BYTES
        );

        // Any resident projection terminates the contiguous expert sequence.
        let mut output = [f32::NAN];
        projector
            .project(MoeProjection::Router, &[1.0], &mut output)
            .unwrap();
        assert_eq!(projector.held_expert(), None);
    }

    #[test]
    fn output_length_errors_are_contextual_and_never_partially_copy() {
        let mut harness = Harness::new(0);
        let mut projector = harness.projector();
        let mut dense_output = [123.0, 456.0];
        let error = projector
            .project(MoeProjection::SharedW3, &[1.0], &mut dense_output)
            .unwrap_err();
        assert!(matches!(
            error,
            KimiK3MoeProjectorError::OutputLength {
                layer: 0,
                projection: MoeProjection::SharedW3,
                expected: 1,
                got: 2,
            }
        ));
        assert_eq!(dense_output, [123.0, 456.0]);

        let mut expert_output = [123.0; WIDTH - 1];
        let error = projector
            .project(
                MoeProjection::RoutedExpertW1 { expert: 0 },
                &[1.0; WIDTH],
                &mut expert_output,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            KimiK3MoeProjectorError::OutputLength {
                layer: 0,
                projection: MoeProjection::RoutedExpertW1 { expert: 0 },
                expected: WIDTH,
                got
            } if got == WIDTH - 1
        ));
        assert_eq!(expert_output, [123.0; WIDTH - 1]);
    }
}
