//! Opt-in correctness gates for the native DeepSeek-V4.1-Flash release.

use std::path::PathBuf;
use urbilateria::models::deepseek_v41::engram::{EngramTable, NgramHashState};
use urbilateria::models::deepseek_v41::runtime::DeepseekV41RuntimeModel;
use urbilateria::models::deepseek_v41::schema;
use urbilateria::runtime::RuntimeLoadOptions;
use urbilateria::storage::{inspect_weight_matrix, load_weight_matrix, TensorIndex};
use urbilateria::tokenizer::ByteBpeTokenizer;
use urbilateria::{ModelConfig, ModelFamily};

fn checkpoint_dir() -> PathBuf {
    std::env::var_os("URB_DEEPSEEK_V41_DIR")
        .map(PathBuf::from)
        .expect("set URB_DEEPSEEK_V41_DIR to the native DeepSeek-V4.1-Flash directory")
}

#[test]
#[ignore = "requires all 48 native DeepSeek-V4.1-Flash shards; reads headers only"]
fn complete_headers_match_the_exact_release_schema() {
    let directory = checkpoint_dir();
    let detected = ModelConfig::load(&directory).unwrap();
    assert_eq!(detected.family(), ModelFamily::DeepseekV41);
    let ModelConfig::DeepseekV41(config) = detected else {
        unreachable!();
    };
    let index = TensorIndex::open(&directory).unwrap();
    let report = schema::inspect_requirements(&config, &index, 1_048_576, 0).unwrap();
    assert_eq!(report.required_tensor_count, 96_085);
    assert_eq!(report.checkpoint_tensor_count, 96_085);
    assert_eq!(report.checkpoint_shard_count, 48);
    assert_eq!(report.global_kv_cache_bytes, 933_232_640);
    assert_eq!(report.engram_lookup_bytes_per_token, 12_672);
}

#[test]
#[ignore = "reads bounded native 32x32 MXFP8 and packed MXFP4 matrices"]
fn native_trunk_and_expert_matrices_decode_without_conversion() {
    let directory = checkpoint_dir();
    let index = TensorIndex::open(&directory).unwrap();

    let trunk_name = "layers.20.attn.wq_a.weight";
    let trunk_layout = inspect_weight_matrix(&index, trunk_name, 1_280, 5_120).unwrap();
    let trunk = load_weight_matrix(
        &index,
        trunk_name,
        1_280,
        5_120,
        trunk_layout.resident_bytes,
    )
    .unwrap();
    assert!(trunk.row(0).unwrap().iter().all(|value| value.is_finite()));

    let expert_name = "layers.0.ffn.experts.0.w1.weight";
    let expert_layout = inspect_weight_matrix(&index, expert_name, 2_304, 5_120).unwrap();
    let expert = load_weight_matrix(
        &index,
        expert_name,
        2_304,
        5_120,
        expert_layout.resident_bytes,
    )
    .unwrap();
    assert!(expert.row(0).unwrap().iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "builds the exact 129280-entry token map and reads one native Engram row"]
fn native_engram_addressing_and_bounded_row_decode_match_release() {
    let directory = checkpoint_dir();
    let config = ModelConfig::load(&directory)
        .unwrap()
        .as_deepseek_v41()
        .unwrap()
        .clone();
    let tokenizer = ByteBpeTokenizer::load(&directory).unwrap();
    let mut hashes = NgramHashState::new(&config, &tokenizer).unwrap();
    let token_hashes = hashes.push(0, &[0, 3, 100_000], None).unwrap();
    assert_eq!(token_hashes.len(), 3);
    assert_eq!(token_hashes[2].len(), 2);
    assert_eq!(token_hashes[2][0].len(), 24);

    let index = TensorIndex::open(&directory).unwrap();
    let table = EngramTable::open(
        &index,
        1,
        config.text_config.engram_num_embeddings[0],
        config.text_config.engram_head_dim,
    )
    .unwrap();
    assert_eq!(table.row_storage_bytes(), 264);
    let row = table.read_row(token_hashes[2][0][0] as usize).unwrap();
    assert_eq!(row.len(), 256);
    assert!(row.iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "constructs the bounded base runtime after a full native header gate"]
fn real_checkpoint_constructs_the_base_runtime() {
    let directory = checkpoint_dir();
    let config = ModelConfig::load(&directory)
        .unwrap()
        .as_deepseek_v41()
        .unwrap()
        .clone();
    let index = TensorIndex::open(&directory).unwrap();
    let requirements = schema::inspect_requirements(&config, &index, 8, 0).unwrap();
    drop(index);
    let runtime = DeepseekV41RuntimeModel::load(
        &directory,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.streamed_layer_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.kv_cache_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.maximum_expert_bytes,
            context_limit: 8,
        },
    )
    .unwrap();
    assert_eq!(runtime.config().text_config.num_hidden_layers, 40);
    assert_eq!(runtime.new_state().unwrap().position(), 0);
}

#[test]
#[ignore = "executes one complete 40-layer base token on the native checkpoint"]
fn real_checkpoint_executes_one_complete_base_token() {
    let directory = checkpoint_dir();
    let config = ModelConfig::load(&directory)
        .unwrap()
        .as_deepseek_v41()
        .unwrap()
        .clone();
    let index = TensorIndex::open(&directory).unwrap();
    let requirements = schema::inspect_requirements(&config, &index, 1, 0).unwrap();
    drop(index);
    let runtime = DeepseekV41RuntimeModel::load(
        &directory,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.streamed_layer_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.kv_cache_bytes,
            expert_slots_per_layer: 0,
            maximum_expert_bytes: requirements.maximum_expert_bytes,
            context_limit: 1,
        },
    )
    .unwrap();
    let mut state = runtime.new_state().unwrap();
    let step = runtime
        .forward_token(config.bos_token_id, &mut state)
        .unwrap();
    assert_eq!(step.logits.len(), config.text_config.vocab_size);
    assert!(step.logits.iter().all(|value| value.is_finite()));
    assert_eq!(step.routes_by_layer.len(), 40);
    assert!(step.routes_by_layer.iter().all(|routes| routes.len() == 6));
    assert_eq!(state.position(), 1);
    // Generated independently by tools/generate_deepseek_v41_real_logits_oracle.py from the
    // published Python equations. Tensor-core/BLAS reduction order can swap near-tied tail
    // experts, so the gate requires the primary route and at least five of six experts per layer.
    let oracle_routes: [[usize; 6]; 40] = [
        [250, 275, 49, 355, 383, 165],
        [382, 248, 316, 251, 220, 372],
        [222, 323, 217, 248, 292, 325],
        [343, 178, 141, 6, 114, 270],
        [19, 317, 323, 176, 85, 255],
        [226, 95, 150, 55, 98, 278],
        [258, 339, 359, 111, 61, 375],
        [294, 72, 6, 213, 141, 30],
        [237, 84, 213, 142, 241, 58],
        [368, 80, 112, 277, 23, 209],
        [102, 211, 81, 73, 198, 350],
        [165, 240, 79, 108, 335, 167],
        [291, 364, 247, 346, 47, 234],
        [182, 183, 230, 69, 380, 7],
        [199, 169, 88, 203, 278, 18],
        [355, 354, 70, 371, 130, 365],
        [3, 378, 2, 356, 256, 95],
        [311, 237, 297, 18, 238, 378],
        [350, 147, 2, 172, 148, 206],
        [259, 148, 5, 362, 205, 325],
        [180, 190, 141, 234, 86, 58],
        [320, 10, 164, 177, 266, 63],
        [121, 296, 289, 42, 363, 301],
        [268, 90, 54, 239, 189, 16],
        [339, 181, 24, 27, 100, 127],
        [244, 82, 158, 115, 247, 226],
        [170, 157, 277, 361, 118, 11],
        [223, 22, 279, 49, 136, 380],
        [7, 382, 287, 180, 115, 383],
        [243, 330, 260, 167, 91, 31],
        [50, 173, 13, 125, 114, 359],
        [370, 333, 8, 176, 310, 338],
        [355, 132, 66, 46, 146, 137],
        [332, 315, 95, 277, 239, 362],
        [344, 351, 180, 7, 226, 101],
        [316, 364, 169, 159, 308, 368],
        [370, 111, 336, 221, 215, 270],
        [127, 290, 344, 42, 203, 373],
        [67, 383, 295, 239, 173, 100],
        [14, 27, 94, 70, 173, 174],
    ];
    for (layer, (actual, expected)) in step.routes_by_layer.iter().zip(&oracle_routes).enumerate() {
        assert_eq!(actual[0].expert, expected[0], "layer {layer} primary route");
        let overlap = actual
            .iter()
            .filter(|route| expected.contains(&route.expert))
            .count();
        assert!(overlap >= 5, "layer {layer} route overlap {overlap}/6");
    }
    let mut top = step.logits.iter().copied().enumerate().collect::<Vec<_>>();
    top.sort_unstable_by(|(left_id, left), (right_id, right)| {
        right.total_cmp(left).then_with(|| left_id.cmp(right_id))
    });
    let oracle_top = [
        (5, 14.304728),
        (201, 12.950_48),
        (372, 12.339_89),
        (271, 11.292046),
        (30, 11.052801),
        (795, 11.051029),
        (1897, 10.940344),
        (6328, 10.582438),
        (7249, 10.486_05),
        (94, 10.281_94),
        (15, 10.187935),
        (3216, 10.157129),
        (1536, 9.943806),
        (671, 9.828828),
        (17231, 9.616922),
        (13318, 9.616095),
    ];
    let actual_top_ids = top[..16]
        .iter()
        .map(|(id, _)| *id)
        .collect::<std::collections::HashSet<_>>();
    for (token, expected) in oracle_top {
        assert!(
            actual_top_ids.contains(&token),
            "oracle top token {token} is absent"
        );
        let actual = step.logits[token];
        assert!(
            (actual - expected).abs() <= 0.08,
            "token {token}: {actual} versus {expected}"
        );
    }
}
