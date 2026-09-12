use std::path::PathBuf;
use urbilateria::models::glm::runtime::GlmRuntimeModel;
use urbilateria::runtime::RuntimeLoadOptions;
use urbilateria::storage::{load_weight_matrices, TensorIndex};
use urbilateria::tokenizer::{
    render_chat, ByteBpeTokenizer, ChatMessage, ChatRole, ChatTemplateOptions,
};

fn checkpoint_dir() -> PathBuf {
    std::env::var_os("URB_GLM52_DIR")
        .map(PathBuf::from)
        .expect("set URB_GLM52_DIR to run the real-checkpoint tests")
}

#[test]
#[ignore = "requires bounded payload reads from the Colibri GLM-5.2 checkpoint"]
fn real_expert_three_projection_batch_loads_exact_geometry() {
    let model_dir = checkpoint_dir();
    let index = TensorIndex::open(&model_dir).unwrap();
    let prefix = "model.layers.3.mlp.experts.0";
    let matrices = load_weight_matrices(
        &index,
        &[
            (&format!("{prefix}.gate_proj.weight"), 2048, 6144),
            (&format!("{prefix}.up_proj.weight"), 2048, 6144),
            (&format!("{prefix}.down_proj.weight"), 6144, 2048),
        ],
        u64::MAX,
    )
    .unwrap();
    assert_eq!(matrices.len(), 3);
    assert_eq!((matrices[0].rows(), matrices[0].cols()), (2048, 6144));
    assert_eq!((matrices[1].rows(), matrices[1].cols()), (2048, 6144));
    assert_eq!((matrices[2].rows(), matrices[2].cols()), (6144, 2048));
    assert!(matrices
        .iter()
        .flat_map(|matrix| matrix.row(0).unwrap())
        .all(f32::is_finite));
}

#[test]
#[ignore = "executes the official no-thinking Hello prompt through the real 78-layer GLM model"]
fn real_checkpoint_chat_prompt_matches_the_colibri_first_token() {
    let model_dir = checkpoint_dir();
    let tokenizer = ByteBpeTokenizer::load(&model_dir).unwrap();
    let rendered = render_chat(
        &[ChatMessage::new(ChatRole::User, "Hello")],
        ChatTemplateOptions {
            enable_thinking: false,
            ..ChatTemplateOptions::default()
        },
    );
    let prompt_tokens = tokenizer.encode(&rendered).unwrap();
    assert_eq!(
        prompt_tokens,
        [154822, 154824, 154827, 9703, 154828, 154841, 154842]
    );

    // Eight slots keeps this opt-in regression usable on a roughly 32 GiB host. Placement and
    // cache capacity cannot change logits; the expected token was independently reproduced by
    // the Colibri engine against this same converted checkpoint.
    let expert_slots = 8;
    let requirements =
        GlmRuntimeModel::inspect_requirements(&model_dir, prompt_tokens.len(), expert_slots)
            .unwrap();
    let model = GlmRuntimeModel::load(
        &model_dir,
        RuntimeLoadOptions {
            resident_budget_bytes: requirements.resident_bytes,
            expert_cache_budget_bytes: requirements.expert_cache_bytes,
            kv_cache_budget_bytes: requirements.kv_cache_bytes,
            expert_slots_per_layer: expert_slots,
            maximum_expert_bytes: requirements.maximum_expert_bytes,
            context_limit: prompt_tokens.len(),
        },
    )
    .unwrap();
    let mut state = model.new_state().unwrap();
    let mut logits = Vec::new();
    for token in prompt_tokens {
        logits = model.forward_token(token, &mut state).unwrap().logits;
    }
    let next = logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0 as u32;
    assert_eq!(next, 9703);
    assert_eq!(tokenizer.decode(&[next], false).unwrap(), "Hello");
}
