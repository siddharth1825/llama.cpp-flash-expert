#pragma once

// Remote expert hook for offloading MoE FFN computation to a remote node.
// When enabled, build_moe_ffn() replaces local expert computation with a
// ggml_map_custom3 node that sends hidden states + expert indices + weights
// over gRPC and receives the combined expert output back.

#include <cstdint>
#include <functional>

struct ggml_tensor;

// Callback signature for remote expert computation.
// Inputs:
//   hidden   - [n_embd, n_tokens] float32 (flattened hidden states)
//   indices  - [n_expert_used, n_tokens] int32 (selected expert indices)
//   weights  - [n_expert_used, n_tokens] float32 (routing weights, already normalized)
//   layer    - layer index
//   output   - pre-allocated [n_embd, n_tokens] float32 buffer to write result into
//
// The callback must:
//   1. Send hidden + indices + weights + layer to the remote expert worker
//   2. The remote worker computes expert FFN + shared expert FFN + combines them
//   3. Write the combined result [n_embd, n_tokens] into output buffer
//
// Returns true on success, false on error.
using llama_remote_expert_fn = std::function<bool(
    const float * hidden,
    int64_t       n_embd,
    int64_t       n_tokens,
    const int32_t * indices,
    const float * weights,
    int64_t       n_expert_used,
    int           layer,
    float *       output)>;

struct llama_remote_expert_hook {
    llama_remote_expert_fn callback;
    bool enabled = false;
};

// Global hook — set before inference, checked during graph construction.
// Thread safety: set once before llama_decode(), do not modify during inference.
llama_remote_expert_hook * llama_get_remote_expert_hook();
void llama_set_remote_expert_hook(llama_remote_expert_fn fn);
void llama_clear_remote_expert_hook();

// Custom op function for ggml_map_custom3 — called during graph execution.
// Do not call directly; it's passed to ggml_map_custom3 in build_moe_ffn.
void llama_remote_expert_custom_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * c,
        int ith, int nth,
        void * userdata);
