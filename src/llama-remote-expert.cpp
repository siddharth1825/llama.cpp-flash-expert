#include "llama-remote-expert.h"
#include "ggml.h"

#include <cstring>

// ── Global hook instance ───────────────────────────────────────────

static llama_remote_expert_hook g_remote_expert_hook;

llama_remote_expert_hook * llama_get_remote_expert_hook() {
    return &g_remote_expert_hook;
}

void llama_set_remote_expert_hook(llama_remote_expert_fn fn) {
    g_remote_expert_hook.callback = std::move(fn);
    g_remote_expert_hook.enabled = true;
}

void llama_clear_remote_expert_hook() {
    g_remote_expert_hook.callback = nullptr;
    g_remote_expert_hook.enabled = false;
}

// ── Custom op for ggml_map_custom3 ─────────────────────────────────
//
// This function is called by GGML's CPU backend during graph execution.
// It reads the input tensors, calls the gRPC hook, and writes the output.
//
// Tensor layout:
//   a (src[0]) = hidden states  [n_embd, n_tokens] F32
//   b (src[1]) = expert indices [n_expert_used, n_tokens] I32
//   c (src[2]) = expert weights [n_expert_used, n_tokens] F32
//   dst        = output         [n_embd, n_tokens] F32
//
// userdata = pointer to int (layer index, heap-allocated, owned by this op)

void llama_remote_expert_custom_op(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b,
        const struct ggml_tensor * c,
        int ith, int nth,
        void * userdata) {
    // Only thread 0 does the work (network call is single-threaded)
    if (ith != 0) return;

    (void)nth;

    const int layer = *static_cast<int *>(userdata);
    const int64_t n_embd        = a->ne[0];
    const int64_t n_tokens      = a->ne[1];
    const int64_t n_expert_used = b->ne[0];

    const float   * hidden  = static_cast<const float *>(a->data);
    const int32_t * indices = static_cast<const int32_t *>(b->data);
    const float   * weights = static_cast<const float *>(c->data);
    float         * output  = static_cast<float *>(dst->data);

    auto * hook = llama_get_remote_expert_hook();
    if (!hook || !hook->enabled || !hook->callback) {
        // Should never happen — the custom op is only inserted when hook is active.
        // Zero output as fallback.
        memset(output, 0, n_embd * n_tokens * sizeof(float));
        return;
    }

    bool ok = hook->callback(hidden, n_embd, n_tokens, indices, weights, n_expert_used, layer, output);
    if (!ok) {
        // gRPC call failed — zero output to avoid garbage
        memset(output, 0, n_embd * n_tokens * sizeof(float));
    }
}

// C API wrappers removed — use C++ API (llama-remote-expert.h) directly.
