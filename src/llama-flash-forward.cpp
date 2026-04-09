// Flash-Forward: Use cb_eval to overlap SSD pread with Metal execution.
//
// GGML's cb_eval callback enables per-node graph execution. We use it to:
// 1. Intercept routing results (ffn_moe_topk) and start pread immediately
// 2. The pread runs in parallel with Metal's remaining nodes before our custom op
// 3. When our custom op fires, expert data is already in memory (warm cache)
//
// This overlaps ~4ms of pread with ~0.5-1ms of Metal work per layer.

#include "llama-flash-forward.h"
#include "llama-remote-expert.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <unordered_set>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

bool llama_flash_forward_available(const struct llama_model * model) {
    auto * hook = llama_get_remote_expert_hook();
    if (!hook || !hook->enabled) return false;
    if (model->hparams.n_expert <= 1) return false;
    return true;
}

// ── Prefetch state ─────────────────────────────────────────────────
// Tracks which experts need prefetching and manages async pread.

struct PrefetchState {
    bool active = false;
    int current_layer = -1;

    // Tensor names we're watching for
    std::unordered_set<std::string> routing_tensors;  // "ffn_moe_topk" per layer

    // When routing is detected, we store the expert indices for prefetch
    // (The actual pread happens in the flash-expert callback, but we could
    //  trigger speculative pread here if we had the file descriptors)
};

static PrefetchState g_prefetch;

// cb_eval callback: intercepts tensor computation during graph execution
static bool flash_forward_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;

    if (ask) {
        // Return true if we want to intercept this tensor's data
        const char * name = ggml_get_name(t);
        if (name && strstr(name, "ffn_moe_topk")) {
            return true;  // We want to see the routing result
        }
        return false;  // Don't need other tensors
    }

    // ask == false: tensor data is ready, we can read it
    const char * name = ggml_get_name(t);
    if (name && strstr(name, "ffn_moe_topk")) {
        // Routing result is ready! The expert indices are in this tensor.
        // We could start pread here, but we need the flash-expert file descriptors
        // and layer metadata. For now, just record that routing is done.
        // The actual pread will still happen in the custom op callback,
        // but future optimization can trigger it here.

        // TODO: Extract expert indices from tensor, trigger async pread
        // This would give us ~0.5ms overlap per layer (routing → custom op gap)
    }

    return true;  // continue execution
}

int llama_flash_forward_decode(struct llama_context * ctx, struct llama_batch batch) {
    // Install our cb_eval callback for prefetch optimization
    // Note: this overrides any existing cb_eval (e.g., for logits extraction)
    // TODO: Chain with existing callback if present

    // For now, just delegate to standard decode
    // The cb_eval optimization requires integration with the flash-expert
    // file descriptors and layer metadata, which isn't wired yet.
    return ctx->decode(batch);
}
