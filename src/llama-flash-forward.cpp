// Flash-Forward: Per-layer graph execution with deferred expert compute.
//
// Instead of executing the entire GGML graph at once (which creates 98 splits
// and forces CPU sync at every expert op), we:
// 1. Build the full graph normally
// 2. Find the MAP_CUSTOM3 (expert) nodes in the graph
// 3. Execute the graph in chunks using ggml_graph_view
// 4. Between chunks: run our flash-expert pipeline with deferred Metal
//
// This eliminates ~50ms of commit+wait overhead per layer and enables
// future pipelining of expert GPU compute with next layer's pread.

#include "llama-flash-forward.h"
#include "llama-remote-expert.h"
#include "llama-flash-expert-metal.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

bool llama_flash_forward_available(const struct llama_model * model) {
    auto * hook = llama_get_remote_expert_hook();
    if (!hook || !hook->enabled) return false;
    if (model->hparams.n_expert <= 1) return false;
    return true;
}

// Find indices of MAP_CUSTOM3 nodes (our expert ops) in the graph
static std::vector<int> find_expert_nodes(struct ggml_cgraph * gf) {
    std::vector<int> indices;
    for (int i = 0; i < gf->n_nodes; i++) {
        if (gf->nodes[i]->op == GGML_OP_MAP_CUSTOM3) {
            indices.push_back(i);
        }
    }
    return indices;
}

// Execute a sub-range of graph nodes [i0, i1) on the given backend scheduler.
// This uses ggml_graph_view to create a view of the sub-range.
static enum ggml_status execute_subgraph(
        ggml_backend_sched_t sched,
        struct ggml_cgraph * gf,
        int i0, int i1,
        bool batched) {
    if (i0 >= i1) return GGML_STATUS_SUCCESS;

    struct ggml_cgraph gv = ggml_graph_view(gf, i0, i1);
    return ggml_backend_sched_graph_compute_async(sched, &gv);
}

int llama_flash_forward_decode(struct llama_context * ctx, struct llama_batch batch) {
    // For now, delegate to standard decode.
    // The per-layer execution requires deeper integration with process_ubatch
    // to intercept after graph build but before graph compute.
    //
    // TODO: Override process_ubatch to use chunked execution:
    //   1. Build graph normally
    //   2. Find expert nodes
    //   3. For each chunk between expert nodes:
    //      a. Execute Metal nodes (attention/routing)
    //      b. Synchronize to read routing results
    //      c. Run flash-expert pipeline (pread + deferred Metal)
    //      d. Write result to the custom op's output tensor
    //      e. Continue to next chunk
    return ctx->decode(batch);
}
