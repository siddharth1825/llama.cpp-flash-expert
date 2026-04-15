#pragma once

// Flash-Forward: Pipeline-parallel per-layer execution for MoE inference.
//
// Replaces the scheduler's compute_splits with a custom callback that
// defers expert GPU work, enabling pipeline overlap with the next
// layer's attention computation on the same Metal command queue.

#include "llama.h"
#include "ggml-backend.h"

// Check if flash-forward is available for this model.
bool llama_flash_forward_available(const struct llama_model * model);

// Custom decode (delegates to standard decode for now).
int llama_flash_forward_decode(struct llama_context * ctx, struct llama_batch batch);

// Pipeline-parallel compute: replaces ggml_backend_sched_compute_splits.
// Precondition: alloc_graph has been called.
ggml_status flash_forward_compute(ggml_backend_sched_t sched);
