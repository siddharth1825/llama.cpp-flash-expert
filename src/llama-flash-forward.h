#pragma once

// Flash-Forward: Custom per-layer decode for pipeline-parallel MoE inference.
//
// Replaces llama_decode() for models with flash-expert support.
// Instead of building one giant GGML graph for all layers, builds and
// executes small sub-graphs per layer. This enables:
//   - Deferred expert GPU compute (CMD3 overlaps with next layer's IO)
//   - Per-layer pread overlap with GPU compute
//   - Pipeline parallelism between attention and expert FFN
//
// Falls back to standard llama_decode() for non-MoE models or when
// flash-expert is not initialized.

#include "llama.h"

// Custom decode that uses per-layer sub-graph execution.
// Returns 0 on success, non-zero on error (same as llama_decode).
int llama_flash_forward_decode(struct llama_context * ctx, struct llama_batch batch);

// Check if flash-forward is available for this model.
bool llama_flash_forward_available(const struct llama_model * model);
