#pragma once

// Metal GPU compute for flash-expert: pread'd quantized expert weights
// are submitted directly to Metal for dequant+matvec, avoiding CPU dequant.

#include "ggml.h"
#include <cstdint>

// Initialize Metal compute pipeline for flash-expert.
// Compiles shaders, creates persistent command queue and buffers.
// max_dim: max(n_embd, n_ff) — determines buffer sizes.
// max_expert_bytes: largest per-expert data across all layers.
bool flash_expert_metal_init(int64_t max_dim, int64_t max_expert_bytes);

// Compute one expert's FFN: gate_proj → up_proj → SwiGLU → down_proj
// expert_data: raw quantized bytes (pread'd from layer file)
// x: input hidden state [n_embd] F32
// out: output [n_embd] F32 (accumulated: out += weight * result)
// gate_bytes/up_bytes/down_bytes: per-tensor sizes within expert_data
// gate_type/up_type/down_type: GGML quant type IDs
// weight: routing weight for this expert
bool flash_expert_metal_compute(
    const void * expert_data,
    int64_t gate_offset, int64_t gate_bytes, enum ggml_type gate_type,
    int64_t up_offset,   int64_t up_bytes,   enum ggml_type up_type,
    int64_t down_offset, int64_t down_bytes,  enum ggml_type down_type,
    const float * x,
    float * out,
    int64_t n_embd,
    int64_t n_ff,
    float weight);

// Compute shared expert FFN (same as above but no routing weight, always weight=1.0)
// If has_gate: applies sigmoid gate before adding to output
bool flash_expert_metal_compute_shared(
    const void * shared_data,
    int64_t gate_offset, int64_t gate_bytes, enum ggml_type gate_type,
    int64_t up_offset,   int64_t up_bytes,   enum ggml_type up_type,
    int64_t down_offset, int64_t down_bytes,  enum ggml_type down_type,
    const float * gate_inp_data, // sigmoid gate weight [n_embd] F32, or nullptr
    const float * x,
    float * out,
    int64_t n_embd,
    int64_t n_ff);

// Batched compute: process all K experts + shared expert in a single Metal command buffer.
// Much faster than calling flash_expert_metal_compute K times (one commit instead of K+1).
struct FlashExpertEntry {
    const void * data;
    int64_t gate_offset, gate_bytes;
    int64_t up_offset, up_bytes;
    int64_t down_offset, down_bytes;
    enum ggml_type gate_type, up_type, down_type;
    float weight;  // routing weight (1.0 for shared)
};

bool flash_expert_metal_compute_batch(
    const FlashExpertEntry * experts, int n_experts,
    const FlashExpertEntry * shared,  // nullptr if no shared expert
    const float * gate_inp_data,      // sigmoid gate for shared expert, or nullptr
    const float * x,
    float * out,
    int64_t n_embd,
    int64_t n_ff);

void flash_expert_metal_free();
