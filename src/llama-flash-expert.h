#pragma once

// Flash-Expert: SSD-based expert loading for MoE models.
// Instead of keeping all expert weights in memory, reads only the K active
// experts per layer from repacked per-layer binary files using pread().
//
// Uses the same hook point as llama-remote-expert (build_moe_ffn custom op),
// but computes locally instead of sending over the network.
//
// Repacked file layout (created by `llamactl repack`):
//   packed_experts/layer_XXX.bin
//   Expert i starts at offset: i * per_expert_bytes
//   Within each expert: [gate_data | up_data | down_data]
//   After all experts: [shared_gate | shared_up | shared_down | shared_gate_inp]
//
// manifest.json describes the layout (per_expert_bytes, tensor sizes, etc.)

#include <string>
#include <vector>

// Initialize the flash-expert system.
// Reads manifest.json, opens file descriptors for all layer files,
// pre-allocates scratch buffers, and installs the remote expert hook.
//
// pin_experts: list of expert indices to keep permanently in RAM.
//   These are loaded from SSD at startup and served from memory during inference.
//   Remaining experts are streamed from SSD on demand via pread().
//   Typically the "hot" experts identified by `llamactl analyze`.
//
// Returns true on success.
bool llama_flash_expert_init(const std::string & experts_dir, const std::vector<int> & pin_experts = {});

// Shut down: close fds, free buffers, clear hook.
void llama_flash_expert_free();
