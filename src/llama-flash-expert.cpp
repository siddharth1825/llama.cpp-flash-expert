// Flash-Expert: SSD streaming for MoE expert weights.
// Reads only K active experts per layer from repacked binary files via pread().
// Computes expert FFN locally using GGML dequant + matmul.
// Hot experts can be pinned in RAM for zero-latency access.

#include "llama-flash-expert.h"
#include "llama-remote-expert.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <unordered_set>
#include <chrono>

#include "llama-flash-expert-metal.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <pthread.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#include <fstream>
#include <sstream>

// ── Per-layer metadata ─────────────────────────────────────────────
// Each layer may have different quant types (Q3_K_M uses mixed quants).
// We detect per-expert-bytes from actual file sizes.

struct LayerInfo {
    int fd;
    int64_t file_size;
    int64_t per_expert_bytes;  // computed from (file_size - shared_bytes) / n_experts
    int64_t shared_offset;     // = n_experts * per_expert_bytes
    int64_t shared_bytes;      // total shared expert data at end of file

    // Per-tensor offsets within one expert block (cumulative)
    std::vector<int64_t> tensor_offsets; // [0, gate_bytes, gate+up_bytes, ...]
    std::vector<int64_t> tensor_sizes;   // [gate_bytes, up_bytes, down_bytes]
};

struct FlashExpertState {
    int n_layers;
    int n_experts;
    int n_experts_used;
    int n_expert_tensors;  // typically 3 (gate, up, down)
    int n_shared_tensors;

    std::vector<LayerInfo> layers;

    // Dimensions (detected on first call)
    int64_t n_embd = 0;
    int64_t n_ff = 0;

    // Quant types per tensor (detected on first call)
    enum ggml_type gate_type = GGML_TYPE_F32;
    enum ggml_type up_type   = GGML_TYPE_F32;
    enum ggml_type down_type = GGML_TYPE_F32;
    bool types_detected = false;

    // Scratch buffers (reused per call)
    static constexpr int MAX_K = 16;      // max experts per token
    std::vector<uint8_t> expert_bufs[MAX_K]; // parallel pread buffers
    std::vector<float>   dequant_buf;    // workspace for dequant+matmul
    std::vector<float>   gate_out;
    std::vector<float>   up_out;
    std::vector<float>   down_out;

    std::vector<uint8_t> shared_buf;

    // Cached shared experts: loaded once at startup, never re-read from SSD
    std::vector<std::vector<uint8_t>> shared_cache; // [n_layers]
    bool shared_cached = false;

    // Hot expert pinning: pinned[layer][expert_idx] → raw data or empty
    // Only populated for experts in the pin list
    std::vector<std::vector<std::vector<uint8_t>>> pinned; // [n_layers][n_experts] (most are empty)
    std::unordered_set<int> pin_set;
    int64_t pinned_bytes = 0;

    // Stats
    double total_ms = 0;
    int64_t total_calls = 0;
    int64_t pin_hits = 0;
    int64_t ssd_reads = 0;

    bool initialized = false;
};

static FlashExpertState g_state;

// ── Minimal JSON parsing ───────────────────────────────────────────

static int64_t json_int(const std::string & json, const std::string & key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0;
    pos = json.find(':', pos);
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r')) pos++;
    return std::stoll(json.substr(pos));
}

static int json_array_count(const std::string & json, const std::string & key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return 0;
    pos = json.find('[', pos);
    auto end = json.find(']', pos);
    std::string arr = json.substr(pos, end - pos);
    int count = 0;
    size_t cur = 0;
    while ((cur = arr.find('{', cur)) != std::string::npos) { count++; cur++; }
    return count;
}

// ── pread helper ───────────────────────────────────────────────────

static bool pread_all(int fd, void * buf, size_t len, off_t offset) {
    char * p = static_cast<char *>(buf);
    while (len > 0) {
        ssize_t n = pread(fd, p, len, offset);
        if (n <= 0) return false;
        p += n;
        len -= n;
        offset += n;
    }
    return true;
}

// ── Dequant + matmul ───────────────────────────────────────────────

// Dequant + matvec: out[i] = dot(W_row_i, x) for i in 0..out_dim
// GGUF stores W as ne1 rows of ne0 elements (ne0 contiguous).
// For gate/up: W is [n_ff rows × n_embd cols], x is [n_embd], out is [n_ff]
// For down:   W is [n_embd rows × n_ff cols], x is [n_ff], out is [n_embd]
static void dequant_matvec(
        const void * quant_w,
        enum ggml_type type,
        const float * x,        // [in_dim]
        float * out,             // [out_dim]
        int64_t out_dim,         // ne1 (number of rows)
        int64_t in_dim,          // ne0 (number of cols, contiguous)
        float * scratch) {       // [out_dim * in_dim]
    const auto * traits = ggml_get_type_traits(type);
    if (traits->to_float) {
        traits->to_float(quant_w, scratch, out_dim * in_dim);
    }

    for (int64_t i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        const float * row = scratch + i * in_dim;
        for (int64_t j = 0; j < in_dim; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

static void swiglu(float * gate, const float * up, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float g = gate[i];
        gate[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// ── Detect quant types from byte sizes ─────────────────────────────

static enum ggml_type detect_quant_type(int64_t bytes, int64_t n_elements) {
    for (int t = GGML_TYPE_Q2_K; t <= GGML_TYPE_Q6_K; t++) {
        const auto * traits = ggml_get_type_traits((enum ggml_type)t);
        if (!traits->type_name) continue;
        int64_t bs = 256; // QK_K for K-quants
        int64_t ts = traits->type_size;
        int64_t expected = (n_elements / bs) * ts;
        if (expected == bytes) return (enum ggml_type)t;
    }
    // Try non-K quants
    for (int t = GGML_TYPE_Q4_0; t <= GGML_TYPE_Q8_1; t++) {
        const auto * traits = ggml_get_type_traits((enum ggml_type)t);
        if (!traits->type_name) continue;
        int64_t expected = (n_elements / traits->blck_size) * traits->type_size;
        if (expected == bytes) return (enum ggml_type)t;
    }
    return GGML_TYPE_F32;
}

// ── Flash expert callback ──────────────────────────────────────────

static bool flash_expert_callback(
        const float * hidden,
        int64_t       n_embd,
        int64_t       n_tokens,
        const int32_t * indices,
        const float * weights,
        int64_t       n_expert_used,
        int           layer,
        float *       output) {

    auto & s = g_state;
    if (!s.initialized || layer < 0 || layer >= s.n_layers) return false;

    auto t0 = std::chrono::high_resolution_clock::now();

    const auto & li = s.layers[layer];

    // Detect dimensions and allocate buffers on first call
    if (!s.types_detected) {
        s.n_embd = n_embd;

        // Detect n_ff from gate tensor bytes: gate is [n_ff, n_embd]
        // Try each K-quant type: gate_bytes = (n_ff * n_embd / 256) * type_size
        int64_t gate_bytes = li.tensor_sizes[0];
        for (int t = GGML_TYPE_Q2_K; t <= GGML_TYPE_Q6_K; t++) {
            const auto * traits = ggml_get_type_traits((enum ggml_type)t);
            if (!traits->type_name || traits->type_size == 0) continue;
            int64_t n_ff_guess = (int64_t)gate_bytes * 256 / ((int64_t)traits->type_size * n_embd);
            if (n_ff_guess > 0 && (n_ff_guess * n_embd / 256) * (int64_t)traits->type_size == gate_bytes) {
                s.n_ff = n_ff_guess;
                break;
            }
        }

        if (s.n_ff == 0) {
            fprintf(stderr, "[flash-expert] Could not detect n_ff from gate_bytes=%lld n_embd=%lld\n",
                (long long)gate_bytes, (long long)n_embd);
            return false;
        }

        fprintf(stderr, "[flash-expert] Detected: n_embd=%lld, n_ff=%lld\n",
            (long long)n_embd, (long long)s.n_ff);

        // Allocate scratch buffers
        int64_t max_dim = std::max(s.n_ff, n_embd);
        s.dequant_buf.resize(max_dim * max_dim);
        s.gate_out.resize(s.n_ff);
        s.up_out.resize(s.n_ff);
        s.down_out.resize(n_embd);

        int64_t max_expert = 0;
        int64_t max_shared = 0;
        for (const auto & l : s.layers) {
            if (l.per_expert_bytes > max_expert) max_expert = l.per_expert_bytes;
            if (l.shared_bytes > max_shared) max_shared = l.shared_bytes;
        }
        for (int k = 0; k < FlashExpertState::MAX_K; k++) {
            s.expert_bufs[k].resize(max_expert);
        }
        s.shared_buf.resize(max_shared);

        s.types_detected = true;
    }

    const int64_t n_ff = s.n_ff;

    // Detect quant types for THIS layer (may differ between layers in mixed quant models)
    enum ggml_type layer_gate_type = detect_quant_type(li.tensor_sizes[0], n_ff * n_embd);
    enum ggml_type layer_up_type   = detect_quant_type(li.tensor_sizes.size() > 1 ? li.tensor_sizes[1] : li.tensor_sizes[0], n_ff * n_embd);
    enum ggml_type layer_down_type = detect_quant_type(li.tensor_sizes.size() > 2 ? li.tensor_sizes[2] : li.tensor_sizes[0], n_embd * n_ff);

    for (int64_t t = 0; t < n_tokens; t++) {
        const float * x = hidden + t * n_embd;
        float * out = output + t * n_embd;
        memset(out, 0, n_embd * sizeof(float));

        // ── Sparse expert FFN ──
        // Phase 1: Parallel pread all K experts from SSD
        const uint8_t * expert_data_ptrs[FlashExpertState::MAX_K] = {};
        int32_t expert_indices[FlashExpertState::MAX_K];
        float   expert_weights[FlashExpertState::MAX_K];
        int     n_to_read = 0;
        int     read_slots[FlashExpertState::MAX_K]; // which slot needs SSD read

        for (int64_t k = 0; k < n_expert_used && k < FlashExpertState::MAX_K; k++) {
            expert_indices[k] = indices[t * n_expert_used + k];
            expert_weights[k] = weights[t * n_expert_used + k];
            if (expert_indices[k] < 0 || expert_indices[k] >= s.n_experts) continue;

            // Check pinned first
            if (!s.pinned.empty() && !s.pinned[layer].empty() &&
                !s.pinned[layer][expert_indices[k]].empty()) {
                expert_data_ptrs[k] = s.pinned[layer][expert_indices[k]].data();
                s.pin_hits++;
            } else {
                read_slots[n_to_read++] = (int)k;
            }
        }

        // Parallel pread using GCD dispatch group
        if (n_to_read > 0) {
#ifdef __APPLE__
            dispatch_group_t group = dispatch_group_create();
            dispatch_queue_t io_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

            for (int r = 0; r < n_to_read; r++) {
                int k = read_slots[r];
                int32_t eidx = expert_indices[k];
                dispatch_group_async(group, io_queue, ^{
                    off_t offset = (off_t)eidx * li.per_expert_bytes;
                    pread_all(li.fd, s.expert_bufs[k].data(), li.per_expert_bytes, offset);
                });
            }
            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
#else
            // Fallback: sequential pread
            for (int r = 0; r < n_to_read; r++) {
                int k = read_slots[r];
                off_t offset = (off_t)expert_indices[k] * li.per_expert_bytes;
                pread_all(li.fd, s.expert_bufs[k].data(), li.per_expert_bytes, offset);
            }
#endif
            for (int r = 0; r < n_to_read; r++) {
                int k = read_slots[r];
                expert_data_ptrs[k] = s.expert_bufs[k].data();
                s.ssd_reads++;
            }
        }

        // Phase 2: Batched Metal compute — all routed experts in ONE command buffer
        // (reduces 8 commit+wait to 1)
        {
            FlashExpertEntry entries[FlashExpertState::MAX_K];
            int n_entries = 0;
            for (int64_t k = 0; k < n_expert_used && k < FlashExpertState::MAX_K; k++) {
                if (!expert_data_ptrs[k]) continue;
                FlashExpertEntry e = {};
                e.data = expert_data_ptrs[k];
                e.gate_offset = li.tensor_offsets[0]; e.gate_bytes = li.tensor_sizes[0];
                e.up_offset = li.tensor_offsets[1]; e.up_bytes = li.tensor_sizes[1];
                e.down_offset = li.tensor_offsets[2]; e.down_bytes = li.tensor_sizes[2];
                e.gate_type = layer_gate_type; e.up_type = layer_up_type; e.down_type = layer_down_type;
                e.weight = expert_weights[k];
                entries[n_entries++] = e;
            }

            // Batch routed experts (one Metal commit)
            flash_expert_metal_compute_batch(
                entries, n_entries,
                nullptr, nullptr,
                x, out, n_embd, n_ff);
        }

        // Shared expert FFN (cached in RAM, separate Metal commit for sigmoid gate)
        if (li.shared_bytes > 0 && s.n_shared_tensors >= 3 &&
            s.shared_cached && !s.shared_cache[layer].empty()) {
            const uint8_t * shared_data = s.shared_cache[layer].data();
            int64_t sh_up_off = li.tensor_sizes[0];
            int64_t sh_down_off = sh_up_off + li.tensor_sizes[1];
            const float * gate_inp_ptr = nullptr;
            if (s.n_shared_tensors >= 4) {
                int64_t gate_inp_off = sh_down_off + li.tensor_sizes[2];
                gate_inp_ptr = reinterpret_cast<const float *>(shared_data + gate_inp_off);
            }
            flash_expert_metal_compute_shared(
                shared_data,
                0, li.tensor_sizes[0], layer_gate_type,
                sh_up_off, li.tensor_sizes[1], layer_up_type,
                sh_down_off, li.tensor_sizes[2], layer_down_type,
                gate_inp_ptr, x, out, n_embd, n_ff);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    s.total_ms += ms;
    s.total_calls++;

    if (s.total_calls % s.n_layers == 0) {
        int64_t token_num = s.total_calls / s.n_layers;
        double tok_ms = s.total_ms / token_num;
        fprintf(stderr, "[flash-expert] token %lld: %.0f ms/tok (%.2f tok/s) | pin_hits=%lld ssd_reads=%lld\n",
            (long long)token_num, tok_ms, 1000.0 / tok_ms,
            (long long)s.pin_hits, (long long)s.ssd_reads);
    }

    return true;
}

// ── Deferred pipeline interface ────────────────────────────────────
// Identical to flash_expert_callback but submits Metal GPU work without
// waiting. The output buffer is NOT populated when this returns.

bool flash_expert_callback_deferred(
        const float * hidden,
        int64_t       n_embd,
        int64_t       n_tokens,
        const int32_t * indices,
        const float * weights,
        int64_t       n_expert_used,
        int           layer,
        float *       output) {

    auto & s = g_state;
    if (!s.initialized || layer < 0 || layer >= s.n_layers) return false;

    const auto & li = s.layers[layer];

    // Ensure types/buffers are detected (they're set on first standard callback)
    if (!s.types_detected) {
        // Fall back to standard (blocking) callback for the first call
        return flash_expert_callback(hidden, n_embd, n_tokens, indices, weights, n_expert_used, layer, output);
    }

    const int64_t n_ff = s.n_ff;
    enum ggml_type layer_gate_type = detect_quant_type(li.tensor_sizes[0], n_ff * n_embd);
    enum ggml_type layer_up_type   = detect_quant_type(li.tensor_sizes.size() > 1 ? li.tensor_sizes[1] : li.tensor_sizes[0], n_ff * n_embd);
    enum ggml_type layer_down_type = detect_quant_type(li.tensor_sizes.size() > 2 ? li.tensor_sizes[2] : li.tensor_sizes[0], n_embd * n_ff);

    for (int64_t t = 0; t < n_tokens; t++) {
        const float * x = hidden + t * n_embd;
        float * out = output + t * n_embd;
        memset(out, 0, n_embd * sizeof(float));

        // Phase 1: pread experts (same as standard callback)
        const uint8_t * expert_data_ptrs[FlashExpertState::MAX_K] = {};
        int32_t expert_indices[FlashExpertState::MAX_K];
        float   expert_weights[FlashExpertState::MAX_K];
        int     n_to_read = 0;
        int     read_slots[FlashExpertState::MAX_K];

        for (int64_t k = 0; k < n_expert_used && k < FlashExpertState::MAX_K; k++) {
            expert_indices[k] = indices[t * n_expert_used + k];
            expert_weights[k] = weights[t * n_expert_used + k];
            if (expert_indices[k] < 0 || expert_indices[k] >= s.n_experts) continue;

            if (!s.pinned.empty() && !s.pinned[layer].empty() &&
                !s.pinned[layer][expert_indices[k]].empty()) {
                expert_data_ptrs[k] = s.pinned[layer][expert_indices[k]].data();
                s.pin_hits++;
            } else {
                read_slots[n_to_read++] = (int)k;
            }
        }

        if (n_to_read > 0) {
#ifdef __APPLE__
            dispatch_group_t group = dispatch_group_create();
            dispatch_queue_t io_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
            for (int r = 0; r < n_to_read; r++) {
                int k = read_slots[r];
                int32_t eidx = expert_indices[k];
                dispatch_group_async(group, io_queue, ^{
                    off_t offset = (off_t)eidx * li.per_expert_bytes;
                    pread_all(li.fd, s.expert_bufs[k].data(), li.per_expert_bytes, offset);
                });
            }
            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
#else
            for (int r = 0; r < n_to_read; r++) {
                int k = read_slots[r];
                off_t offset = (off_t)expert_indices[k] * li.per_expert_bytes;
                pread_all(li.fd, s.expert_bufs[k].data(), li.per_expert_bytes, offset);
            }
#endif
            for (int r = 0; r < n_to_read; r++) {
                int k = read_slots[r];
                expert_data_ptrs[k] = s.expert_bufs[k].data();
                s.ssd_reads++;
            }
        }

        // Phase 2: Build entry arrays for deferred batch
        FlashExpertEntry entries[FlashExpertState::MAX_K];
        int n_entries = 0;
        for (int64_t k = 0; k < n_expert_used && k < FlashExpertState::MAX_K; k++) {
            if (!expert_data_ptrs[k]) continue;
            FlashExpertEntry e = {};
            e.data = expert_data_ptrs[k];
            e.gate_offset = li.tensor_offsets[0]; e.gate_bytes = li.tensor_sizes[0];
            e.up_offset = li.tensor_offsets[1]; e.up_bytes = li.tensor_sizes[1];
            e.down_offset = li.tensor_offsets[2]; e.down_bytes = li.tensor_sizes[2];
            e.gate_type = layer_gate_type; e.up_type = layer_up_type; e.down_type = layer_down_type;
            e.weight = expert_weights[k];
            entries[n_entries++] = e;
        }

        // Build shared expert entry
        FlashExpertEntry shared_entry = {};
        FlashExpertEntry * shared_ptr = nullptr;
        if (li.shared_bytes > 0 && s.n_shared_tensors >= 3 &&
            s.shared_cached && !s.shared_cache[layer].empty()) {
            const uint8_t * shared_data = s.shared_cache[layer].data();
            shared_entry.data = shared_data;
            shared_entry.gate_offset = 0; shared_entry.gate_bytes = li.tensor_sizes[0];
            shared_entry.up_offset = li.tensor_sizes[0]; shared_entry.up_bytes = li.tensor_sizes[1];
            int64_t sh_down_off = li.tensor_sizes[0] + li.tensor_sizes[1];
            shared_entry.down_offset = sh_down_off; shared_entry.down_bytes = li.tensor_sizes[2];
            shared_entry.gate_type = layer_gate_type; shared_entry.up_type = layer_up_type;
            shared_entry.down_type = layer_down_type;
            shared_entry.weight = 1.0f;
            shared_ptr = &shared_entry;
        }

        // Phase 3: Routed expert — DEFERRED
        flash_expert_metal_compute_batch_deferred(
            entries, n_entries, nullptr, x, out, n_embd, n_ff);

        // Phase 4: Shared expert — DEFERRED
        bool has_shared = false;
        if (li.shared_bytes > 0 && s.n_shared_tensors >= 3 &&
            s.shared_cached && !s.shared_cache[layer].empty()) {
            const uint8_t * shared_data = s.shared_cache[layer].data();
            int64_t sh_up_off = li.tensor_sizes[0];
            int64_t sh_down_off = sh_up_off + li.tensor_sizes[1];
            const float * gate_inp_ptr2 = nullptr;
            if (s.n_shared_tensors >= 4) {
                int64_t gate_inp_off = sh_down_off + li.tensor_sizes[2];
                gate_inp_ptr2 = reinterpret_cast<const float *>(shared_data + gate_inp_off);
            }
            flash_expert_metal_compute_shared_deferred(
                shared_data,
                0, li.tensor_sizes[0], layer_gate_type,
                sh_up_off, li.tensor_sizes[1], layer_up_type,
                sh_down_off, li.tensor_sizes[2], layer_down_type,
                gate_inp_ptr2, x, out, n_embd, n_ff);
            has_shared = true;
        }

        // CRITICAL: For n_tokens > 1 (prefill), the next token's commit will
        // overwrite g_buf_out. We MUST collect this token's result before
        // moving to the next iteration. Only the LAST token can be left
        // deferred (collected by pre_copy of the next split).
        if (t + 1 < n_tokens) {
            flash_expert_metal_wait_deferred(out);
            if (has_shared) {
                flash_expert_metal_wait_shared_deferred(out);
            }
        }
    }

    s.total_calls++;
    return true;
}

bool flash_expert_wait_deferred(float * out) {
    return flash_expert_metal_wait_deferred(out);
}

// ── Public API ─────────────────────────────────────────────────────

bool llama_flash_expert_init(const std::string & experts_dir, const std::vector<int> & pin_experts) {
    auto & s = g_state;

    // Read manifest
    std::string manifest_path = experts_dir + "/manifest.json";
    std::ifstream f(manifest_path);
    if (!f.is_open()) {
        fprintf(stderr, "[flash-expert] Failed to open %s\n", manifest_path.c_str());
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    std::string json = ss.str();

    s.n_layers = (int)json_int(json, "n_layers");
    s.n_experts = (int)json_int(json, "n_experts");
    s.n_experts_used = (int)json_int(json, "n_experts_used");
    int64_t manifest_per_expert = json_int(json, "per_expert_bytes");
    int64_t manifest_shared = json_int(json, "shared_bytes");
    s.n_expert_tensors = json_array_count(json, "expert_layout");
    s.n_shared_tensors = json_array_count(json, "shared_layout");

    fprintf(stderr, "[flash-expert] %d layers, %d experts (%d active), %d expert tensors, %d shared tensors\n",
        s.n_layers, s.n_experts, s.n_experts_used, s.n_expert_tensors, s.n_shared_tensors);

    // Open layer files and compute per-layer sizes
    s.layers.resize(s.n_layers);
    for (int i = 0; i < s.n_layers; i++) {
        auto & li = s.layers[i];
        char path[512];
        snprintf(path, sizeof(path), "%s/layer_%03d.bin", experts_dir.c_str(), i);

        li.fd = open(path, O_RDONLY);
        if (li.fd < 0) {
            fprintf(stderr, "[flash-expert] Failed to open %s\n", path);
            return false;
        }

        struct stat st;
        fstat(li.fd, &st);
        li.file_size = st.st_size;

        // Read per-layer metadata (exact tensor sizes, written by llamactl repack)
        char meta_path[512];
        snprintf(meta_path, sizeof(meta_path), "%s/layer_%03d.json", experts_dir.c_str(), i);
        std::ifstream mf(meta_path);
        if (mf.is_open()) {
            std::stringstream mss;
            mss << mf.rdbuf();
            std::string mjson = mss.str();

            li.per_expert_bytes = json_int(mjson, "per_expert_bytes");
            li.shared_bytes = json_int(mjson, "shared_bytes");
            li.shared_offset = (int64_t)s.n_experts * li.per_expert_bytes;

            // Parse expert tensor sizes
            auto epos = mjson.find("\"expert_tensors\"");
            if (epos != std::string::npos) {
                auto arr_start = mjson.find('[', epos);
                auto arr_end = mjson.find(']', arr_start);
                std::string arr = mjson.substr(arr_start, arr_end - arr_start + 1);

                int64_t offset = 0;
                size_t cur = 0;
                while ((cur = arr.find("\"bytes\"", cur)) != std::string::npos) {
                    auto colon = arr.find(':', cur);
                    int64_t bytes = std::stoll(arr.substr(colon + 1));
                    li.tensor_offsets.push_back(offset);
                    li.tensor_sizes.push_back(bytes);
                    offset += bytes;
                    cur = colon + 1;
                }
            }
        } else {
            // Fallback: use manifest values (may be wrong for mixed quant layers)
            li.shared_bytes = manifest_shared;
            li.per_expert_bytes = (li.file_size - li.shared_bytes) / s.n_experts;
            li.shared_offset = (int64_t)s.n_experts * li.per_expert_bytes;
            // Assume equal gate/up, rest is down
            int64_t gate_bytes = li.per_expert_bytes / 3;
            int64_t up_bytes = gate_bytes;
            int64_t down_bytes = li.per_expert_bytes - gate_bytes - up_bytes;
            li.tensor_offsets = {0, gate_bytes, gate_bytes + up_bytes};
            li.tensor_sizes = {gate_bytes, up_bytes, down_bytes};
        }
    }

    fprintf(stderr, "[flash-expert] Layer 0: per_expert=%lld shared=%lld file=%lld\n",
        (long long)s.layers[0].per_expert_bytes, (long long)s.layers[0].shared_bytes,
        (long long)s.layers[0].file_size);

    // Pin hot experts in RAM
    if (!pin_experts.empty()) {
        s.pin_set.insert(pin_experts.begin(), pin_experts.end());
        s.pinned.resize(s.n_layers);

        for (int layer = 0; layer < s.n_layers; layer++) {
            const auto & li = s.layers[layer];
            s.pinned[layer].resize(s.n_experts);

            for (int expert_idx : pin_experts) {
                if (expert_idx < 0 || expert_idx >= s.n_experts) continue;
                s.pinned[layer][expert_idx].resize(li.per_expert_bytes);
                off_t offset = (off_t)expert_idx * li.per_expert_bytes;
                if (!pread_all(li.fd, s.pinned[layer][expert_idx].data(),
                               li.per_expert_bytes, offset)) {
                    fprintf(stderr, "[flash-expert] Failed to pin expert %d layer %d\n", expert_idx, layer);
                    s.pinned[layer][expert_idx].clear();
                    continue;
                }
                s.pinned_bytes += li.per_expert_bytes;
            }
        }
        fprintf(stderr, "[flash-expert] Pinned %zu experts across %d layers: %.1f MB\n",
            pin_experts.size(), s.n_layers, s.pinned_bytes / (1024.0 * 1024.0));
    }

    // Cache shared expert data in RAM (same for every token, ~230MB total)
    {
        s.shared_cache.resize(s.n_layers);
        int64_t shared_total = 0;
        for (int layer = 0; layer < s.n_layers; layer++) {
            const auto & li = s.layers[layer];
            if (li.shared_bytes > 0) {
                s.shared_cache[layer].resize(li.shared_bytes);
                pread_all(li.fd, s.shared_cache[layer].data(), li.shared_bytes, li.shared_offset);
                shared_total += li.shared_bytes;
            }
        }
        s.shared_cached = true;
        fprintf(stderr, "[flash-expert] Cached shared experts: %.1f MB\n", shared_total / (1024.0 * 1024.0));
    }

    // Initialize Metal compute pipeline
    int64_t max_expert = 0;
    int64_t max_dim = 0;
    for (const auto & l : s.layers) {
        if (l.per_expert_bytes > max_expert) max_expert = l.per_expert_bytes;
    }
    // max_dim will be set properly on first call when we know n_embd and n_ff
    // For now use a reasonable upper bound
    max_dim = 8192; // covers models up to 8k hidden dim
    if (!flash_expert_metal_init(max_dim, max_expert)) {
        fprintf(stderr, "[flash-expert] Warning: Metal init failed, falling back to CPU\n");
        // Could fall back to CPU dequant here, but for now just warn
    }

    // Install the hook
    llama_set_remote_expert_hook(flash_expert_callback);

    s.initialized = true;
    fprintf(stderr, "[flash-expert] Ready — experts stream from SSD → Metal GPU, %zu pinned in RAM\n", pin_experts.size());
    return true;
}

void llama_flash_expert_free() {
    auto & s = g_state;
    llama_clear_remote_expert_hook();
    flash_expert_metal_free();

    for (auto & li : s.layers) {
        if (li.fd >= 0) close(li.fd);
    }
    s.layers.clear();
    s.pinned.clear();

    if (s.total_calls > 0) {
        fprintf(stderr, "[flash-expert] Stats: %lld layers computed, %.1f ms avg, pin_hits=%lld ssd_reads=%lld\n",
            (long long)s.total_calls, s.total_ms / s.total_calls,
            (long long)s.pin_hits, (long long)s.ssd_reads);
    }

    s.initialized = false;
}
