// Metal GPU compute for flash-expert dequant+matvec.
// Maintains a persistent Metal pipeline — no per-call setup overhead.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "llama-flash-expert-metal.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>

// ── Persistent Metal state ─────────────────────────────────────────

static id<MTLDevice>              g_device;
static id<MTLCommandQueue>        g_queue;
static id<MTLComputePipelineState> g_matvec_pipeline;
static id<MTLComputePipelineState> g_swiglu_pipeline;
static id<MTLComputePipelineState> g_weighted_add_pipeline;

// Persistent buffers (reused per call)
static id<MTLBuffer> g_buf_expert;    // quantized expert weights
static id<MTLBuffer> g_buf_x;         // input vector
static id<MTLBuffer> g_buf_gate_out;  // gate projection output
static id<MTLBuffer> g_buf_up_out;    // up projection output
static id<MTLBuffer> g_buf_down_out;  // down projection output (temp per expert)
static id<MTLBuffer> g_buf_out;       // accumulated output
static id<MTLBuffer> g_buf_params;    // kernel parameters

static int64_t g_max_dim = 0;
static int64_t g_max_expert_bytes = 0;
static bool g_initialized = false;

// ── GGML type to our shader type_id ────────────────────────────────

static uint32_t ggml_type_to_shader_id(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K: return 10;
        case GGML_TYPE_Q3_K: return 11;
        case GGML_TYPE_Q4_K: return 12;
        case GGML_TYPE_Q5_K: return 13;
        case GGML_TYPE_Q6_K: return 14;
        default: return 0;
    }
}

// ── Init ───────────────────────────────────────────────────────────

bool flash_expert_metal_init(int64_t max_dim, int64_t max_expert_bytes) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "[flash-expert-metal] No Metal device\n");
            return false;
        }

        g_queue = [g_device newCommandQueue];

        // Load shader library from source
        NSString * shader_path = [NSString stringWithFormat:@"%s/src/llama-flash-expert-metal.metal",
            [[[NSProcessInfo processInfo] environment][@"PWD"] UTF8String] ?: "."];

        // Try loading from the same directory as the binary
        NSString * binary_dir = [[[NSProcessInfo processInfo] arguments][0] stringByDeletingLastPathComponent];
        NSArray * search_paths = @[
            @"src/llama-flash-expert-metal.metal",
            @"llama.cpp/src/llama-flash-expert-metal.metal",
            [binary_dir stringByAppendingPathComponent:@"../../../src/llama-flash-expert-metal.metal"],
            [binary_dir stringByAppendingPathComponent:@"../../src/llama-flash-expert-metal.metal"],
            @"llama-flash-expert-metal.metal",
        ];

        id<MTLLibrary> library = nil;
        NSError * error = nil;

        // First try compiling from source
        for (NSString * path in search_paths) {
            NSString * source = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            if (source) {
                MTLCompileOptions * opts = [[MTLCompileOptions alloc] init];
                opts.fastMathEnabled = YES;
                library = [g_device newLibraryWithSource:source options:opts error:&error];
                if (library) {
                    fprintf(stderr, "[flash-expert-metal] Compiled shaders from %s\n", [path UTF8String]);
                    break;
                }
            }
        }

        if (!library) {
            // Try embedded shader source as fallback
            fprintf(stderr, "[flash-expert-metal] Could not find shader source, trying default library\n");
            library = [g_device newDefaultLibrary];
        }

        if (!library) {
            fprintf(stderr, "[flash-expert-metal] Failed to load Metal shaders: %s\n",
                error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        // Prefer SIMD kernel (32 threads per row with simd_sum reduction)
        id<MTLFunction> matvec_fn = [library newFunctionWithName:@"dequant_matvec_simd"];
        if (!matvec_fn) {
            fprintf(stderr, "[flash-expert-metal] SIMD kernel not found, using scalar\n");
            matvec_fn = [library newFunctionWithName:@"dequant_matvec"];
        }
        id<MTLFunction> swiglu_fn = [library newFunctionWithName:@"swiglu_fused"];
        id<MTLFunction> wadd_fn   = [library newFunctionWithName:@"weighted_add"];

        if (!matvec_fn || !swiglu_fn || !wadd_fn) {
            fprintf(stderr, "[flash-expert-metal] Missing shader functions\n");
            return false;
        }

        g_matvec_pipeline       = [g_device newComputePipelineStateWithFunction:matvec_fn error:&error];
        g_swiglu_pipeline       = [g_device newComputePipelineStateWithFunction:swiglu_fn error:&error];
        g_weighted_add_pipeline = [g_device newComputePipelineStateWithFunction:wadd_fn error:&error];

        if (!g_matvec_pipeline || !g_swiglu_pipeline || !g_weighted_add_pipeline) {
            fprintf(stderr, "[flash-expert-metal] Failed to create pipelines: %s\n",
                [[error localizedDescription] UTF8String]);
            return false;
        }

        // Allocate persistent buffers
        g_max_dim = max_dim;
        g_max_expert_bytes = max_expert_bytes;

        // Allocate enough for 16 experts + 1 shared in one buffer (for batched compute)
        g_buf_expert   = [g_device newBufferWithLength:max_expert_bytes * 17 options:MTLResourceStorageModeShared];
        g_buf_x        = [g_device newBufferWithLength:max_dim * sizeof(float) options:MTLResourceStorageModeShared];
        g_buf_gate_out = [g_device newBufferWithLength:max_dim * sizeof(float) options:MTLResourceStorageModeShared];
        g_buf_up_out   = [g_device newBufferWithLength:max_dim * sizeof(float) options:MTLResourceStorageModeShared];
        g_buf_down_out = [g_device newBufferWithLength:max_dim * sizeof(float) options:MTLResourceStorageModeShared];
        g_buf_out      = [g_device newBufferWithLength:max_dim * sizeof(float) options:MTLResourceStorageModeShared];
        g_buf_params   = [g_device newBufferWithLength:16 * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        fprintf(stderr, "[flash-expert-metal] Initialized: device=%s, max_dim=%lld, expert_buf=%.1f MB\n",
            [[g_device name] UTF8String], (long long)max_dim, max_expert_bytes / (1024.0 * 1024.0));

        g_initialized = true;
        return true;
    }
}

// ── Dispatch helper ────────────────────────────────────────────────

static void dispatch_matvec(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> w_buf, uint64_t w_offset,
        id<MTLBuffer> x_buf,
        id<MTLBuffer> out_buf,
        uint32_t out_dim, uint32_t in_dim, uint32_t type_id) {

    uint32_t params[4] = { out_dim, in_dim, type_id, 0 };

    [enc setComputePipelineState:g_matvec_pipeline];
    [enc setBuffer:w_buf offset:w_offset atIndex:0];
    [enc setBuffer:x_buf offset:0 atIndex:1];
    [enc setBuffer:out_buf offset:0 atIndex:2];
    [enc setBytes:params length:sizeof(params) atIndex:3];

    // SIMD: one threadgroup of 32 threads per output row
    [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void dispatch_swiglu(
        id<MTLComputeCommandEncoder> enc,
        id<MTLBuffer> gate_buf,
        id<MTLBuffer> up_buf,
        uint32_t n) {

    uint32_t params[1] = { n };

    [enc setComputePipelineState:g_swiglu_pipeline];
    [enc setBuffer:gate_buf offset:0 atIndex:0];
    [enc setBuffer:up_buf offset:0 atIndex:1];
    [enc setBytes:params length:sizeof(uint32_t) atIndex:2];

    NSUInteger tpg = g_swiglu_pipeline.maxTotalThreadsPerThreadgroup;
    if (tpg > 256) tpg = 256;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

// ── Public: compute one expert ─────────────────────────────────────

bool flash_expert_metal_compute(
        const void * expert_data,
        int64_t gate_offset, int64_t gate_bytes, enum ggml_type gate_type,
        int64_t up_offset,   int64_t up_bytes,   enum ggml_type up_type,
        int64_t down_offset, int64_t down_bytes,  enum ggml_type down_type,
        const float * x,
        float * out,
        int64_t n_embd,
        int64_t n_ff,
        float weight) {

    if (!g_initialized) return false;

    @autoreleasepool {
        // Copy expert data and input to Metal buffers
        int64_t total_expert = gate_offset + gate_bytes;
        if (up_offset + up_bytes > total_expert) total_expert = up_offset + up_bytes;
        if (down_offset + down_bytes > total_expert) total_expert = down_offset + down_bytes;
        memcpy([g_buf_expert contents], expert_data, total_expert);
        memcpy([g_buf_x contents], x, n_embd * sizeof(float));

        // Copy current output to Metal buffer (for accumulation)
        memcpy([g_buf_out contents], out, n_embd * sizeof(float));

        uint32_t gate_tid = ggml_type_to_shader_id(gate_type);
        uint32_t up_tid   = ggml_type_to_shader_id(up_type);
        uint32_t down_tid = ggml_type_to_shader_id(down_type);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // 1. gate_out = gate_w @ x  [n_ff]
        dispatch_matvec(enc, g_buf_expert, gate_offset, g_buf_x, g_buf_gate_out,
                        (uint32_t)n_ff, (uint32_t)n_embd, gate_tid);

        // 2. up_out = up_w @ x  [n_ff]
        dispatch_matvec(enc, g_buf_expert, up_offset, g_buf_x, g_buf_up_out,
                        (uint32_t)n_ff, (uint32_t)n_embd, up_tid);

        // 3. SwiGLU: gate_out = silu(gate_out) * up_out
        dispatch_swiglu(enc, g_buf_gate_out, g_buf_up_out, (uint32_t)n_ff);

        // 4. down_out = down_w @ gate_out  [n_embd]
        dispatch_matvec(enc, g_buf_expert, down_offset, g_buf_gate_out, g_buf_down_out,
                        (uint32_t)n_embd, (uint32_t)n_ff, down_tid);

        // 5. out += weight * down_out
        float w = weight;
        uint32_t wadd_params[1] = { (uint32_t)n_embd };

        [enc setComputePipelineState:g_weighted_add_pipeline];
        [enc setBuffer:g_buf_out offset:0 atIndex:0];
        [enc setBuffer:g_buf_down_out offset:0 atIndex:1];
        [enc setBytes:&w length:sizeof(float) atIndex:2];
        [enc setBytes:wadd_params length:sizeof(uint32_t) atIndex:3];

        NSUInteger tpg = g_weighted_add_pipeline.maxTotalThreadsPerThreadgroup;
        if (tpg > 256) tpg = 256;
        [enc dispatchThreads:MTLSizeMake(n_embd, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Copy result back
        memcpy(out, [g_buf_out contents], n_embd * sizeof(float));
    }

    return true;
}

// ── Public: compute shared expert ──────────────────────────────────

bool flash_expert_metal_compute_shared(
        const void * shared_data,
        int64_t gate_offset, int64_t gate_bytes, enum ggml_type gate_type,
        int64_t up_offset,   int64_t up_bytes,   enum ggml_type up_type,
        int64_t down_offset, int64_t down_bytes,  enum ggml_type down_type,
        const float * gate_inp_data,
        const float * x,
        float * out,
        int64_t n_embd,
        int64_t n_ff) {

    if (!g_initialized) return false;

    @autoreleasepool {
        int64_t total = down_offset + down_bytes;
        memcpy([g_buf_expert contents], shared_data, total);
        memcpy([g_buf_x contents], x, n_embd * sizeof(float));
        memcpy([g_buf_out contents], out, n_embd * sizeof(float));

        uint32_t gate_tid = ggml_type_to_shader_id(gate_type);
        uint32_t up_tid   = ggml_type_to_shader_id(up_type);
        uint32_t down_tid = ggml_type_to_shader_id(down_type);

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        dispatch_matvec(enc, g_buf_expert, gate_offset, g_buf_x, g_buf_gate_out,
                        (uint32_t)n_ff, (uint32_t)n_embd, gate_tid);
        dispatch_matvec(enc, g_buf_expert, up_offset, g_buf_x, g_buf_up_out,
                        (uint32_t)n_ff, (uint32_t)n_embd, up_tid);
        dispatch_swiglu(enc, g_buf_gate_out, g_buf_up_out, (uint32_t)n_ff);
        dispatch_matvec(enc, g_buf_expert, down_offset, g_buf_gate_out, g_buf_down_out,
                        (uint32_t)n_embd, (uint32_t)n_ff, down_tid);

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Read down_out result
        float * down_result = (float *)[g_buf_down_out contents];

        // Apply sigmoid gate if present
        float gate_val = 1.0f;
        if (gate_inp_data) {
            float dot = 0;
            for (int64_t i = 0; i < n_embd; i++) dot += gate_inp_data[i] * x[i];
            gate_val = 1.0f / (1.0f + expf(-dot));
        }

        // Accumulate: out += gate_val * down_out
        for (int64_t i = 0; i < n_embd; i++) {
            out[i] += gate_val * down_result[i];
        }
    }

    return true;
}

// ── Batched compute: all experts in one command buffer ──────────────

bool flash_expert_metal_compute_batch(
        const FlashExpertEntry * experts, int n_experts,
        const FlashExpertEntry * shared,
        const float * gate_inp_data,
        const float * x,
        float * out,
        int64_t n_embd,
        int64_t n_ff) {

    if (!g_initialized) return false;

    @autoreleasepool {
        // Copy input and zero output accumulator
        memcpy([g_buf_x contents], x, n_embd * sizeof(float));
        memset([g_buf_out contents], 0, n_embd * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // Phase 1: Copy ALL expert data into distinct regions of the Metal buffer (CPU side)
        // This must complete before encoding any GPU dispatches.
        int64_t slot_size = g_max_expert_bytes;  // each expert gets one slot
        for (int e = 0; e < n_experts; e++) {
            int64_t total = experts[e].down_offset + experts[e].down_bytes;
            memcpy((uint8_t *)[g_buf_expert contents] + e * slot_size, experts[e].data, total);
        }
        int shared_slot = n_experts;
        if (shared) {
            int64_t total = shared->down_offset + shared->down_bytes;
            memcpy((uint8_t *)[g_buf_expert contents] + shared_slot * slot_size, shared->data, total);
        }

        // Phase 2: Encode all GPU dispatches into one command encoder
        for (int e = 0; e < n_experts; e++) {
            const auto & exp = experts[e];
            uint64_t base = (uint64_t)e * slot_size;

            uint32_t gate_tid = ggml_type_to_shader_id(exp.gate_type);
            uint32_t up_tid   = ggml_type_to_shader_id(exp.up_type);
            uint32_t down_tid = ggml_type_to_shader_id(exp.down_type);

            dispatch_matvec(enc, g_buf_expert, base + exp.gate_offset, g_buf_x, g_buf_gate_out,
                            (uint32_t)n_ff, (uint32_t)n_embd, gate_tid);
            dispatch_matvec(enc, g_buf_expert, base + exp.up_offset, g_buf_x, g_buf_up_out,
                            (uint32_t)n_ff, (uint32_t)n_embd, up_tid);
            dispatch_swiglu(enc, g_buf_gate_out, g_buf_up_out, (uint32_t)n_ff);
            dispatch_matvec(enc, g_buf_expert, base + exp.down_offset, g_buf_gate_out, g_buf_down_out,
                            (uint32_t)n_embd, (uint32_t)n_ff, down_tid);

            float w = exp.weight;
            uint32_t wadd_p[1] = { (uint32_t)n_embd };
            [enc setComputePipelineState:g_weighted_add_pipeline];
            [enc setBuffer:g_buf_out offset:0 atIndex:0];
            [enc setBuffer:g_buf_down_out offset:0 atIndex:1];
            [enc setBytes:&w length:sizeof(float) atIndex:2];
            [enc setBytes:wadd_p length:sizeof(uint32_t) atIndex:3];
            NSUInteger tpg = g_weighted_add_pipeline.maxTotalThreadsPerThreadgroup;
            if (tpg > 256) tpg = 256;
            [enc dispatchThreads:MTLSizeMake(n_embd,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
        }

        if (shared) {
            uint64_t base = (uint64_t)shared_slot * slot_size;
            uint32_t gate_tid = ggml_type_to_shader_id(shared->gate_type);
            uint32_t up_tid   = ggml_type_to_shader_id(shared->up_type);
            uint32_t down_tid = ggml_type_to_shader_id(shared->down_type);

            dispatch_matvec(enc, g_buf_expert, base + shared->gate_offset, g_buf_x, g_buf_gate_out,
                            (uint32_t)n_ff, (uint32_t)n_embd, gate_tid);
            dispatch_matvec(enc, g_buf_expert, base + shared->up_offset, g_buf_x, g_buf_up_out,
                            (uint32_t)n_ff, (uint32_t)n_embd, up_tid);
            dispatch_swiglu(enc, g_buf_gate_out, g_buf_up_out, (uint32_t)n_ff);
            dispatch_matvec(enc, g_buf_expert, base + shared->down_offset, g_buf_gate_out, g_buf_down_out,
                            (uint32_t)n_embd, (uint32_t)n_ff, down_tid);

            float w = 1.0f;
            uint32_t wadd_p[1] = { (uint32_t)n_embd };
            [enc setComputePipelineState:g_weighted_add_pipeline];
            [enc setBuffer:g_buf_out offset:0 atIndex:0];
            [enc setBuffer:g_buf_down_out offset:0 atIndex:1];
            [enc setBytes:&w length:sizeof(float) atIndex:2];
            [enc setBytes:wadd_p length:sizeof(uint32_t) atIndex:3];
            NSUInteger tpg = g_weighted_add_pipeline.maxTotalThreadsPerThreadgroup;
            if (tpg > 256) tpg = 256;
            [enc dispatchThreads:MTLSizeMake(n_embd,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(out, [g_buf_out contents], n_embd * sizeof(float));
    }

    return true;
}

// ── Deferred batch: commit without wait, overlap with next layer's IO ──

static id<MTLCommandBuffer> g_deferred_cmd = nil;
static int64_t g_deferred_n_embd = 0;

bool flash_expert_metal_compute_batch_deferred(
        const FlashExpertEntry * experts, int n_experts,
        const float * x,
        float * out,
        int64_t n_embd,
        int64_t n_ff) {

    if (!g_initialized) return false;

    // Wait for previous deferred command if any
    if (g_deferred_cmd) {
        [g_deferred_cmd waitUntilCompleted];
        g_deferred_cmd = nil;
    }

    @autoreleasepool {
        memcpy([g_buf_x contents], x, n_embd * sizeof(float));
        memset([g_buf_out contents], 0, n_embd * sizeof(float));

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        int64_t slot_size = g_max_expert_bytes;
        for (int e = 0; e < n_experts; e++) {
            int64_t total = experts[e].down_offset + experts[e].down_bytes;
            memcpy((uint8_t *)[g_buf_expert contents] + e * slot_size, experts[e].data, total);
        }

        for (int e = 0; e < n_experts; e++) {
            const auto & exp = experts[e];
            uint64_t base = (uint64_t)e * slot_size;
            uint32_t gate_tid = ggml_type_to_shader_id(exp.gate_type);
            uint32_t up_tid   = ggml_type_to_shader_id(exp.up_type);
            uint32_t down_tid = ggml_type_to_shader_id(exp.down_type);

            dispatch_matvec(enc, g_buf_expert, base + exp.gate_offset, g_buf_x, g_buf_gate_out,
                            (uint32_t)n_ff, (uint32_t)n_embd, gate_tid);
            dispatch_matvec(enc, g_buf_expert, base + exp.up_offset, g_buf_x, g_buf_up_out,
                            (uint32_t)n_ff, (uint32_t)n_embd, up_tid);
            dispatch_swiglu(enc, g_buf_gate_out, g_buf_up_out, (uint32_t)n_ff);
            dispatch_matvec(enc, g_buf_expert, base + exp.down_offset, g_buf_gate_out, g_buf_down_out,
                            (uint32_t)n_embd, (uint32_t)n_ff, down_tid);

            float w = exp.weight;
            uint32_t wadd_p[1] = { (uint32_t)n_embd };
            [enc setComputePipelineState:g_weighted_add_pipeline];
            [enc setBuffer:g_buf_out offset:0 atIndex:0];
            [enc setBuffer:g_buf_down_out offset:0 atIndex:1];
            [enc setBytes:&w length:sizeof(float) atIndex:2];
            [enc setBytes:wadd_p length:sizeof(uint32_t) atIndex:3];
            NSUInteger tpg = g_weighted_add_pipeline.maxTotalThreadsPerThreadgroup;
            if (tpg > 256) tpg = 256;
            [enc dispatchThreads:MTLSizeMake(n_embd,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
        }

        [enc endEncoding];
        [cmd commit];
        // DON'T wait — return immediately so caller can start next layer's IO

        g_deferred_cmd = cmd;
        g_deferred_n_embd = n_embd;
    }

    return true;
}

// Read back the deferred result. Must call before using `out`.
bool flash_expert_metal_wait_deferred(float * out) {
    if (!g_deferred_cmd) return false;

    [g_deferred_cmd waitUntilCompleted];
    g_deferred_cmd = nil;

    memcpy(out, [g_buf_out contents], g_deferred_n_embd * sizeof(float));
    return true;
}

void flash_expert_metal_free() {
    g_buf_expert   = nil;
    g_buf_x        = nil;
    g_buf_gate_out = nil;
    g_buf_up_out   = nil;
    g_buf_down_out = nil;
    g_buf_out      = nil;
    g_buf_params   = nil;
    g_matvec_pipeline       = nil;
    g_swiglu_pipeline       = nil;
    g_weighted_add_pipeline = nil;
    g_queue  = nil;
    g_device = nil;
    g_initialized = false;
}
