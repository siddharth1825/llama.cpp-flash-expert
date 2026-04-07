// Expert worker — runs on the big memory device (e.g., M4 Pro 24GB).
// Loads all expert FFN weights from GGUF and serves computation requests over TCP.
// Uses GGML's optimized Metal/CUDA kernels for the actual matmul.
//
// Protocol (binary, little-endian):
//   Request:
//     [4B magic 0x45585054 "EXPT"]
//     [4B layer]
//     [4B n_tokens]
//     [4B n_expert_used]
//     [4B n_embd]
//     [n_embd * n_tokens * 4B] hidden states (F32)
//     [n_expert_used * n_tokens * 4B] expert indices (I32)
//     [n_expert_used * n_tokens * 4B] expert weights (F32)
//   Response:
//     [4B magic 0x52455350 "RESP"]
//     [4B compute_us]
//     [n_embd * n_tokens * 4B] output (F32)

#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef int socklen_t;
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

static const uint32_t MAGIC_REQ  = 0x45585054; // "EXPT"
static const uint32_t MAGIC_RESP = 0x52455350; // "RESP"

// ── Helpers ────────────────────────────────────────────────────────

static bool recv_all(int fd, void * buf, size_t len) {
    char * p = static_cast<char *>(buf);
    while (len > 0) {
        ssize_t n = recv(fd, p, len, 0);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

static bool send_all(int fd, const void * buf, size_t len) {
    const char * p = static_cast<const char *>(buf);
    while (len > 0) {
        ssize_t n = send(fd, p, len, 0);
        if (n <= 0) return false;
        p += n;
        len -= n;
    }
    return true;
}

// ── Persistent compute context ─────────────────────────────────────
// Initialized once at startup, reused for every request.

struct expert_compute_ctx {
    std::vector<ggml_backend_t>             backends;
    std::vector<ggml_backend_buffer_type_t> backend_bufts;

    bool init(const llama_model & /*model*/) {
        // CPU-only backend for the expert worker.
        // The model weights are on CPU (--ngl 0), and we build small per-request
        // graphs that reference them. Using only CPU avoids Metal scheduler issues
        // with ad-hoc graph creation.
        ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!cpu) return false;
        backends.push_back(cpu);
        backend_bufts.push_back(ggml_backend_get_default_buffer_type(cpu));

        fprintf(stderr, "[expert-worker] Backends: CPU\n");
        return true;
    }

    ~expert_compute_ctx() {
        for (auto * b : backends) {
            ggml_backend_free(b);
        }
    }
};

// ── Expert FFN computation using GGML ──────────────────────────────

static bool compute_expert_ffn(
        expert_compute_ctx & compute,
        const llama_model & model,
        int layer,
        const float * hidden_data,
        int64_t n_embd,
        int64_t n_tokens,
        const int32_t * indices_data,
        const float * weights_data,
        int64_t n_expert_used,
        float * output_data) {

    const auto & layer_data = model.layers[layer];

    const size_t max_nodes = 4096;
    const size_t ctx_size = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false) + 1024 * 1024;
    struct ggml_init_params ctx_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ctx_params);
    if (!ctx) return false;

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // Input tensors (data set after allocation)
    struct ggml_tensor * cur = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(cur, "hidden_input");
    ggml_set_input(cur);

    struct ggml_tensor * indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
    ggml_set_name(indices, "expert_indices");
    ggml_set_input(indices);

    struct ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert_used, n_tokens);
    ggml_set_name(weights, "expert_weights");
    ggml_set_input(weights);

    // Reshape cur for mul_mat_id: [n_embd, 1, n_tokens]
    struct ggml_tensor * cur_3d = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);

    // ── Expert FFN (sparse) ──
    struct ggml_tensor * expert_out = nullptr;

    if (layer_data.ffn_gate_up_exps) {
        // Fused gate+up path
        struct ggml_tensor * gate_up = ggml_mul_mat_id(ctx, layer_data.ffn_gate_up_exps, cur_3d, indices);

        const int64_t n_ff = gate_up->ne[0] / 2;
        struct ggml_tensor * gate_part = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
            gate_up->nb[1], gate_up->nb[2], 0);
        struct ggml_tensor * up_part = ggml_view_3d(ctx, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2],
            gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);

        struct ggml_tensor * activated = ggml_swiglu_split(ctx, gate_part, up_part);
        expert_out = ggml_mul_mat_id(ctx, layer_data.ffn_down_exps, activated, indices);
    } else {
        struct ggml_tensor * up = ggml_mul_mat_id(ctx, layer_data.ffn_up_exps, cur_3d, indices);
        struct ggml_tensor * gate = ggml_mul_mat_id(ctx, layer_data.ffn_gate_exps, cur_3d, indices);
        struct ggml_tensor * activated = ggml_swiglu_split(ctx, gate, up);
        expert_out = ggml_mul_mat_id(ctx, layer_data.ffn_down_exps, activated, indices);
    }

    // Apply routing weights and sum across experts
    struct ggml_tensor * weights_3d = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    expert_out = ggml_mul(ctx, expert_out, weights_3d);

    struct ggml_tensor * moe_out = ggml_view_2d(ctx, expert_out, n_embd, n_tokens,
        expert_out->nb[2], 0);
    for (int64_t i = 1; i < n_expert_used; ++i) {
        struct ggml_tensor * expert_i = ggml_view_2d(ctx, expert_out, n_embd, n_tokens,
            expert_out->nb[2], i * expert_out->nb[1]);
        moe_out = ggml_add(ctx, moe_out, expert_i);
    }

    // ── Shared expert FFN ──
    struct ggml_tensor * result = moe_out;

    if (layer_data.ffn_up_shexp) {
        struct ggml_tensor * shexp_up = ggml_mul_mat(ctx, layer_data.ffn_up_shexp, cur);
        struct ggml_tensor * shexp_gate = ggml_mul_mat(ctx, layer_data.ffn_gate_shexp, cur);
        struct ggml_tensor * shexp_activated = ggml_swiglu_split(ctx, shexp_gate, shexp_up);
        struct ggml_tensor * shexp_out = ggml_mul_mat(ctx, layer_data.ffn_down_shexp, shexp_activated);

        if (layer_data.ffn_gate_inp_shexp) {
            struct ggml_tensor * shared_gate = ggml_mul_mat(ctx, layer_data.ffn_gate_inp_shexp, cur);
            shared_gate = ggml_sigmoid(ctx, shared_gate);
            shexp_out = ggml_mul(ctx, shexp_out, shared_gate);
        }

        result = ggml_add(ctx, moe_out, shexp_out);
    }

    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);

    // ── Allocate and execute using backend scheduler ──
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        compute.backends.data(), compute.backend_bufts.data(),
        compute.backends.size(), max_nodes, false, false);

    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        fprintf(stderr, "[expert-worker] Failed to allocate graph\n");
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return false;
    }

    // Set input data
    ggml_backend_tensor_set(cur, hidden_data, 0, n_embd * n_tokens * sizeof(float));
    ggml_backend_tensor_set(indices, indices_data, 0, n_expert_used * n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(weights, weights_data, 0, n_expert_used * n_tokens * sizeof(float));

    // Execute
    ggml_backend_sched_graph_compute(sched, gf);

    // Read output
    ggml_backend_tensor_get(result, output_data, 0, n_embd * n_tokens * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);

    return true;
}

// ── Main ───────────────────────────────────────────────────────────

int main(int argc, char ** argv) {
    // Parse args
    const char * model_path = nullptr;
    int port = 50100;
    int ngl = 999; // default: offload all layers to GPU

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ngl") == 0 && i + 1 < argc) {
            ngl = atoi(argv[++i]);
        }
    }

    if (!model_path) {
        fprintf(stderr, "Usage: llama-expert-worker --model <path.gguf> [--port <port>] [--ngl <layers>]\n");
        return 1;
    }

    // Load model
    fprintf(stderr, "[expert-worker] Loading model: %s\n", model_path);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "[expert-worker] Failed to load model\n");
        return 1;
    }

    const int n_layers = model->hparams.n_layer;
    const int n_embd   = model->hparams.n_embd;
    const int n_expert = model->hparams.n_expert;
    const int n_expert_used = model->hparams.n_expert_used;

    fprintf(stderr, "[expert-worker] Model loaded: %d layers, %d embd, %d experts (%d active)\n",
            n_layers, n_embd, n_expert, n_expert_used);

    // Count expert memory
    size_t expert_bytes = 0;
    for (int il = 0; il < n_layers; il++) {
        const auto & layer = model->layers[il];
        if (layer.ffn_gate_exps)    expert_bytes += ggml_nbytes(layer.ffn_gate_exps);
        if (layer.ffn_up_exps)      expert_bytes += ggml_nbytes(layer.ffn_up_exps);
        if (layer.ffn_down_exps)    expert_bytes += ggml_nbytes(layer.ffn_down_exps);
        if (layer.ffn_gate_up_exps) expert_bytes += ggml_nbytes(layer.ffn_gate_up_exps);
        if (layer.ffn_up_shexp)     expert_bytes += ggml_nbytes(layer.ffn_up_shexp);
        if (layer.ffn_gate_shexp)   expert_bytes += ggml_nbytes(layer.ffn_gate_shexp);
        if (layer.ffn_down_shexp)   expert_bytes += ggml_nbytes(layer.ffn_down_shexp);
    }
    fprintf(stderr, "[expert-worker] Expert weights: %.1f MB\n", expert_bytes / (1024.0 * 1024.0));

    // Initialize compute backends
    expert_compute_ctx compute;
    if (!compute.init(*model)) {
        fprintf(stderr, "[expert-worker] Failed to initialize compute backends\n");
        return 1;
    }

    // Create TCP server
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 1;
    }

    listen(srv, 1);
    fprintf(stderr, "[expert-worker] Listening on port %d\n", port);

    // Accept connections (one at a time)
    while (true) {
        struct sockaddr_in client_addr = {};
        socklen_t client_len = sizeof(client_addr);
        int client = accept(srv, (struct sockaddr *)&client_addr, &client_len);
        if (client < 0) {
            perror("accept");
            continue;
        }

        // Disable Nagle for low latency
        int flag = 1;
        setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, sizeof(client_ip));
        fprintf(stderr, "[expert-worker] Client connected: %s\n", client_ip);

        // Serve requests on this connection
        while (true) {
            // Read header
            uint32_t header[5]; // magic, layer, n_tokens, n_expert_used, n_embd
            if (!recv_all(client, header, sizeof(header))) {
                fprintf(stderr, "[expert-worker] Client disconnected\n");
                break;
            }

            if (header[0] != MAGIC_REQ) {
                fprintf(stderr, "[expert-worker] Bad magic: 0x%08x\n", header[0]);
                break;
            }

            const uint32_t layer        = header[1];
            const uint32_t n_tok        = header[2];
            const uint32_t n_exp_used   = header[3];
            const uint32_t recv_n_embd  = header[4];

            if (layer >= (uint32_t)n_layers || recv_n_embd != (uint32_t)n_embd) {
                fprintf(stderr, "[expert-worker] Invalid request: layer=%u n_tok=%u n_exp=%u n_embd=%u\n",
                        layer, n_tok, n_exp_used, recv_n_embd);
                break;
            }

            // Read payload
            const size_t hidden_bytes  = recv_n_embd * n_tok * sizeof(float);
            const size_t indices_bytes = n_exp_used * n_tok * sizeof(int32_t);
            const size_t weights_bytes = n_exp_used * n_tok * sizeof(float);

            std::vector<float>   hidden(recv_n_embd * n_tok);
            std::vector<int32_t> indices(n_exp_used * n_tok);
            std::vector<float>   weights(n_exp_used * n_tok);
            std::vector<float>   output(recv_n_embd * n_tok, 0.0f);

            if (!recv_all(client, hidden.data(), hidden_bytes) ||
                !recv_all(client, indices.data(), indices_bytes) ||
                !recv_all(client, weights.data(), weights_bytes)) {
                fprintf(stderr, "[expert-worker] Failed to read payload\n");
                break;
            }

            // Compute
            auto t0 = std::chrono::high_resolution_clock::now();

            bool ok = compute_expert_ffn(compute, *model, layer, hidden.data(), recv_n_embd, n_tok,
                                          indices.data(), weights.data(), n_exp_used, output.data());

            auto t1 = std::chrono::high_resolution_clock::now();
            uint32_t compute_us = (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

            if (!ok) {
                fprintf(stderr, "[expert-worker] Compute failed for layer %u\n", layer);
                // Send zero output
                memset(output.data(), 0, output.size() * sizeof(float));
            }

            // Send response
            uint32_t resp_header[2] = { MAGIC_RESP, compute_us };
            if (!send_all(client, resp_header, sizeof(resp_header)) ||
                !send_all(client, output.data(), recv_n_embd * n_tok * sizeof(float))) {
                fprintf(stderr, "[expert-worker] Failed to send response\n");
                break;
            }
        }

#ifdef _WIN32
        closesocket(client);
#else
        close(client);
#endif
    }

    llama_model_free(model);
    return 0;
}
