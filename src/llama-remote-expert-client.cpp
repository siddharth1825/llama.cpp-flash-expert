// TCP client for remote expert computation.
// Connects to llama-expert-worker and installs the remote expert hook.

#include "llama-remote-expert-client.h"
#include "llama-remote-expert.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <mutex>

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
#include <netdb.h>
#include <unistd.h>
#endif

static const uint32_t MAGIC_REQ  = 0x45585054; // "EXPT"
static const uint32_t MAGIC_RESP = 0x52455350; // "RESP"

static int g_socket = -1;
static std::mutex g_mutex; // serialize requests (one at a time)

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

// The hook callback — called during GGML graph execution
static bool remote_expert_callback(
        const float * hidden,
        int64_t       n_embd,
        int64_t       n_tokens,
        const int32_t * indices,
        const float * weights,
        int64_t       n_expert_used,
        int           layer,
        float *       output) {

    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_socket < 0) return false;

    // Send request header
    uint32_t header[5] = {
        MAGIC_REQ,
        (uint32_t)layer,
        (uint32_t)n_tokens,
        (uint32_t)n_expert_used,
        (uint32_t)n_embd,
    };

    if (!send_all(g_socket, header, sizeof(header))) return false;

    // Send payload
    if (!send_all(g_socket, hidden,  n_embd * n_tokens * sizeof(float)))    return false;
    if (!send_all(g_socket, indices, n_expert_used * n_tokens * sizeof(int32_t))) return false;
    if (!send_all(g_socket, weights, n_expert_used * n_tokens * sizeof(float)))   return false;

    // Receive response header
    uint32_t resp_header[2]; // magic, compute_us
    if (!recv_all(g_socket, resp_header, sizeof(resp_header))) return false;

    if (resp_header[0] != MAGIC_RESP) {
        fprintf(stderr, "[remote-expert] Bad response magic: 0x%08x\n", resp_header[0]);
        return false;
    }

    // Receive output
    if (!recv_all(g_socket, output, n_embd * n_tokens * sizeof(float))) return false;

    return true;
}

static bool remote_expert_connect_impl(const std::string & addr) {
    // Parse host:port
    size_t colon = addr.rfind(':');
    if (colon == std::string::npos) {
        fprintf(stderr, "[remote-expert] Invalid address: %s (expected host:port)\n", addr.c_str());
        return false;
    }

    std::string host = addr.substr(0, colon);
    int port = atoi(addr.substr(colon + 1).c_str());

#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
#endif

    // Resolve hostname
    struct addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo * res = nullptr;

    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        fprintf(stderr, "[remote-expert] Failed to resolve: %s\n", host.c_str());
        return false;
    }

    g_socket = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (g_socket < 0) {
        freeaddrinfo(res);
        perror("[remote-expert] socket");
        return false;
    }

    if (connect(g_socket, res->ai_addr, res->ai_addrlen) < 0) {
        freeaddrinfo(res);
        perror("[remote-expert] connect");
#ifdef _WIN32
        closesocket(g_socket);
#else
        close(g_socket);
#endif
        g_socket = -1;
        return false;
    }

    freeaddrinfo(res);

    // Disable Nagle
    int flag = 1;
    setsockopt(g_socket, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    fprintf(stderr, "[remote-expert] Connected to %s\n", addr.c_str());

    // Install the hook
    llama_set_remote_expert_hook(remote_expert_callback);

    return true;
}

static void remote_expert_disconnect_impl() {
    llama_clear_remote_expert_hook();

    if (g_socket >= 0) {
#ifdef _WIN32
        closesocket(g_socket);
#else
        close(g_socket);
#endif
        g_socket = -1;
    }
}

// Public API (declared in llama.h and llama-remote-expert-client.h)

bool llama_remote_expert_connect(const std::string & addr) {
    return remote_expert_connect_impl(addr);
}

void llama_remote_expert_disconnect() {
    remote_expert_disconnect_impl();
}

// C API wrappers removed — use C++ API directly.
