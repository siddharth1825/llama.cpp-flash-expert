#pragma once

// TCP client that connects to llama-expert-worker and provides a
// llama_remote_expert_fn callback for the remote expert hook.

#include "llama-remote-expert.h"
#include <string>

// Connect to a remote expert worker and install the hook.
// Returns true on success.
// addr: "host:port" (e.g., "192.168.1.100:50100")
bool llama_remote_expert_connect(const std::string & addr);

// Disconnect and clear the hook.
void llama_remote_expert_disconnect();
