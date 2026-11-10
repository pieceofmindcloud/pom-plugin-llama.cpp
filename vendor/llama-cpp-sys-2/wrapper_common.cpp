#include "wrapper_common.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <stdint.h>
#include <unordered_map>
#include <vector>

#include "common.h"
// Cabeçalhos internos: `llama_model` (`tensors_by_name`, `dev_layer`), o
// `llama-ext.h` de estágio (`llama_get_memory_breakdown`) e o `llama_context`
// (`get_cparams`, o Flash Attention resolvido). Nada disso está no `llama.h`
// público, e é justamente o que responde onde cada camada ficou e com que
// parâmetros o contexto ficou de pé.
#include "llama-ext.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "fit.h"
#include "log.h"
#include "json-schema-to-grammar.h"
#include "speculative.h"
#include "llama.h"
#include "wrapper_utils.h"

#include <nlohmann/json.hpp>

extern "C" llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar) {
    if (!schema_json || !out_grammar) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    *out_grammar = nullptr;
    try {
        const auto schema = common_json::parse(schema_json);
        const auto grammar = json_schema_to_grammar(schema, force_gbnf);
        *out_grammar = llama_rs_dup_string(grammar);
        return *out_grammar ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" void llama_rs_string_free(char * ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    try {
        return llama_sampler_init_grammar(vocab, grammar_str, grammar_root);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        std::vector<std::string> trigger_patterns;
        trigger_patterns.reserve(num_trigger_words);
        for (size_t i = 0; i < num_trigger_words; ++i) {
            const char * word = trigger_words ? trigger_words[i] : nullptr;
            if (word && word[0] != '\0') {
                trigger_patterns.push_back(regex_escape(word));
            }
        }
        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & pattern : trigger_patterns) {
            trigger_patterns_c.push_back(pattern.c_str());
        }
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns_c.data(),
            trigger_patterns_c.size(),
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens) {
    try {
        return llama_sampler_init_grammar_lazy_patterns(
            vocab,
            grammar_str,
            grammar_root,
            trigger_patterns,
            num_trigger_patterns,
            trigger_tokens,
            num_trigger_tokens);
    } catch (...) {
        return nullptr;
    }
}

extern "C" llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token) {
    if (!sampler) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    try {
        llama_sampler_accept(sampler, token);
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

class common_log_trace_guard {
public:
    common_log_trace_guard()
        : previous_(common_log_get_verbosity_thold()) {
        common_log_set_verbosity_thold(LOG_LEVEL_TRACE);
    }

    ~common_log_trace_guard() {
        common_log_set_verbosity_thold(previous_);
    }

private:
    int previous_;
};

// Thin pass-through to llama.cpp's common_fit_params (a C++ symbol in libcommon).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
// The fit decision is otherwise invisible at the default log threshold. Its
// trace is enabled only for this non-thread-safe call and restored on return.
extern "C" int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level) {
    common_log_trace_guard trace;
    return static_cast<int>(common_fit_params(
        { std::string(path_model) },
        mparams,
        cparams,
        tensor_split,
        tensor_buft_overrides,
        margins,
        n_ctx_min,
        nullptr,
        log_level));
}

// Backend plugins expose metadata as borrowed C strings. The GGML interface
// describes these values but does not make non-null a part of the ABI, and a
// third-party/backend failure can therefore return nullptr. Passing such a
// value to std::string or nlohmann::json calls strlen(nullptr) on Windows.
// Health diagnostics must remain observational even when backend metadata is
// incomplete: losing the label is preferable to taking down the node after a
// successful model load.
static const char * llama_rs_cstr_or(const char * value, const char * fallback) {
    return value ? value : fallback;
}

static std::vector<ggml_backend_dev_t> llama_rs_default_devices() {
    // Keep this selection in lockstep with llama_prepare_model_devices: RPC
    // devices first, then discrete GPUs, then integrated GPUs only when there
    // is no discrete GPU. This is backend metadata, not a vendor allow-list.
    std::vector<ggml_backend_dev_t> rpc_servers;
    std::vector<ggml_backend_dev_t> gpus;
    std::vector<ggml_backend_dev_t> igpus;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) {
            continue;
        }
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
            case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                break;

            case GGML_BACKEND_DEVICE_TYPE_GPU: {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                if (reg && std::string(llama_rs_cstr_or(ggml_backend_reg_name(reg), "")) == "RPC") {
                    rpc_servers.push_back(dev);
                    break;
                }

                ggml_backend_dev_props props = {};
                ggml_backend_dev_get_props(dev, &props);
                const bool duplicate = std::any_of(
                    gpus.begin(),
                    gpus.end(),
                    [&props](ggml_backend_dev_t existing) {
                        ggml_backend_dev_props existing_props = {};
                        ggml_backend_dev_get_props(existing, &existing_props);
                        return props.device_id && existing_props.device_id
                            && std::strcmp(props.device_id, existing_props.device_id) == 0;
                    });
                if (!duplicate) {
                    gpus.push_back(dev);
                }
                break;
            }

            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                if (igpus.empty()
                    || ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(igpus.back())) {
                    igpus.push_back(dev);
                }
                break;

            case GGML_BACKEND_DEVICE_TYPE_META:
                break;
        }
    }

    std::vector<ggml_backend_dev_t> devices;
    devices.reserve(rpc_servers.size() + gpus.size() + igpus.size());
    devices.insert(devices.end(), rpc_servers.begin(), rpc_servers.end());
    devices.insert(devices.end(), gpus.begin(), gpus.end());
    if (gpus.empty()) {
        devices.insert(devices.end(), igpus.begin(), igpus.end());
    }
    return devices;
}

extern "C" size_t llama_rs_get_devices(
    ggml_backend_dev_t * out,
    size_t capacity) {
    if (!out || capacity == 0) {
        return 0;
    }

    const std::vector<ggml_backend_dev_t> devices = llama_rs_default_devices();
    const size_t written = std::min(capacity, devices.size());
    std::copy_n(devices.begin(), written, out);
    return written;
}

extern "C" size_t llama_rs_get_device_memory(
    struct llama_rs_device_memory * out,
    size_t capacity) {
    if (!out || capacity == 0) {
        return 0;
    }

    const std::vector<ggml_backend_dev_t> devices = llama_rs_default_devices();

    const size_t written = std::min(capacity, devices.size());
    for (size_t i = 0; i < written; ++i) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_dev_memory(devices[i], &free_bytes, &total_bytes);
        out[i] = {free_bytes, total_bytes};
    }
    return written;
}

static float llama_rs_benchmark_device_uncached(ggml_backend_dev_t device) {
    if (!device) {
        return 0.0f;
    }

    // A matrix-vector product approximates the memory-bound shape of one
    // decode token while keeping the one-time probe below 16 MiB of weights.
    // It uses only the public ggml backend graph API, so RPC and every other
    // backend are measured through the same interface.
    constexpr int n = 2048;
    constexpr int runs = 3;
    constexpr size_t context_bytes = 1024 * 1024;

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        return 0.0f;
    }

    ggml_init_params init_params = {
        /* .mem_size   = */ context_bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(init_params);
    if (!ctx) {
        ggml_backend_free(backend);
        return 0.0f;
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    if (!weights || !input || !output || !graph) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 0.0f;
    }
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 0.0f;
    }
    ggml_backend_buffer_clear(buffer, 1);

    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 0.0f;
    }
    ggml_backend_synchronize(backend);

    double elapsed_seconds = 0.0;
    for (int i = 0; i < runs; ++i) {
        const auto started = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return 0.0f;
        }
        ggml_backend_synchronize(backend);
        const auto elapsed = std::chrono::steady_clock::now() - started;
        elapsed_seconds += std::chrono::duration<double>(elapsed).count();
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);

    return elapsed_seconds > 0.0
        ? static_cast<float>(runs / elapsed_seconds)
        : 0.0f;
}

extern "C" float llama_rs_benchmark_device(ggml_backend_dev_t device) {
    static std::mutex mutex;
    static std::unordered_map<ggml_backend_dev_t, float> scores;

    std::lock_guard<std::mutex> lock(mutex);
    const auto found = scores.find(device);
    if (found != scores.end()) {
        return found->second;
    }

    const float score = llama_rs_benchmark_device_uncached(device);
    scores.emplace(device, score);
    return score;
}

extern "C" void llama_rs_memory_breakdown_print(const struct llama_context * ctx) {
    common_memory_breakdown_print(ctx);
}

// ── Onde o modelo ficou ───────────────────────────────────────────────────────

// O bloco a que um tensor pertence, ou -1 quando ele não é de bloco nenhum
// (embeddings, norma de saída, cabeça). Os nomes do GGUF são `blk.<i>.<resto>`.
static int llama_rs_tensor_block(const std::string & name) {
    static const std::string prefix = "blk.";
    if (name.compare(0, prefix.size(), prefix) != 0) {
        return -1;
    }
    const size_t dot = name.find('.', prefix.size());
    if (dot == std::string::npos) {
        return -1;
    }
    try {
        return std::stoi(name.substr(prefix.size(), dot - prefix.size()));
    } catch (const std::exception &) {
        return -1;
    }
}

// Quantos bytes de um grupo de tensores foram parar em cada buffer.
// `map` ordenado só para a saída sair estável entre execuções.
using llama_rs_buffer_bytes = std::map<std::string, size_t>;

static const char * llama_rs_backend_name(ggml_backend_dev_t dev) {
    return dev ? llama_rs_cstr_or(ggml_backend_dev_name(dev), "<unknown-device>") : "CPU";
}

static const char * llama_rs_backend_description(ggml_backend_dev_t dev) {
    return dev
        ? llama_rs_cstr_or(ggml_backend_dev_description(dev), "<unknown-description>")
        : "CPU";
}

static const char * llama_rs_buffer_name(ggml_backend_buffer_t buffer) {
    return buffer
        ? llama_rs_cstr_or(ggml_backend_buffer_name(buffer), "<unknown-buffer>")
        : "<unknown-buffer>";
}

static const char * llama_rs_buft_name(ggml_backend_buffer_type_t buft) {
    return buft
        ? llama_rs_cstr_or(ggml_backend_buft_name(buft), "<unknown-buffer-type>")
        : "<unknown-buffer-type>";
}

static void llama_rs_account(llama_rs_buffer_bytes & into, const ggml_tensor * t) {
    if (!t || !t->buffer) {
        return;
    }
    into[llama_rs_buffer_name(t->buffer)] += ggml_nbytes(t);
}

static nlohmann::ordered_json llama_rs_buffers_json(const llama_rs_buffer_bytes & bytes) {
    auto out = nlohmann::ordered_json::array();
    for (const auto & [name, n] : bytes) {
        out.push_back({{"buffer", name}, {"bytes", n}});
    }
    return out;
}

// Backend metadata is borrowed from llama.cpp. Keep the null-to-JSON policy in
// one place so every field that is meant to be textual is safe even when a
// backend returns incomplete properties. `device_id` is intentionally kept as
// JSON null when absent because it is an optional identifier, not a label.
static nlohmann::ordered_json llama_rs_backend_device_json(
        const char * backend,
        const char * description,
        const char * device_id) {
    return {
        {"backend",     llama_rs_cstr_or(backend, "")},
        {"description", llama_rs_cstr_or(description, "")},
        {"device_id",   device_id ? nlohmann::ordered_json(device_id) : nullptr},
    };
}

extern "C" llama_rs_status llama_rs_model_placement_json(const struct llama_model * model, char ** out_json) {
    if (!model || !out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    try {
        const int n_layer = llama_model_n_layer(model);

        std::vector<llama_rs_buffer_bytes> per_layer(n_layer > 0 ? (size_t) n_layer : 0);
        llama_rs_buffer_bytes other;
        llama_rs_buffer_bytes totals;

        for (const auto & [name, tensor] : model->tensors_by_name) {
            const int il = llama_rs_tensor_block(name);
            if (il >= 0 && il < n_layer) {
                llama_rs_account(per_layer[(size_t) il], tensor);
            } else {
                llama_rs_account(other, tensor);
            }
            llama_rs_account(totals, tensor);
        }

        auto layers = nlohmann::ordered_json::array();
        for (int il = 0; il < n_layer; ++il) {
            // O dispositivo é a atribuição que o llama.cpp fez ao bloco; os
            // buffers são onde os pesos dele estão. Os dois divergem
            // exatamente no caso que interessa ver.
            ggml_backend_dev_t dev = model->dev_layer(il);
            layers.push_back({
                {"layer",   il},
                {"device",  llama_rs_backend_name(dev)},
                {"buffers", llama_rs_buffers_json(per_layer[(size_t) il])},
            });
        }

        ggml_backend_dev_t dev_out = model->dev_output();
        auto devices = nlohmann::ordered_json::array();
        for (int i = 0; i < llama_model_n_devices(model); ++i) {
            ggml_backend_dev_t dev = llama_model_get_device(model, i);
            if (!dev) {
                continue;
            }
            devices.push_back({
                {"name",        llama_rs_backend_name(dev)},
                {"description", llama_rs_backend_description(dev)},
            });
        }

        const nlohmann::ordered_json out = {
            {"n_layer",              n_layer},
            {"n_gpu_layers",         model->n_gpu_layers()},
            {"has_tensor_overrides", model->has_tensor_overrides()},
            {"devices",              devices},
            {"layers",               layers},
            {"output", {
                {"device",  llama_rs_backend_name(dev_out)},
                {"buffers", llama_rs_buffers_json(other)},
            }},
            {"totals", llama_rs_buffers_json(totals)},
        };

        *out_json = llama_rs_dup_string(out.dump());
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

static nlohmann::ordered_json llama_rs_context_memory_value(
        const struct llama_context * ctx,
        const struct llama_model * model) {
    auto buffers = nlohmann::ordered_json::array();
    for (const auto & [buft, data] : llama_get_memory_breakdown(ctx)) {
        if (!buft) {
            continue;
        }
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        int device_index = -1;
        if (dev && model) {
            for (int i = 0; i < llama_model_n_devices(model); ++i) {
                if (dev == llama_model_get_device(model, i)) {
                    device_index = i;
                    break;
                }
            }
        }
        buffers.push_back({
            {"buffer",       llama_rs_buft_name(buft)},
            {"device",       llama_rs_backend_name(dev)},
            {"host",         ggml_backend_buft_is_host(buft)},
            {"device_index", device_index >= 0 ? nlohmann::ordered_json(device_index) : nullptr},
            {"model",        data.model},
            // `context` é o cache KV (mais o estado recorrente, quando o
            // modelo tem um). É esta linha que responde onde o KV ficou.
            {"context",      data.context},
            {"compute",      data.compute},
        });
    }
    return {{"buffers", buffers}};
}

extern "C" llama_rs_status llama_rs_backend_devices_json(char ** out_json) {
    if (!out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    try {
        auto devices = nlohmann::ordered_json::array();
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!dev) {
                continue;
            }

            ggml_backend_dev_props props = {};
            ggml_backend_dev_get_props(dev, &props);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            devices.push_back(llama_rs_backend_device_json(
                reg ? ggml_backend_reg_name(reg) : nullptr,
                props.description,
                props.device_id));
        }

        *out_json = llama_rs_dup_string(devices.dump());
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_context_memory_json(const struct llama_context * ctx, char ** out_json) {
    if (!ctx || !out_json) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    *out_json = nullptr;

    try {
        *out_json = llama_rs_dup_string(llama_rs_context_memory_value(ctx, nullptr).dump());
        return *out_json ? LLAMA_RS_STATUS_OK : LLAMA_RS_STATUS_ALLOCATION_FAILED;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_context_flash_attn(
        const struct llama_context * ctx,
        bool * out_enabled,
        bool * out_auto) {
    if (!ctx || !out_enabled || !out_auto) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        // `get_cparams` é a decisão já tomada, não o pedido: o construtor do
        // contexto resolve `AUTO` sondando o backend antes de gravar aqui.
        const llama_cparams & cparams = const_cast<llama_context *>(ctx)->get_cparams();
        *out_enabled = cparams.flash_attn;
        *out_auto    = cparams.auto_fa;
        return LLAMA_RS_STATUS_OK;
    } catch (const std::exception &) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

static ggml_tensor * llama_rs_moe_expert_view(
        ggml_context * ctx,
        const ggml_tensor * tensor,
        uint32_t expert) {
    if (!tensor || ggml_n_dims(tensor) != 3 || expert >= static_cast<uint32_t>(tensor->ne[2])) {
        return nullptr;
    }
    return ggml_view_2d(
        ctx,
        const_cast<ggml_tensor *>(tensor),
        tensor->ne[0],
        tensor->ne[1],
        tensor->nb[1],
        static_cast<size_t>(expert) * tensor->nb[2]);
}

static ggml_tensor * llama_rs_moe_expert_scale(
        ggml_context * ctx,
        const llama_model * model,
        const std::string & base,
        uint32_t expert) {
    const ggml_tensor * tensor = model->get_tensor((base + ".scale").c_str());
    const std::string weight_suffix = ".weight";
    if (!tensor && base.size() >= weight_suffix.size() &&
        base.compare(base.size() - weight_suffix.size(), weight_suffix.size(), weight_suffix) == 0) {
        const std::string stem = base.substr(0, base.size() - weight_suffix.size());
        tensor = model->get_tensor((stem + ".scale").c_str());
    }
    if (!tensor || ggml_n_dims(tensor) != 1 || expert >= static_cast<uint32_t>(tensor->ne[0])) {
        return nullptr;
    }
    return ggml_view_1d(
        ctx,
        const_cast<ggml_tensor *>(tensor),
        1,
        static_cast<size_t>(expert) * tensor->nb[0]);
}

static ggml_tensor * llama_rs_moe_scaled_matmul(
        ggml_context * ctx,
        const llama_model * model,
        const ggml_tensor * weight,
        const std::string & weight_name,
        uint32_t expert,
        ggml_tensor * hidden) {
    ggml_tensor * view = llama_rs_moe_expert_view(ctx, weight, expert);
    if (!view) {
        return nullptr;
    }
    ggml_tensor * result = ggml_mul_mat(ctx, view, hidden);
    if (ggml_tensor * scale = llama_rs_moe_expert_scale(ctx, model, weight_name, expert)) {
        result = ggml_mul(ctx, result, scale);
    }
    return result;
}

extern "C" int llama_rs_moe_reference_expert(
        const struct llama_model * model,
        int32_t layer,
        uint32_t expert,
        const float * hidden,
        size_t hidden_count,
        int32_t activation,
        float * gate,
        size_t gate_capacity,
        float * up,
        size_t up_capacity,
        float * activated,
        size_t activated_capacity,
        float * output,
        size_t output_capacity) {
    if (!model || !hidden || !gate || !up || !activated || !output || layer < 0) {
        return 1;
    }

    try {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const std::string gate_up_name = prefix + "ffn_gate_up_exps.weight";
        const std::string gate_name = prefix + "ffn_gate_exps.weight";
        const std::string up_name = prefix + "ffn_up_exps.weight";
        const std::string down_name = prefix + "ffn_down_exps.weight";
        const ggml_tensor * gate_up_weight = model->get_tensor(gate_up_name.c_str());
        const ggml_tensor * gate_weight = model->get_tensor(gate_name.c_str());
        const ggml_tensor * up_weight = model->get_tensor(up_name.c_str());
        const ggml_tensor * down_weight = model->get_tensor(down_name.c_str());
        const ggml_tensor * first_weight = gate_up_weight ? gate_up_weight : gate_weight;
        if (!first_weight || !down_weight || hidden_count != static_cast<size_t>(first_weight->ne[0])) {
            return 2;
        }

        const size_t intermediate = gate_up_weight
            ? static_cast<size_t>(gate_up_weight->ne[1] / 2)
            : static_cast<size_t>(gate_weight->ne[1]);
        const size_t output_width = static_cast<size_t>(down_weight->ne[1]);
        if (intermediate == 0 || output_width == 0 ||
            gate_capacity < intermediate || up_capacity < intermediate ||
            activated_capacity < intermediate || output_capacity < output_width) {
            return 2;
        }

        // Keep this wrapper compatible with the C++17 mode used by cc-rs and CMake.
        ggml_init_params init{};
        init.mem_size = 64ull * 1024ull * 1024ull;
        init.mem_buffer = nullptr;
        init.no_alloc = true;
        ggml_context * ctx = ggml_init(init);
        if (!ctx) {
            return 3;
        }

        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_count, 1);

        ggml_tensor * gate_node = nullptr;
        ggml_tensor * up_node = nullptr;
        if (gate_up_weight) {
            ggml_tensor * gate_up = llama_rs_moe_expert_view(ctx, gate_up_weight, expert);
            if (!gate_up) {
                ggml_free(ctx);
                return 2;
            }
            ggml_tensor * gate_up_result = ggml_mul_mat(ctx, gate_up, input);
            gate_node = ggml_view_1d(ctx, gate_up_result, intermediate, 0);
            up_node = ggml_view_1d(ctx, gate_up_result, intermediate, intermediate * sizeof(float));
        } else {
            if (!gate_weight || !up_weight) {
                ggml_free(ctx);
                return 2;
            }
            gate_node = llama_rs_moe_scaled_matmul(
                ctx, model, gate_weight, gate_name, expert, input);
            up_node = llama_rs_moe_scaled_matmul(
                ctx, model, up_weight, up_name, expert, input);
        }
        if (!gate_node || !up_node) {
            ggml_free(ctx);
            return 2;
        }

        // The fused Gemma4 path has no native gate/up scale tensors. The
        // separate path was already scaled by llama_rs_moe_scaled_matmul.
        ggml_tensor * activated_node = nullptr;
        switch (activation) {
            case 1:
                activated_node = ggml_swiglu_split(ctx, gate_node, up_node);
                break;
            case 2:
                activated_node = ggml_geglu_split(ctx, gate_node, up_node);
                break;
            default:
                ggml_free(ctx);
                return 4;
        }
        if (!activated_node) {
            ggml_free(ctx);
            return 3;
        }

        ggml_tensor * down_node = llama_rs_moe_scaled_matmul(
            ctx, model, down_weight, down_name, expert, activated_node);
        if (!down_node) {
            ggml_free(ctx);
            return 2;
        }

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, down_node);
        // Through the backend registry, not ggml_backend_cpu_init(): with
        // loadable backends (GGML_BACKEND_DL) the CPU backend lives in its own
        // module, and the Windows link against the release found no such
        // symbol.
        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend) {
            ggml_free(ctx);
            return 3;
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            ggml_backend_free(backend);
            ggml_free(ctx);
            return 3;
        }
        ggml_backend_tensor_set(input, hidden, 0, hidden_count * sizeof(float));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buffer);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return 3;
        }

        ggml_backend_tensor_get(gate_node, gate, 0, intermediate * sizeof(float));
        ggml_backend_tensor_get(up_node, up, 0, intermediate * sizeof(float));
        ggml_backend_tensor_get(activated_node, activated, 0, intermediate * sizeof(float));
        ggml_backend_tensor_get(down_node, output, 0, output_width * sizeof(float));
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return 0;
    } catch (...) {
        return 3;
    }
}

struct llama_rs_mtp_speculative {
    common_params_speculative params;
    common_speculative * spec = nullptr;
    std::vector<llama_token> prompt;
    std::vector<llama_token> draft;
    size_t last_draft_len = 0;
    bool draft_pending = false;
};

static constexpr llama_seq_id LLAMA_RS_MTP_SEQ_ID = 0;

static bool llama_rs_mtp_batch_compatible(const struct llama_batch & batch) {
    if (batch.n_tokens <= 0 || !batch.token || batch.embd || !batch.pos || !batch.n_seq_id ||
        !batch.seq_id) {
        return false;
    }
    for (int32_t k = 0; k < batch.n_tokens; ++k) {
        if (batch.n_seq_id[k] != 1 || !batch.seq_id[k] ||
            batch.seq_id[k][0] != LLAMA_RS_MTP_SEQ_ID) {
            return false;
        }
    }
    return true;
}

static void llama_rs_assign_tokens(
    std::vector<llama_token> & dst,
    const llama_token * tokens,
    size_t count) {
    if (count == 0) {
        dst.clear();
        return;
    }
    dst.assign(tokens, tokens + count);
}

extern "C" struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min) {
    if (!ctx_tgt || !ctx_dft || n_max <= 0 || n_min < 0 || n_min > n_max) {
        return nullptr;
    }

    try {
        auto wrapper = std::make_unique<llama_rs_mtp_speculative>();
        wrapper->params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        wrapper->params.draft.ctx_tgt = ctx_tgt;
        wrapper->params.draft.ctx_dft = ctx_dft;
        wrapper->params.draft.n_max = n_max;
        wrapper->params.draft.n_min = n_min;
        wrapper->params.draft.p_min = p_min;

        wrapper->spec = common_speculative_init(wrapper->params, 1);
        if (!wrapper->spec) {
            return nullptr;
        }

        return wrapper.release();
    } catch (...) {
        return nullptr;
    }
}

extern "C" void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec) {
    if (!spec) {
        return;
    }
    if (spec->spec) {
        common_speculative_free(spec->spec);
        spec->spec = nullptr;
    }
    delete spec;
}

extern "C" llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        common_speculative_begin(spec->spec, LLAMA_RS_MTP_SEQ_ID, spec->prompt);
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch) {
    if (!spec || !spec->spec || !batch) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!llama_rs_mtp_batch_compatible(*batch)) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        // The POM fork's common API now consumes common_batch after the
        // upstream batch_ext migration. Convert the validated public batch;
        // this hook only needs its token IDs, positions, sequence IDs and
        // output flags, not the native batch allocation.
        common_batch common;
        for (int32_t i = 0; i < batch->n_tokens; ++i) {
            const bool output = batch->logits
                ? batch->logits[i] != 0
                : i == batch->n_tokens - 1;
            common.add(batch->token[i], batch->pos[i], LLAMA_RS_MTP_SEQ_ID, output);
        }
        return common_speculative_process(spec->spec, common)
            ? LLAMA_RS_STATUS_OK
            : LLAMA_RS_STATUS_EXCEPTION;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count) {
    if (!spec || !spec->spec || (!prompt_tokens && prompt_tokens_count > 0) ||
        !out_tokens_count || n_past < 0) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        if (spec->draft_pending) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        llama_rs_assign_tokens(spec->prompt, prompt_tokens, prompt_tokens_count);
        spec->draft.clear();
        spec->last_draft_len = 0;

        auto & params = common_speculative_get_draft_params(spec->spec, LLAMA_RS_MTP_SEQ_ID);
        params = {
            true,
            spec->params.draft.n_max,
            n_past,
            id_last,
            &spec->prompt,
            &spec->draft,
        };

        common_speculative_draft(spec->spec);

        *out_tokens_count = spec->draft.size();
        if (spec->draft.size() > out_tokens_capacity) {
            return LLAMA_RS_STATUS_ALLOCATION_FAILED;
        }
        if (!spec->draft.empty() && !out_tokens) {
            return LLAMA_RS_STATUS_INVALID_ARGUMENT;
        }
        if (!spec->draft.empty()) {
            std::memcpy(out_tokens, spec->draft.data(), spec->draft.size() * sizeof(llama_token));
        }
        spec->last_draft_len = spec->draft.size();
        spec->draft_pending = !spec->draft.empty();
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}

extern "C" llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted) {
    if (!spec || !spec->spec) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }
    if (!spec->draft_pending || n_accepted > spec->last_draft_len) {
        return LLAMA_RS_STATUS_INVALID_ARGUMENT;
    }

    try {
        common_speculative_accept(spec->spec, LLAMA_RS_MTP_SEQ_ID, n_accepted);
        spec->last_draft_len = 0;
        spec->draft_pending = false;
        return LLAMA_RS_STATUS_OK;
    } catch (...) {
        return LLAMA_RS_STATUS_EXCEPTION;
    }
}
