#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stddef.h>

struct llama_model;
struct llama_sampler;
struct llama_rs_mtp_speculative;
struct llama_vocab;

#include "wrapper_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

llama_rs_status llama_rs_json_schema_to_grammar(
    const char * schema_json,
    bool force_gbnf,
    char ** out_grammar);

struct llama_sampler * llama_rs_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_words,
    size_t num_trigger_words,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

struct llama_sampler * llama_rs_sampler_init_grammar_lazy_patterns(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    const char ** trigger_patterns,
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens,
    size_t num_trigger_tokens);

llama_rs_status llama_rs_sampler_accept(struct llama_sampler * sampler, llama_token token);

// Fit model/context params to device memory (wraps llama.cpp's common_fit_params).
// Returns common_params_fit_status as an int: 0 = success, 1 = failure, 2 = error.
int llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    enum ggml_log_level log_level);

// Returns the current free and total memory for the devices selected by the
// default llama.cpp model placement, in the same order used by common_fit_params.
// The return value is the number of records written, capped by `capacity`.
struct llama_rs_device_memory {
    size_t free_bytes;
    size_t total_bytes;
};

// Returns the devices selected by llama.cpp's default layer-split policy, in
// the order it would pass to the model. The handles remain owned by ggml.
size_t llama_rs_get_devices(
    ggml_backend_dev_t * out,
    size_t capacity);

size_t llama_rs_get_device_memory(
    struct llama_rs_device_memory * out,
    size_t capacity);

// Measures one small generic matrix-vector graph on the backend. The result is
// cached per device for the process; zero means that the backend could not be
// measured and callers must retain the default device order.
float llama_rs_benchmark_device(ggml_backend_dev_t device);

void llama_rs_memory_breakdown_print(const struct llama_context * ctx);

// Onde cada camada do modelo carregado realmente ficou, em JSON.
//
// Não é o `n_gpu_layers` pedido nem o efetivo: é o buffer em que cada tensor
// de cada bloco foi mesmo alocado, lido de `tensors_by_name`. Só assim um
// override de buffer type (o `fit` empurrando os experts de um MoE para a
// CPU) aparece, em vez de uma camada contada como "na GPU" porque o
// dispositivo dela é a GPU.
//
// O chamador é dono da string e a libera com `llama_rs_string_free`.
llama_rs_status llama_rs_model_placement_json(const struct llama_model * model, char ** out_json);

// A memória do contexto por buffer, em JSON: pesos, contexto (o cache KV) e
// buffers de cálculo. Responde onde o cache KV ficou de verdade - a flag
// `offload_kqv` é o pedido, e o llama.cpp pode não atendê-lo numa GPU cheia.
//
// O chamador é dono da string e a libera com `llama_rs_string_free`.
llama_rs_status llama_rs_context_memory_json(const struct llama_context * ctx, char ** out_json);

// O Flash Attention **resolvido** deste contexto, lido de `get_cparams()`.
//
// `llama_context_params::flash_attn_type` é o pedido, e o padrão é `AUTO`: o
// llama.cpp sonda o backend na criação do contexto e decide. Nada no
// `llama.h` público devolve a decisão, então derivar a resposta do pedido
// mentiria sempre que a sonda recusasse o kernel. `out_auto` diz se a decisão
// veio da sonda (`AUTO`) ou do pedido explícito do operador.
llama_rs_status llama_rs_context_flash_attn(
    const struct llama_context * ctx,
    bool * out_enabled,
    bool * out_auto);

// Lista os dispositivos registrados pelos backends do ggml, incluindo a
// descricao comercial e o identificador PCI quando o backend os conhece.
// O chamador e dono da string e a libera com `llama_rs_string_free`.
llama_rs_status llama_rs_backend_devices_json(char ** out_json);

// CPU reference for the differential MoE proof. It uses the tensors loaded by
// llama.cpp itself and returns the intermediate expert vectors in execution
// order: gate, up, activation, down. The caller owns all output buffers.
int llama_rs_moe_reference_expert(
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
    size_t output_capacity);

struct llama_rs_mtp_speculative * llama_rs_mtp_speculative_init(
    struct llama_context * ctx_tgt,
    struct llama_context * ctx_dft,
    int32_t n_max,
    int32_t n_min,
    float p_min);

void llama_rs_mtp_speculative_free(struct llama_rs_mtp_speculative * spec);

llama_rs_status llama_rs_mtp_speculative_begin(
    struct llama_rs_mtp_speculative * spec,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count);

llama_rs_status llama_rs_mtp_speculative_process(
    struct llama_rs_mtp_speculative * spec,
    const struct llama_batch * batch);

llama_rs_status llama_rs_mtp_speculative_draft(
    struct llama_rs_mtp_speculative * spec,
    llama_pos n_past,
    llama_token id_last,
    const llama_token * prompt_tokens,
    size_t prompt_tokens_count,
    llama_token * out_tokens,
    size_t out_tokens_capacity,
    size_t * out_tokens_count);

llama_rs_status llama_rs_mtp_speculative_accept(
    struct llama_rs_mtp_speculative * spec,
    uint16_t n_accepted);

void llama_rs_string_free(char * ptr);

#ifdef __cplusplus
}
#endif
