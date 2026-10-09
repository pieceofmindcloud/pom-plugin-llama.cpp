#!/usr/bin/env python3
"""Gera um GGUF qwen4exp minimo e estruturalmente valido, para a tarefa
sl-mtp-draft-attach (ver o `## Defeitos anotados` de docs/subsystems/inference.md
para a causa raiz e o conserto).

mode="draft": espelha o rascunho externo real da Unsloth ("shared MTP
module") - so o ultimo bloco (indice n_layer_all - 1) mais
token_embd/output/nextn.*, sem os tensores de nivel de modelo
output_hc_{norm,down,up}. Antes do porte do padrao qwen35/deepseek4 para
qwen4exp.cpp (mtp_only + layer.nextn.* + graph_mtp), esse arquivo falhava
com "check_tensor_dims: tensor 'output_hc_norm.weight' not found", igual ao
log da ksia; depois do porte ele carrega e um contexto ctx_type=MTP sobre
ele reserva o grafo `graph_mtp` normalmente - ver o teste
`a_qwen4exp_draft_without_output_hc_attaches` em
`crates/llama-engine/src/llm/model.rs`.

mode="draft_fixed": igual ao anterior, mas com output_hc_{norm,down,up}
tambem presentes - variante historica usada para descartar a hipotese de
que bastava exportar mais tensores (nao bastava: o tronco blk.0..n_layer-1
tambem precisava do gate mtp_only, ausente antes do porte).

O fixture `qwen4exp-external-draft-without-output-hc.gguf` neste diretorio e a
saida deste script em mode="draft"; regenere com:

    python3 generate_synthetic_gguf.py qwen4exp-external-draft-without-output-hc.gguf draft

Requer o gguf-py do fork vendorizado no PYTHONPATH e numpy instalado.
"""
import sys
sys.path.insert(0, "/home/tiago/projects/rdma-p2p/llama.cpp/gguf-py")

import numpy as np
import gguf

N_EMBD = 8
N_VOCAB = 6
HC = 2
HC_LOW_RANK = 4
HC_DIM = HC * N_EMBD
SSM_D_CONV = 2
SSM_D_INNER = 4
SSM_D_STATE = 2
SSM_DT_RANK = 2
SSM_N_GROUP = 1
IDX_HEADS = 1
IDX_HEAD_DIM = 2
IDX_TOPK = 1
N_HEAD = 2
N_HEAD_KV = 2
N_EMBD_HEAD = N_EMBD // N_HEAD
N_FF = 8
N_FF_SHEXP = 8
N_EXPERT = 2
N_EXPERT_USED = 1
BLOCK_COUNT = 2          # n_layer_all: blk.0 (trunk-only) + blk.1 (nextn block)
NEXTN_PREDICT_LAYERS = 1  # n_layer() = BLOCK_COUNT - NEXTN_PREDICT_LAYERS = 1


def write(path: str, mode: str):
    w = gguf.GGUFWriter(path, "qwen4exp")
    w.add_type(gguf.GGUFType.MODEL)
    w.add_name("pom-synthetic-qwen4exp")
    w.add_vocab_size(N_VOCAB)
    w.add_context_length(64)
    w.add_embedding_length(N_EMBD)
    w.add_block_count(BLOCK_COUNT)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_rope_dimension_count(N_EMBD_HEAD)
    w.add_rope_dimension_sections([1, 1, 1, 1])
    w.add_file_type(gguf.LlamaFileType.ALL_F32)

    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_EXPERT_USED)
    w.add_expert_shared_feed_forward_length(N_FF_SHEXP)

    w.add_ssm_conv_kernel(SSM_D_CONV)
    w.add_ssm_inner_size(SSM_D_INNER)
    w.add_ssm_state_size(SSM_D_STATE)
    w.add_ssm_time_step_rank(SSM_DT_RANK)
    w.add_ssm_group_count(SSM_N_GROUP)

    w.add_hyper_connection_count(HC)
    w.add_hyper_connection_low_rank(HC_LOW_RANK)

    w.add_indexer_head_count(IDX_HEADS)
    w.add_indexer_key_length(IDX_HEAD_DIM)
    w.add_indexer_top_k(IDX_TOPK)

    w.add_uint32("qwen4exp.nextn_predict_layers", NEXTN_PREDICT_LAYERS)

    w.add_tokenizer_model("none")
    w.add_token_list([f"<t{i}>" for i in range(N_VOCAB)])
    w.add_token_types([int(gguf.TokenType.NORMAL)] * N_VOCAB)

    # --- tensores --------------------------------------------------------
    # `ne(*dims)` recebe os `dims` colados literalmente do `{...}` de
    # qwen4exp.cpp (mesma ordem) e inverte por dentro para numpy - conferido
    # contra o proprio "expected"/"got" do erro do loader nativo com um
    # fixture fresco (o GGUFWriter grava dims[] invertido e o llama.cpp le
    # dims[] de novo sem inverter, entao ne[] == reversed(numpy.shape)).
    def ne(*dims):
        return np.zeros(tuple(reversed(dims)), dtype=np.float32)

    w.add_tensor("token_embd.weight", ne(N_EMBD, N_VOCAB))
    w.add_tensor("output.weight", ne(N_EMBD, N_VOCAB))

    if mode in ("draft_fixed",):
        w.add_tensor("output_hc_norm.weight", ne(N_EMBD, HC))
        w.add_tensor("output_hc_down.weight", ne(HC_DIM, HC_LOW_RANK))
        w.add_tensor("output_hc_up.weight", ne(HC_LOW_RANK, HC_DIM))

    il = BLOCK_COUNT - 1  # 1: o unico bloco presente, espelha "blk.48" no rascunho real
    key_dim = SSM_D_STATE * SSM_N_GROUP
    value_dim = SSM_D_STATE * SSM_DT_RANK
    conv_dim = key_dim * 2 + value_dim

    def blk(name, *dims):
        w.add_tensor(f"blk.{il}.{name}", ne(*dims))

    blk("attn_k.weight", N_EMBD, N_EMBD_HEAD * N_HEAD_KV)
    blk("attn_k_norm.weight", N_EMBD_HEAD)
    blk("attn_output.weight", N_EMBD_HEAD * N_HEAD, N_EMBD)
    blk("attn_q.weight", N_EMBD, N_EMBD_HEAD * N_HEAD * 2)
    blk("attn_q_norm.weight", N_EMBD_HEAD)
    blk("attn_v.weight", N_EMBD, N_EMBD_HEAD * N_HEAD_KV)
    blk("ffn_down_exps.weight", N_FF, N_EMBD, N_EXPERT)
    blk("ffn_down_shexp.weight", N_FF_SHEXP, N_EMBD)
    blk("ffn_gate_exps.weight", N_EMBD, N_FF, N_EXPERT)
    blk("ffn_gate_inp.weight", N_EMBD, N_EXPERT)
    blk("ffn_gate_inp_shexp.weight", N_EMBD)
    blk("ffn_gate_shexp.weight", N_EMBD, N_FF_SHEXP)
    blk("ffn_up_exps.weight", N_EMBD, N_FF, N_EXPERT)
    blk("ffn_up_shexp.weight", N_EMBD, N_FF_SHEXP)
    blk("hc_attn_down.weight", HC_DIM, HC_LOW_RANK)
    blk("hc_attn_inject.weight", HC_DIM, HC)
    blk("hc_attn_norm.weight", N_EMBD, HC)  # TENSOR_ALLOW_RESHAPE
    blk("hc_attn_up.weight", HC_LOW_RANK, HC_DIM)
    blk("hc_ffn_down.weight", HC_DIM, HC_LOW_RANK)
    blk("hc_ffn_inject.weight", HC_DIM, HC)
    blk("hc_ffn_norm.weight", N_EMBD, HC)  # TENSOR_ALLOW_RESHAPE
    blk("hc_ffn_up.weight", HC_LOW_RANK, HC_DIM)
    blk("indexer.k_norm.weight", IDX_HEAD_DIM)
    blk("indexer.k_proj.weight", N_EMBD, IDX_HEAD_DIM)
    blk("indexer.q_norm.weight", IDX_HEAD_DIM)
    blk("indexer.q_proj.weight", N_EMBD, IDX_HEADS * IDX_HEAD_DIM)
    blk("nextn.eh_proj.weight", 2 * N_EMBD, N_EMBD)
    blk("nextn.enorm.weight", N_EMBD)
    blk("nextn.hc_head_down.weight", HC_DIM, HC_LOW_RANK)
    blk("nextn.hc_head_norm.weight", N_EMBD, HC)  # TENSOR_ALLOW_RESHAPE
    blk("nextn.hc_head_up.weight", HC_LOW_RANK, HC_DIM)
    blk("nextn.hnorm.weight", HC_DIM)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    out, mode = sys.argv[1], sys.argv[2]
    write(out, mode)
    print(f"wrote {out} (mode={mode})")
