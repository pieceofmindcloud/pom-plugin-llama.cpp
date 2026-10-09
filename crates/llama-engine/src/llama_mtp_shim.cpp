// Pontes C para as entradas NextN/MTP do llama.cpp.
//
// As funções de NextN moram em `llama.cpp/src/llama-ext.h`, o cabeçalho de
// estágio ("breaking changes and C++ are allowed"), **fora** do bloco
// `extern "C"` de `llama.h`. O bindgen do `llama-cpp-sys-2` lê só
// `wrapper.h` → `llama.h`, então nunca as emitiu: elas não existem em
// `llama_cpp_sys_2::`. Os símbolos existem no `libllama.a` - conferido com
// `nm -g libllama.a | grep nextn` - apenas com o nome mangled do C++.
//
// Redeclarar os protótipos aqui e reexportá-los com `extern "C"` deixa o
// próprio compilador C++ fazer o mangling. A alternativa era escrever
// `_Z26llama_set_embeddings_nextnP13llama_contextbb` à mão num
// `#[link_name]` do Rust, que ninguém consegue conferir por leitura e que
// quebra em qualquer plataforma cujo ABI não seja o Itanium (MSVC).
//
// Nenhum cabeçalho do llama.cpp é incluído de propósito: uma declaração
// antecipada de `llama_context` basta para formar a chamada, e assim o build
// não precisa descobrir onde a crate `-sys` guardou os includes (ela não
// exporta esse caminho). O preço é que as assinaturas abaixo têm que casar
// com `llama-ext.h`; divergir dá **erro de link**, alto e na hora, nunca
// resultado errado em silêncio.

struct llama_context;

void    llama_set_embeddings_nextn  (llama_context * ctx, bool value, bool masked);
void    llama_set_nextn_layer_offset(llama_context * ctx, int offset);
float * llama_get_embeddings_nextn  (llama_context * ctx);
float * llama_get_embeddings_nextn_ith(llama_context * ctx, int i);

extern "C" {

void pom_llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    llama_set_embeddings_nextn(ctx, value, masked);
}

void pom_llama_set_nextn_layer_offset(llama_context * ctx, int offset) {
    llama_set_nextn_layer_offset(ctx, offset);
}

float * pom_llama_get_embeddings_nextn(llama_context * ctx) {
    return llama_get_embeddings_nextn(ctx);
}

float * pom_llama_get_embeddings_nextn_ith(llama_context * ctx, int i) {
    return llama_get_embeddings_nextn_ith(ctx, i);
}

} // extern "C"
