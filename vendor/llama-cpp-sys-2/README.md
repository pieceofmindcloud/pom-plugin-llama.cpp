# llama-cpp-sys

Raw bindings to llama.cpp with cuda support.

See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a safe API.

O `wrapper_common` também expõe as sondagens que o POM usa para ajustar a
memória: a reserva seca do segundo contexto MTP por dispositivo e o diagnóstico
dos buffers de contexto. O recuo para RAM usa um modelo irmão CPU-only para
manter todo o grafo `nextn` no host e desliga o offload do cache MTP. Um
override apenas nos tensores `nextn` foi investigado, mas não basta porque o
scheduler pode seguir a colocação da camada e do output head. Isto pertence à
ponte do POM.

O checkout local do fork também contém a implementação nativa dos modelos
híbridos. Em particular, Kimi-K3 precisa declarar suporte a rollback recorrente
e gravar as janelas de convolução nos snapshots `n_rs_seq`; sem isso, o contexto
zera o pedido de rollback e uma geração especulativa pode deixar o estado
híbrido incompatível com os ubatches seguintes. A regressão fica em
`llama.cpp/tests/test-hybrid-memory-batch.cpp` e é registrada no CTest como
`test-hybrid-memory-batch-kimi-k3`. Alterações nesse checkout continuam exigindo
autorização explícita, como descrito no `AGENTS.md` deste diretório.
