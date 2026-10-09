#![cfg_attr(test, allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

const GENERATED_FILE: &str = "mtp_graph_architectures.rs";
// O fork atual tem treze verificações efetivas, uma por arquitetura. Menos que
// isso é tratado como quebra do extrator, não como uma capacidade upstream que
// possa desligar o MTP silenciosamente. O piso não fornece nomes: os nomes
// continuam vindo integralmente das fontes do llama.cpp. Se o upstream remover
// MTP de forma intencional, o build deve parar para que essa mudança seja
// auditada em vez de aceitá-la silenciosamente.
const MIN_EXPECTED_MTP_ARCHITECTURES: usize = 13;
const MTP_GRAPH_TYPE: &str = "LLM_GRAPH_TYPE_DECODER_MTP";

pub(crate) fn llama_cpp_source(
    workspace_root: &Path,
    prepared_source: Option<&Path>,
    source_override: Option<&Path>,
) -> PathBuf {
    prepared_source
        .or(source_override)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| workspace_root.join("vendor/llama-cpp-sys-2/llama.cpp"))
}

#[derive(Debug, PartialEq, Eq)]
enum Token {
    Identifier(String),
    StringLiteral(String),
    Punctuation(u8),
}

pub(crate) fn generate(llama_cpp_root: &Path, out_dir: &Path) {
    ensure_llama_cpp_root(llama_cpp_root).unwrap_or_else(|error| panic!("{}", error));

    let source_root = llama_cpp_root.join("src");
    let models_root = source_root.join("models");
    let dispatch_path = source_root.join("llama-model.cpp");
    let names_path = source_root.join("llama-arch.cpp");

    println!("cargo:rerun-if-changed={}", models_root.display());
    println!("cargo:rerun-if-changed={}", dispatch_path.display());
    println!("cargo:rerun-if-changed={}", names_path.display());

    let model_sources = read_model_sources(&models_root);
    let model_dispatch = read_required_source(&dispatch_path);
    let architecture_names = read_required_source(&names_path);
    let architectures =
        extract_mtp_architectures(&model_sources, &model_dispatch, &architecture_names)
            .unwrap_or_else(|error| {
                panic!(
                    "não foi possível derivar as arquiteturas MTP do llama.cpp em {}: {error}",
                    llama_cpp_root.display()
                )
            });

    fs::write(
        out_dir.join(GENERATED_FILE),
        render_generated_file(&architectures),
    )
    .unwrap_or_else(|error| {
        panic!(
            "não foi possível escrever {}: {error}",
            out_dir.join(GENERATED_FILE).display()
        )
    });
}

fn ensure_llama_cpp_root(llama_cpp_root: &Path) -> Result<(), String> {
    if llama_cpp_root.is_dir() {
        return Ok(());
    }
    Err(format!(
        "llama.cpp não encontrado em {}; a lista de arquiteturas MTP não pode ser gerada",
        llama_cpp_root.display()
    ))
}

fn read_model_sources(models_root: &Path) -> Vec<(PathBuf, String)> {
    let mut paths = fs::read_dir(models_root)
        .unwrap_or_else(|error| {
            panic!(
                "não foi possível ler a pasta de modelos do llama.cpp em {}: {error}",
                models_root.display()
            )
        })
        .map(|entry| {
            entry
                .unwrap_or_else(|error| {
                    panic!("não foi possível ler uma entrada de models: {}", error)
                })
                .path()
        })
        .filter(|path| path.extension().is_some_and(|extension| extension == "cpp"))
        .collect::<Vec<_>>();
    paths.sort();

    paths
        .into_iter()
        .map(|path| {
            let source = read_required_source(&path);
            (path, source)
        })
        .collect()
}

fn read_required_source(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("não foi possível ler {}: {error}", path.display()))
}

pub(crate) fn extract_mtp_architectures(
    model_sources: &[(PathBuf, String)],
    model_dispatch: &str,
    architecture_names: &str,
) -> Result<Vec<String>, String> {
    if model_sources.is_empty() {
        return Err(
            "nenhuma verificação efetiva de LLM_GRAPH_TYPE_DECODER_MTP foi encontrada em src/models"
                .to_string(),
        );
    }
    let dispatch = parse_model_dispatch(model_dispatch)?;
    let names = parse_architecture_names(architecture_names)?;
    let mut architectures = BTreeSet::new();
    let mut checked_files = 0usize;

    for (path, source) in model_sources {
        let tokens = tokenize(source)?;
        let check_positions = mtp_check_positions(&tokens)?;
        if check_positions.is_empty() {
            continue;
        }
        checked_files += 1;

        let graph_functions = graph_functions(&tokens);
        let checked_functions = graph_functions
            .iter()
            .filter_map(|function| {
                let checks = check_positions
                    .iter()
                    .filter(|&&position| {
                        position > function.body_start && position < function.body_end
                    })
                    .count();
                (checks > 0).then_some((function, checks))
            })
            .collect::<Vec<_>>();
        let associated_checks = checked_functions
            .iter()
            .map(|(_, checks)| *checks)
            .sum::<usize>();
        if associated_checks != check_positions.len() {
            return Err(format!(
                "{} contém uma verificação efetiva de MTP fora de um escopo build_arch_graph",
                path.display(),
            ));
        }
        if checked_functions.len() != 1 || associated_checks != 1 {
            return Err(format!(
                "{} contém {} verificações MTP em {} funções build_arch_graph; a associação não é inequívoca",
                path.display(),
                associated_checks,
                checked_functions.len()
            ));
        }
        let class = &checked_functions[0].0.class;
        let arch = dispatch.get(class).ok_or_else(|| {
            format!(
                "a classe {class} de {} não foi encontrada no dispatch de llama-model.cpp",
                path.display()
            )
        })?;
        let name = names.get(arch).ok_or_else(|| {
            format!(
                "a arquitetura {arch}, usada pela classe {class}, não foi encontrada na tabela de llama-arch.cpp"
            )
        })?;
        if !architectures.insert(name.clone()) {
            return Err(format!(
                "a arquitetura oficial {name} foi encontrada em mais de um arquivo com grafo MTP"
            ));
        }
    }

    if checked_files == 0 {
        return Err(
            "nenhuma verificação efetiva de LLM_GRAPH_TYPE_DECODER_MTP foi encontrada em src/models"
                .to_string(),
        );
    }

    let architectures = architectures.into_iter().collect::<Vec<_>>();
    validate_mtp_architectures(&architectures)?;
    Ok(architectures)
}

pub(crate) fn validate_mtp_architectures(architectures: &[String]) -> Result<(), String> {
    if architectures.len() < MIN_EXPECTED_MTP_ARCHITECTURES {
        return Err(format!(
            "extração suspeita: {} arquiteturas MTP encontradas; mínimo esperado é {}",
            architectures.len(),
            MIN_EXPECTED_MTP_ARCHITECTURES
        ));
    }
    if architectures
        .iter()
        .any(|architecture| architecture.is_empty())
    {
        return Err("extração suspeita: nome de arquitetura MTP vazio".to_string());
    }
    let unique = architectures.iter().collect::<BTreeSet<_>>();
    if unique.len() != architectures.len() {
        return Err("extração suspeita: nomes de arquitetura MTP duplicados".to_string());
    }
    Ok(())
}

fn render_generated_file(architectures: &[String]) -> String {
    let mut output =
        String::from("// Gerado por crates/llama-engine/build/mtp_arch.rs; não edite.\n");
    output.push_str("pub(crate) const GENERATED_MTP_GRAPH_ARCHITECTURES: &[&str] = &[\n");
    for architecture in architectures {
        writeln!(output, "    {architecture:?},").expect("String não falha ao escrever");
    }
    output.push_str("];\n");
    output
}

fn tokenize(source: &str) -> Result<Vec<Token>, String> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0usize;

    while index < bytes.len() {
        match bytes[index] {
            byte if byte.is_ascii_whitespace() => index += 1,
            b'/' if bytes.get(index + 1) == Some(&b'/') => {
                index += 2;
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                index += 2;
                let start = index;
                while index + 1 < bytes.len() && !(bytes[index] == b'*' && bytes[index + 1] == b'/')
                {
                    index += 1;
                }
                if index + 1 >= bytes.len() {
                    return Err(format!("comentário C++ não terminado na posição {start}"));
                }
                index += 2;
            }
            b'"' => {
                let (literal, next) = read_quoted(bytes, index, b'"')?;
                tokens.push(Token::StringLiteral(literal));
                index = next;
            }
            b'\'' => {
                let (_, next) = read_quoted(bytes, index, b'\'')?;
                index = next;
            }
            byte if byte.is_ascii_alphabetic() || byte == b'_' => {
                let start = index;
                index += 1;
                while index < bytes.len()
                    && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
                {
                    index += 1;
                }
                tokens.push(Token::Identifier(source[start..index].to_string()));
            }
            byte => {
                tokens.push(Token::Punctuation(byte));
                index += 1;
            }
        }
    }

    Ok(tokens)
}

fn read_quoted(bytes: &[u8], start: usize, quote: u8) -> Result<(String, usize), String> {
    let mut index = start + 1;
    let mut value = String::new();
    while index < bytes.len() {
        match bytes[index] {
            byte if byte == quote => return Ok((value, index + 1)),
            b'\\' => {
                let Some(&escaped) = bytes.get(index + 1) else {
                    break;
                };
                value.push(escaped as char);
                index += 2;
            }
            byte => {
                value.push(byte as char);
                index += 1;
            }
        }
    }
    Err(format!(
        "literal entre aspas não terminado na posição {start}"
    ))
}

fn mtp_check_positions(tokens: &[Token]) -> Result<Vec<usize>, String> {
    let marker_positions = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| is_identifier(Some(token), MTP_GRAPH_TYPE).then_some(index))
        .collect::<Vec<_>>();
    let mut check_positions = Vec::with_capacity(marker_positions.len());

    for marker_position in marker_positions {
        let start = marker_position
            .checked_sub(7)
            .filter(|&start| is_effective_mtp_check_start(tokens, start));
        let Some(start) = start else {
            return Err(
                "uma ocorrência efetiva de LLM_GRAPH_TYPE_DECODER_MTP não usa a forma esperada de verificação"
                    .to_string(),
            );
        };
        check_positions.push(start);
    }

    Ok(check_positions)
}

fn is_effective_mtp_check_start(tokens: &[Token], start: usize) -> bool {
    let Some(window) = tokens.get(start..start + 9) else {
        return false;
    };
    matches!(
        window,
        [
            Token::Identifier(if_keyword),
            Token::Punctuation(b'('),
            Token::Identifier(params),
            Token::Punctuation(b'.'),
            Token::Identifier(gtype),
            Token::Punctuation(b'='),
            Token::Punctuation(b'='),
            Token::Identifier(mtp_type),
            Token::Punctuation(b')'),
        ] if if_keyword == "if"
            && params == "params"
            && gtype == "gtype"
            && mtp_type == MTP_GRAPH_TYPE
    )
}

struct GraphFunction {
    class: String,
    body_start: usize,
    body_end: usize,
}

fn graph_functions(tokens: &[Token]) -> Vec<GraphFunction> {
    let mut functions = Vec::new();
    for index in 0..tokens.len().saturating_sub(3) {
        let [Token::Identifier(class), Token::Punctuation(b':'), Token::Punctuation(b':'), Token::Identifier(function)] =
            &tokens[index..index + 4]
        else {
            continue;
        };
        if !class.starts_with("llama_model_") || function != "build_arch_graph" {
            continue;
        }

        let mut cursor = index + 4;
        while cursor < tokens.len() {
            if is_punctuation(tokens.get(cursor), b';') {
                break;
            }
            if is_punctuation(tokens.get(cursor), b'{') {
                if let Some(body_end) = matching_brace(tokens, cursor) {
                    functions.push(GraphFunction {
                        class: class.clone(),
                        body_start: cursor,
                        body_end,
                    });
                }
                break;
            }
            cursor += 1;
        }
    }
    functions
}

fn parse_model_dispatch(source: &str) -> Result<BTreeMap<String, String>, String> {
    let tokens = tokenize(source)?;
    let mut dispatch = BTreeMap::new();
    for index in 0..tokens.len() {
        if !is_identifier(tokens.get(index), "case") {
            continue;
        }
        let Some(Token::Identifier(arch)) = tokens.get(index + 1) else {
            continue;
        };
        if !is_punctuation(tokens.get(index + 2), b':') {
            continue;
        }

        let mut cursor = index + 3;
        while cursor + 4 < tokens.len() && !is_identifier(tokens.get(cursor), "case") {
            if is_identifier(tokens.get(cursor), "return")
                && is_identifier(tokens.get(cursor + 1), "new")
                && is_punctuation(tokens.get(cursor + 3), b'(')
            {
                if let Token::Identifier(class) = &tokens[cursor + 2] {
                    if dispatch.insert(class.clone(), arch.clone()).is_some() {
                        return Err(format!(
                            "a classe {class} aparece mais de uma vez no dispatch de llama-model.cpp"
                        ));
                    }
                }
                break;
            }
            cursor += 1;
        }
    }

    if dispatch.is_empty() {
        return Err(
            "nenhum mapeamento case LLM_ARCH_* -> llama_model_* foi encontrado".to_string(),
        );
    }
    Ok(dispatch)
}

fn parse_architecture_names(source: &str) -> Result<BTreeMap<String, String>, String> {
    let tokens = tokenize(source)?;
    let table_name = tokens
        .iter()
        .position(|token| is_identifier(Some(token), "LLM_ARCH_NAMES"))
        .ok_or_else(|| {
            "a tabela LLM_ARCH_NAMES não foi encontrada em llama-arch.cpp".to_string()
        })?;
    let table_start = tokens[table_name..]
        .iter()
        .position(|token| is_punctuation(Some(token), b'{'))
        .map(|offset| table_name + offset)
        .ok_or_else(|| "a abertura de LLM_ARCH_NAMES não foi encontrada".to_string())?;
    let table_end = matching_brace(&tokens, table_start)
        .ok_or_else(|| "a tabela LLM_ARCH_NAMES não foi encerrada".to_string())?;
    let mut names = BTreeMap::new();
    for window in tokens[table_start..=table_end].windows(5) {
        if let [Token::Punctuation(b'{'), Token::Identifier(arch), Token::Punctuation(b','), Token::StringLiteral(name), Token::Punctuation(b'}')] =
            window
        {
            if names.insert(arch.clone(), name.clone()).is_some() {
                return Err(format!(
                    "a arquitetura {arch} aparece duas vezes em LLM_ARCH_NAMES"
                ));
            }
        }
    }

    if names.is_empty() {
        return Err(
            "nenhuma entrada { LLM_ARCH_*, nome } foi encontrada em llama-arch.cpp".to_string(),
        );
    }
    Ok(names)
}

fn matching_brace(tokens: &[Token], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(start) {
        if is_punctuation(Some(token), b'{') {
            depth += 1;
        } else if is_punctuation(Some(token), b'}') {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index);
            }
        }
    }
    None
}

fn is_identifier(token: Option<&Token>, expected: &str) -> bool {
    matches!(token, Some(Token::Identifier(value)) if value == expected)
}

fn is_punctuation(token: Option<&Token>, expected: u8) -> bool {
    matches!(token, Some(Token::Punctuation(value)) if *value == expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_the_vendored_checkout_by_default() {
        let workspace_root = Path::new(r"D:\a\rdma-p2p\rdma-p2p");

        assert_eq!(
            llama_cpp_source(workspace_root, None, None),
            workspace_root
                .join("vendor")
                .join("llama-cpp-sys-2")
                .join("llama.cpp")
        );
    }

    #[test]
    fn respects_an_explicit_llama_source() {
        let workspace_root = Path::new(r"D:\a\rdma-p2p\rdma-p2p");
        let override_source = Path::new(r"E:\forks\llama.cpp");

        assert_eq!(
            llama_cpp_source(workspace_root, None, Some(override_source)),
            override_source
        );
    }

    #[test]
    fn uses_the_source_prepared_by_the_dependency() {
        let workspace_root = Path::new(r"D:\a\rdma-p2p\rdma-p2p");
        let prepared_source = Path::new(r"D:\a\rdma-p2p\rdma-p2p\vendor\llama-cpp-sys-2\llama.cpp");
        let source_override = Path::new(r"E:\forks\llama.cpp");

        assert_eq!(
            llama_cpp_source(workspace_root, Some(prepared_source), Some(source_override)),
            prepared_source
        );
    }

    #[test]
    fn missing_llama_cpp_fails_loudly() {
        let error = ensure_llama_cpp_root(Path::new("/caminho/que/nao/existe/llama.cpp"))
            .expect_err("um fork ausente não pode gerar uma lista vazia");

        assert!(
            error.contains("llama.cpp não encontrado"),
            "erro: {}",
            error
        );
    }

    fn fixture_sources() -> Vec<(PathBuf, String)> {
        IntoIterator::into_iter([
            ("bailingmoe3.cpp", "bailingmoe3", "BAILINGMOE3"),
            ("cohere2moe.cpp", "cohere2moe", "COHERE2MOE"),
            ("deepseek2.cpp", "deepseek2", "DEEPSEEK2"),
            ("deepseek32.cpp", "deepseek32", "DEEPSEEK32"),
            ("deepseek4.cpp", "deepseek4", "DEEPSEEK4"),
            ("glm-dsa.cpp", "glm_dsa", "GLM_DSA"),
            ("hy-v3.cpp", "hy_v3", "HY_V3"),
            ("mimo2.cpp", "mimo2", "MIMO2"),
            ("nemotron-h-moe.cpp", "nemotron_h_moe", "NEMOTRON_H_MOE"),
            ("qwen35.cpp", "qwen35", "QWEN35"),
            ("qwen35moe.cpp", "qwen35moe", "QWEN35MOE"),
            ("qwen3next.cpp", "qwen3next", "QWEN3NEXT"),
            ("step35.cpp", "step35", "STEP35"),
        ])
        .map(|(file, class, _)| {
            (
                PathBuf::from(file),
                format!(
                    "std::unique_ptr<llm_graph_context> llama_model_{class}::build_arch_graph(\n\
                        const llm_graph_params & params) const {{\n\
                        if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {{\n\
                            return std::make_unique<graph_mtp>(*this, params);\n\
                        }}\n\
                        return std::make_unique<graph>(*this, params);\n\
                    }}"
                ),
            )
        })
        .collect()
    }

    fn fixture_dispatch() -> &'static str {
        r#"
            case LLM_ARCH_BAILINGMOE3:
                return new llama_model_bailingmoe3(params);
            case LLM_ARCH_COHERE2MOE:
                return new llama_model_cohere2moe(params);
            case LLM_ARCH_DEEPSEEK2:
                return new llama_model_deepseek2(params);
            case LLM_ARCH_DEEPSEEK32:
                return new llama_model_deepseek32(params);
            case LLM_ARCH_DEEPSEEK4:
                return new llama_model_deepseek4(params);
            case LLM_ARCH_GLM_DSA:
                return new llama_model_glm_dsa(params);
            case LLM_ARCH_HY_V3:
                return new llama_model_hy_v3(params);
            case LLM_ARCH_MIMO2:
                return new llama_model_mimo2(params);
            case LLM_ARCH_NEMOTRON_H_MOE:
                return new llama_model_nemotron_h_moe(params);
            case LLM_ARCH_QWEN35:
                return new llama_model_qwen35(params);
            case LLM_ARCH_QWEN35MOE:
                return new llama_model_qwen35moe(params);
            case LLM_ARCH_QWEN3NEXT:
                return new llama_model_qwen3next(params);
            case LLM_ARCH_STEP35:
                return new llama_model_step35(params);
        "#
    }

    fn fixture_names() -> &'static str {
        r#"
            static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
            { LLM_ARCH_BAILINGMOE3, "bailingmoe3" },
            { LLM_ARCH_COHERE2MOE, "cohere2moe" },
            { LLM_ARCH_DEEPSEEK2, "deepseek2" },
            { LLM_ARCH_DEEPSEEK32, "deepseek32" },
            { LLM_ARCH_DEEPSEEK4, "deepseek4" },
            { LLM_ARCH_GLM_DSA, "glm-dsa" },
            { LLM_ARCH_HY_V3, "hy_v3" },
            { LLM_ARCH_MIMO2, "mimo2" },
            { LLM_ARCH_NEMOTRON_H_MOE, "nemotron_h_moe" },
            { LLM_ARCH_QWEN35, "qwen35" },
            { LLM_ARCH_QWEN35MOE, "qwen35moe" },
            { LLM_ARCH_QWEN3NEXT, "qwen3next" },
            { LLM_ARCH_STEP35, "step35" },
            };
        "#
    }

    #[test]
    fn extracts_official_names_from_the_dispatch_and_the_table() {
        let names =
            extract_mtp_architectures(&fixture_sources(), fixture_dispatch(), fixture_names())
                .expect("fontes MTP válidas devem ser extraídas");

        assert_eq!(
            names,
            vec![
                "bailingmoe3".to_string(),
                "cohere2moe".to_string(),
                "deepseek2".to_string(),
                "deepseek32".to_string(),
                "deepseek4".to_string(),
                "glm-dsa".to_string(),
                "hy_v3".to_string(),
                "mimo2".to_string(),
                "nemotron_h_moe".to_string(),
                "qwen35".to_string(),
                "qwen35moe".to_string(),
                "qwen3next".to_string(),
                "step35".to_string(),
            ]
        );
    }

    #[test]
    fn a_mention_in_a_comment_does_not_count_as_a_check() {
        let mut sources = fixture_sources();
        sources.push((
            PathBuf::from("comentario.cpp"),
            "// if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP)\n".to_string(),
        ));

        let names = extract_mtp_architectures(&sources, fixture_dispatch(), fixture_names())
            .expect("uma menção em comentário não deve alterar a extração");

        assert_eq!(names.len(), 13);
        assert!(names.iter().any(|name| name == "hy_v3"));
    }

    #[test]
    fn a_check_outside_build_arch_graph_fails() {
        let mut sources = fixture_sources();
        sources.push((
            PathBuf::from("fora-do-grafo.cpp"),
            "void helper(const llm_graph_params & params) {\n\
                if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {}\n\
            }"
            .to_string(),
        ));

        let error = extract_mtp_architectures(&sources, fixture_dispatch(), fixture_names())
            .expect_err("uma verificação fora do grafo não deve ser associada por arquivo");

        assert!(error.contains("build_arch_graph"), "erro: {}", error);
    }

    #[test]
    fn a_duplicate_dispatch_fails_loudly() {
        let error = parse_model_dispatch(
            "case LLM_ARCH_QWEN35: return new llama_model_qwen35(params);\n\
             case LLM_ARCH_QWEN35MOE: return new llama_model_qwen35(params);",
        )
        .expect_err("uma classe repetida no dispatch é ambígua");

        assert!(error.contains("mais de uma vez"), "erro: {}", error);
    }

    #[test]
    fn an_empty_extraction_fails_loudly() {
        let error =
            extract_mtp_architectures(&[], "", "").expect_err("lista vazia não pode ser aceita");

        assert!(error.contains("nenhuma"), "erro: {}", error);
    }

    #[test]
    fn a_visibly_small_extraction_fails_loudly() {
        let error = validate_mtp_architectures(&[
            "qwen3next".to_string(),
            "qwen35".to_string(),
            "qwen35moe".to_string(),
        ])
        .expect_err("a antiga lista de três arquiteturas é pequena demais");

        assert!(error.contains("suspeita"), "erro: {}", error);
    }
}
