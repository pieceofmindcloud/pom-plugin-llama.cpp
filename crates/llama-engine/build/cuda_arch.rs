pub fn nvcc_code_generations(architectures: &str) -> Result<Vec<String>, String> {
    architectures
        .split(';')
        .map(str::trim)
        .filter(|arch| !arch.is_empty())
        .map(|raw| {
            let (raw, virtual_only) = if let Some(arch) = raw.strip_suffix("-virtual") {
                (arch, true)
            } else if let Some(arch) = raw.strip_suffix("-real") {
                (arch, false)
            } else {
                (raw, false)
            };
            let arch = match raw {
                "120" | "121" => format!("{raw}a"),
                _ => raw.to_owned(),
            };
            let digits = arch.strip_suffix('a').unwrap_or(&arch);
            if digits.is_empty() || !digits.chars().all(|digit| digit.is_ascii_digit()) {
                return Err(format!("unsupported CUDA architecture `{raw}`"));
            }
            if virtual_only && arch.ends_with('a') {
                return Err(format!(
                    "architecture-specific CUDA target `{arch}` has no PTX variant"
                ));
            }
            let code = if virtual_only {
                format!("compute_{arch}")
            } else {
                format!("sm_{arch}")
            };
            Ok(format!("arch=compute_{arch},code={code}"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::nvcc_code_generations;

    #[test]
    fn emits_one_nvcc_target_for_each_architecture() {
        assert_eq!(
            nvcc_code_generations("86;120").unwrap(),
            [
                "arch=compute_86,code=sm_86",
                "arch=compute_120a,code=sm_120a",
            ]
        );
    }

    #[test]
    fn supports_real_and_virtual_targets_and_rejects_invalid_architectures() {
        assert_eq!(
            nvcc_code_generations("86-real;89-virtual").unwrap(),
            [
                "arch=compute_86,code=sm_86",
                "arch=compute_89,code=compute_89",
            ]
        );
        assert!(nvcc_code_generations("120-virtual").is_err());
        assert!(nvcc_code_generations("native").is_err());
    }
}
