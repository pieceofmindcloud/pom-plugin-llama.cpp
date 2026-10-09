//! See [llama-cpp-2](https://crates.io/crates/llama-cpp-2) for a documented and safe API.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unpredictable_function_pointer_comparisons)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// The x86 instruction sets the linked ggml CPU backend was compiled to use,
/// spelled as `is_x86_feature_detected!` spells them.
///
/// ggml does not check the processor before using them: on a machine without
/// one of these the first vector instruction kills the process with SIGILL,
/// already inside `llama_backend_init`. Empty when the CPU backend is not
/// linked into this binary (it is loaded at run time and chooses its own
/// variant) and outside x86-64.
pub fn ggml_cpu_built_with() -> Vec<&'static str> {
    #[cfg(all(ggml_cpu_linked, target_arch = "x86_64"))]
    {
        let probes: [(&'static str, unsafe extern "C" fn() -> ::std::os::raw::c_int); 9] = [
            ("sse3", ggml_cpu_has_sse3),
            ("ssse3", ggml_cpu_has_ssse3),
            ("avx", ggml_cpu_has_avx),
            ("avx2", ggml_cpu_has_avx2),
            ("bmi2", ggml_cpu_has_bmi2),
            ("f16c", ggml_cpu_has_f16c),
            ("fma", ggml_cpu_has_fma),
            ("avx512f", ggml_cpu_has_avx512),
            ("avx512vnni", ggml_cpu_has_avx512_vnni),
        ];
        // Each probe returns a constant fixed at compile time and runs no
        // vector instruction, so it is safe on any processor.
        probes
            .into_iter()
            .filter(|(_, probe)| unsafe { probe() } != 0)
            .map(|(name, _)| name)
            .collect()
    }
    #[cfg(not(all(ggml_cpu_linked, target_arch = "x86_64")))]
    {
        Vec::new()
    }
}
