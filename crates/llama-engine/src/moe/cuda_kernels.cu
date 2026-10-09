#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <limits>
#include <new>
#include <vector>

// Os layouts e codebooks IQ são parte da convenção GGML. Incluí-los aqui
// evita uma segunda tabela que poderia divergir do decoder de referência;
// o executor e os kernels abaixo continuam sendo específicos deste motor.
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
// build.rs points this include at the exact fork selected for the build.
#include "ggml-common.h"
// Reuse llama.cpp's canonical IQ3_XXS q8_1/dp4a decoder. The include root is
// supplied by build.rs from the active fork, so LLAMA_SOURCE remains valid.
#include "ggml-cuda/vecdotq.cuh"

namespace {

constexpr int kBlockElements = 64;
constexpr int kSubblockElements = 16;
constexpr int kBlockBytes = 36;
constexpr int kPayloadHeaderBytes = 32;
constexpr int kKQuantPayloadHeaderBytes = 32;
constexpr int kScalarPayloadHeaderBytes = 8;
// The llama.cpp scheduler changes the current CUDA device while it moves a
// graph between backends. The external MoE hook is a host callback, so every
// CUDA object it owns must be created and used on one explicit device, and the
// callback must leave the caller's current device untouched. Otherwise the
// scheduler may wait on a CUDA event for another device while the host thread
// still has the MoE device current.
class ScopedMoeCudaDevice {
public:
    explicit ScopedMoeCudaDevice(int device_index) {
        int device_count = 0;
        if (device_index < 0 ||
            cudaGetDeviceCount(&device_count) != cudaSuccess ||
            device_index >= device_count) {
            return;
        }
        if (cudaGetDevice(&previous_device_) != cudaSuccess) {
            return;
        }
        if (previous_device_ != device_index) {
            if (cudaSetDevice(device_index) != cudaSuccess) {
                previous_device_ = -1;
                return;
            }
            restore_previous_ = true;
        }
        ready_ = true;
    }

    ~ScopedMoeCudaDevice() {
        if (restore_previous_) {
            (void) cudaSetDevice(previous_device_);
        }
    }

    bool ready() const {
        return ready_;
    }

private:
    int previous_device_ = -1;
    bool restore_previous_ = false;
    bool ready_ = false;
};

struct CudaMoeCallTiming {
    uint64_t expert_calls;
    uint64_t kernel_launches;
    uint64_t sync_count;
    double payload_validation_ms;
    double alloc_ms;
    double h2d_ms;
    uint64_t h2d_bytes;
    double kernel_ms;
    double kernel_span_ms;
    double launch_ms;
    double sync_ms;
    double d2h_ms;
    uint64_t d2h_bytes;
    double transfer_span_ms;
    double free_ms;
    uint64_t buffer_mask;
    uint64_t buffer_switches;
};

struct CudaHeterogeneousTensor {
    const uint8_t * data;
    std::size_t bytes;
    std::size_t rows;
    std::size_t columns;
    uint32_t dtype;
    float scale;
    float input_scale;
    float secondary_scale;
    float secondary_input_scale;
};

struct CudaHeterogeneousExpert {
    CudaHeterogeneousTensor gate_up;
    CudaHeterogeneousTensor gate;
    CudaHeterogeneousTensor up;
    CudaHeterogeneousTensor down;
};

static_assert(offsetof(CudaHeterogeneousTensor, data) == 0,
    "CudaHeterogeneousTensor.data ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, bytes) == 8,
    "CudaHeterogeneousTensor.bytes ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, rows) == 16,
    "CudaHeterogeneousTensor.rows ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, columns) == 24,
    "CudaHeterogeneousTensor.columns ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, dtype) == 32,
    "CudaHeterogeneousTensor.dtype ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, scale) == 36,
    "CudaHeterogeneousTensor.scale ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, input_scale) == 40,
    "CudaHeterogeneousTensor.input_scale ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, secondary_scale) == 44,
    "CudaHeterogeneousTensor.secondary_scale ABI offset differs from Rust");
static_assert(offsetof(CudaHeterogeneousTensor, secondary_input_scale) == 48,
    "CudaHeterogeneousTensor.secondary_input_scale ABI offset differs from Rust");
static_assert(sizeof(CudaHeterogeneousTensor) == 56,
    "CudaHeterogeneousTensor ABI must match its Rust descriptor");

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

__device__ float fp4_value(uint8_t value) {
    constexpr float values[16] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
        0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
    };
    return values[value & 0x0f];
}

__device__ float ue4m3_to_f32(uint8_t value) {
    if (value == 0 || value == 0x7f) {
        return 0.0f;
    }
    const int exponent = (value >> 3) & 0x0f;
    const float mantissa = static_cast<float>(value & 0x07);
    const float raw = exponent == 0
        ? mantissa * exp2f(-9.0f)
        : (1.0f + mantissa / 8.0f) * exp2f(static_cast<float>(exponent - 7));
    return raw * 0.5f;
}

__device__ __forceinline__ float round_to_f16(float value) {
    return __half2float(__float2half_rn(value));
}

__device__ __forceinline__ float payload_scale(const uint8_t * payload, int index) {
    const std::size_t offset = 8 + static_cast<std::size_t>(index) * sizeof(float);
    uint32_t bits = static_cast<uint32_t>(payload[offset]);
    bits |= static_cast<uint32_t>(payload[offset + 1]) << 8;
    bits |= static_cast<uint32_t>(payload[offset + 2]) << 16;
    bits |= static_cast<uint32_t>(payload[offset + 3]) << 24;
    return __uint_as_float(bits);
}

__global__ void quantize_q8_0_kernel(
    const float * input,
    int length,
    float * scales,
    int8_t * values,
    int vector_count
) {
    __shared__ float maxima[32];
    const int lane = threadIdx.x;
    const int blocks_per_vector = (length + 31) / 32;
    const int vector = blockIdx.x / blocks_per_vector;
    const int block = blockIdx.x % blocks_per_vector;
    if (vector >= vector_count) {
        return;
    }
    const int offset = block * 32 + lane;
    const int index = vector * length + offset;
    const float value = offset < length ? input[index] : 0.0f;
    maxima[lane] = fabsf(value);
    __syncthreads();

    for (int stride = 16; stride > 0; stride >>= 1) {
        if (lane < stride) {
            maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        }
        __syncthreads();
    }

    const float max_scalar = maxima[0];
    const float unrounded_scale = max_scalar / 127.0f;
    const float scale = round_to_f16(unrounded_scale);
    if (lane == 0) {
        scales[vector * blocks_per_vector + block] = scale;
    }
    __syncthreads();

    if (offset < length) {
        const float inverse_scale = max_scalar != 0.0f ? 127.0f / max_scalar : 0.0f;
        int quantized = __float2int_rn(input[index] * inverse_scale);
        quantized = max(-128, min(127, quantized));
        values[index] = static_cast<int8_t>(quantized);
    }
}

__device__ float dot_nvfp4(
    const uint8_t * matrix,
    int row,
    int columns,
    const int8_t * input_values,
    const float * input_scales
) {
    const int row_bytes = (columns / kBlockElements) * kBlockBytes;
    const uint8_t * encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int block = 0; block < columns / kBlockElements; ++block) {
        const uint8_t * packed = encoded + block * kBlockBytes;
        for (int subblock = 0; subblock < kBlockElements / kSubblockElements; ++subblock) {
            const float scale = ue4m3_to_f32(packed[subblock]);
            const uint8_t * values = packed + 4 + subblock * (kSubblockElements / 2);
            const int q8_block = block * 2 + subblock / 2;
            const int q8_start = q8_block * 32 + (subblock % 2) * kSubblockElements;
            const float input_scale = input_scales[q8_block];
            int sum_lo = 0;
            int sum_hi = 0;
            for (int index = 0; index < kSubblockElements / 2; ++index) {
                sum_lo += static_cast<int>(input_values[q8_start + index]) *
                    static_cast<int>(fp4_value(values[index]));
                sum_hi += static_cast<int>(input_values[q8_start + index + kSubblockElements / 2]) *
                    static_cast<int>(fp4_value(values[index] >> 4));
            }
            result += input_scale * scale * static_cast<float>(sum_lo + sum_hi);
        }
    }
    return result;
}

__global__ void gate_up_kernel(
    const uint8_t * payloads,
    std::size_t payload_stride,
    int batch,
    int intermediate,
    int hidden,
    const int8_t * hidden_values,
    const float * hidden_scales,
    float * gate_up_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows_per_expert = 2 * intermediate;
    const int expert = index / rows_per_expert;
    const int row = index % rows_per_expert;
    if (expert >= batch) {
        return;
    }
    const uint8_t * payload = payloads + expert * payload_stride;
    const uint8_t * gate_up = payload + kPayloadHeaderBytes;
    const float gate_scale = payload_scale(payload, 0);
    const float up_scale = payload_scale(payload, 1);
    const bool is_up = row >= intermediate;
    const float value = dot_nvfp4(
        gate_up,
        row,
        hidden,
        hidden_values,
        hidden_scales
    );
    gate_up_values[expert * rows_per_expert + row] = value * (is_up ? up_scale : gate_scale);
}

__global__ void gate_up_resident_kernel(
    const uint8_t * const * payloads,
    int batch,
    int intermediate,
    int hidden,
    const int8_t * hidden_values,
    const float * hidden_scales,
    float * gate_up_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows_per_expert = 2 * intermediate;
    const int expert = index / rows_per_expert;
    const int row = index % rows_per_expert;
    if (expert >= batch) {
        return;
    }
    const uint8_t * payload = payloads[expert];
    const uint8_t * gate_up = payload + kPayloadHeaderBytes;
    const float gate_scale = payload_scale(payload, 0);
    const float up_scale = payload_scale(payload, 1);
    const bool is_up = row >= intermediate;
    const float value = dot_nvfp4(
        gate_up,
        row,
        hidden,
        hidden_values,
        hidden_scales
    );
    gate_up_values[expert * rows_per_expert + row] = value * (is_up ? up_scale : gate_scale);
}

__device__ float ggml_gelu_f16(float value) {
    const float input = round_to_f16(value);
    const float gelu = 0.5f * input *
        (1.0f + tanhf(0.7978845608028654f * input *
            (1.0f + 0.044715f * input * input)));
    return round_to_f16(gelu);
}

__global__ void geglu_kernel(
    const float * gate_up_values,
    int batch,
    int intermediate,
    float * activated
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int expert = index / intermediate;
    const int row = index % intermediate;
    if (expert >= batch) {
        return;
    }
    const float * gate_up = gate_up_values + expert * 2 * intermediate;
    const float gate = gate_up[row];
    const float up = gate_up[intermediate + row];
    const float activated_gate = gate <= -10.0f
        ? 0.0f
        : gate >= 10.0f
            ? gate
            : ggml_gelu_f16(gate);
    activated[expert * intermediate + row] = activated_gate * up;
}

__global__ void down_kernel(
    const uint8_t * payloads,
    std::size_t payload_stride,
    int batch,
    int hidden,
    int intermediate,
    const int8_t * activated_values,
    const float * activated_scales,
    float * output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int expert = index / hidden;
    const int row = index % hidden;
    if (expert >= batch) {
        return;
    }
    const uint8_t * payload = payloads + expert * payload_stride;
    const std::size_t gate_up_len = static_cast<std::size_t>(payload[0]) |
        (static_cast<std::size_t>(payload[1]) << 8) |
        (static_cast<std::size_t>(payload[2]) << 16) |
        (static_cast<std::size_t>(payload[3]) << 24);
    const uint8_t * down = payload + kPayloadHeaderBytes + gate_up_len;
    output[expert * hidden + row] = dot_nvfp4(
        down,
        row,
        intermediate,
        activated_values + expert * intermediate,
        activated_scales + expert * (intermediate / 32)
    ) * payload_scale(payload, 2);
}

__global__ void down_resident_kernel(
    const uint8_t * const * payloads,
    int batch,
    int hidden,
    int intermediate,
    const int8_t * activated_values,
    const float * activated_scales,
    float * output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int expert = index / hidden;
    const int row = index % hidden;
    if (expert >= batch) {
        return;
    }
    const uint8_t * payload = payloads[expert];
    const std::size_t gate_up_len = static_cast<std::size_t>(payload[0]) |
        (static_cast<std::size_t>(payload[1]) << 8) |
        (static_cast<std::size_t>(payload[2]) << 16) |
        (static_cast<std::size_t>(payload[3]) << 24);
    const uint8_t * down = payload + kPayloadHeaderBytes + gate_up_len;
    output[expert * hidden + row] = dot_nvfp4(
        down,
        row,
        intermediate,
        activated_values + expert * intermediate,
        activated_scales + expert * (intermediate / 32)
    ) * payload_scale(payload, 2);
}

__global__ void gate_up_route_batch_resident_kernel(
    const uint8_t * const * payloads,
    const uint32_t * token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const int8_t * hidden_values,
    const float * hidden_scales,
    float * gate_up_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows_per_pair = 2 * intermediate;
    const int pair = index / rows_per_pair;
    const int row = index % rows_per_pair;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t * payload = payloads[pair];
    const uint8_t * gate_up = payload + kPayloadHeaderBytes;
    const bool is_up = row >= intermediate;
    const int token = static_cast<int>(token_ids[pair]);
    const float value = dot_nvfp4(
        gate_up,
        row,
        hidden,
        hidden_values + token * hidden,
        hidden_scales + token * (hidden / 32)
    );
    gate_up_values[pair * rows_per_pair + row] =
        value * (is_up ? payload_scale(payload, 1) : payload_scale(payload, 0));
}

__global__ void down_route_batch_resident_kernel(
    const uint8_t * const * payloads,
    int pair_count,
    int hidden,
    int intermediate,
    const int8_t * activated_values,
    const float * activated_scales,
    float * pair_output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = index / hidden;
    const int row = index % hidden;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t * payload = payloads[pair];
    const std::size_t gate_up_len = static_cast<std::size_t>(payload[0]) |
        (static_cast<std::size_t>(payload[1]) << 8) |
        (static_cast<std::size_t>(payload[2]) << 16) |
        (static_cast<std::size_t>(payload[3]) << 24);
    const uint8_t * down = payload + kPayloadHeaderBytes + gate_up_len;
    pair_output[pair * hidden + row] = dot_nvfp4(
        down,
        row,
        intermediate,
        activated_values + pair * intermediate,
        activated_scales + pair * (intermediate / 32)
    ) * payload_scale(payload, 2);
}

__global__ void aggregate_route_batch_kernel(
    const float * pair_output,
    const float * weights,
    const std::size_t * token_offsets,
    int token_count,
    int hidden,
    float * output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = index / hidden;
    const int row = index % hidden;
    if (token >= token_count) {
        return;
    }
    float value = 0.0f;
    const std::size_t begin = token_offsets[token];
    const std::size_t end = token_offsets[token + 1];
    for (std::size_t pair = begin; pair < end; ++pair) {
        value += weights[pair] * pair_output[pair * hidden + row];
    }
    output[token * hidden + row] = value;
}

__global__ void aggregate_route_batch_deduplicated_kernel(
    const float *pair_output,
    const float *weights,
    const uint32_t *route_to_pair,
    const std::size_t *token_offsets,
    int token_count,
    int hidden,
    float *output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = index / hidden;
    const int row = index % hidden;
    if (token >= token_count) {
        return;
    }
    float value = 0.0f;
    const std::size_t begin = token_offsets[token];
    const std::size_t end = token_offsets[token + 1];
    for (std::size_t route = begin; route < end; ++route) {
        value += weights[route] * pair_output[
            static_cast<std::size_t>(route_to_pair[route]) * hidden + row];
    }
    output[token * hidden + row] = value;
}

// K-quants usam superblocos de 256 valores. Estas rotinas reimplementam a
// convenção GGML a partir do layout do formato, sem depender de arquitetura
// ou de um modelo específico.
__device__ float k_f16(uint16_t bits) {
    const float sign = (bits & 0x8000u) != 0 ? -1.0f : 1.0f;
    const int exponent = (bits >> 10) & 0x1f;
    const uint32_t fraction = bits & 0x03ffu;
    if (exponent == 0) {
        return sign * static_cast<float>(fraction) * exp2f(-24.0f);
    }
    if (exponent == 31) {
        const float infinity = __int_as_float(0x7f800000);
        const float nan = __int_as_float(0x7fc00000);
        return fraction == 0 ? sign * infinity : nan;
    }
    return sign * (1.0f + static_cast<float>(fraction) / 1024.0f) *
        exp2f(static_cast<float>(exponent - 15));
}

__device__ uint16_t k_u16(const uint8_t *bytes) {
    return static_cast<uint16_t>(bytes[0]) |
        (static_cast<uint16_t>(bytes[1]) << 8);
}

__device__ float k_dot_q2_k(
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    constexpr int block_bytes = 84;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int block = 0; block < columns / 256; ++block) {
        const uint8_t *packed = encoded + block * block_bytes;
        const uint8_t *scales = packed;
        const uint8_t *quants = packed + 16;
        const float d = k_f16(k_u16(packed + 80));
        const float minimum = k_f16(k_u16(packed + 82));
        int scale_index = 0;
        for (int chunk = 0; chunk < 2; ++chunk) {
            const uint8_t *values = quants + chunk * 32;
            int shift = 0;
            for (int group = 0; group < 4; ++group) {
                const uint8_t scale0 = scales[scale_index++];
                const uint8_t scale1 = scales[scale_index++];
                const float dl0 = d * static_cast<float>(scale0 & 0x0fu);
                const float ml0 = minimum * static_cast<float>(scale0 >> 4);
                const float dl1 = d * static_cast<float>(scale1 & 0x0fu);
                const float ml1 = minimum * static_cast<float>(scale1 >> 4);
                const int input_base = block * 256 + chunk * 128 + group * 32;
                for (int lane = 0; lane < 16; ++lane) {
                    result += input[input_base + lane] *
                        (dl0 * static_cast<float>((values[lane] >> shift) & 3u) - ml0);
                    result += input[input_base + lane + 16] *
                        (dl1 * static_cast<float>((values[lane + 16] >> shift) & 3u) - ml1);
                }
                shift += 2;
            }
        }
    }
    return result;
}

__device__ float k_dot_q3_k(
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    constexpr int block_bytes = 110;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int block = 0; block < columns / 256; ++block) {
        const uint8_t *packed = encoded + block * block_bytes;
        const uint8_t *high = packed;
        const uint8_t *quants = packed + 32;
        uint32_t aux[4];
        for (int word = 0; word < 4; ++word) {
            const uint8_t *bytes = packed + 96 + word * 4;
            aux[word] = static_cast<uint32_t>(k_u16(bytes)) |
                (static_cast<uint32_t>(k_u16(bytes + 2)) << 16);
        }
        constexpr uint32_t mask1 = 0x03030303u;
        constexpr uint32_t mask2 = 0x0f0f0f0fu;
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & mask2) | (((tmp >> 4) & mask1) << 4);
        aux[3] = ((aux[1] >> 4) & mask2) | (((tmp >> 6) & mask1) << 4);
        aux[0] = (aux[0] & mask2) | ((tmp & mask1) << 4);
        aux[1] = (aux[1] & mask2) | (((tmp >> 2) & mask1) << 4);
        uint8_t scales[16];
        for (int word = 0; word < 4; ++word) {
            scales[word * 4] = static_cast<uint8_t>(aux[word]);
            scales[word * 4 + 1] = static_cast<uint8_t>(aux[word] >> 8);
            scales[word * 4 + 2] = static_cast<uint8_t>(aux[word] >> 16);
            scales[word * 4 + 3] = static_cast<uint8_t>(aux[word] >> 24);
        }
        const float d = k_f16(k_u16(packed + 108));
        int scale_index = 0;
        uint8_t high_mask = 1;
        for (int chunk = 0; chunk < 2; ++chunk) {
            const uint8_t *values = quants + chunk * 32;
            int shift = 0;
            for (int group = 0; group < 4; ++group) {
                const int scale0 = static_cast<int>(static_cast<int8_t>(scales[scale_index++])) - 32;
                const int scale1 = static_cast<int>(static_cast<int8_t>(scales[scale_index++])) - 32;
                const float factor0 = d * static_cast<float>(scale0);
                const float factor1 = d * static_cast<float>(scale1);
                const int input_base = block * 256 + chunk * 128 + group * 32;
                for (int lane = 0; lane < 16; ++lane) {
                    const int high0 = (high[lane] & high_mask) != 0 ? 0 : 4;
                    const int high1 = (high[lane + 16] & high_mask) != 0 ? 0 : 4;
                    result += input[input_base + lane] * factor0 * static_cast<float>(
                        static_cast<int>((values[lane] >> shift) & 3u) - high0);
                    result += input[input_base + lane + 16] * factor1 * static_cast<float>(
                        static_cast<int>((values[lane + 16] >> shift) & 3u) - high1);
                }
                shift += 2;
                high_mask <<= 1;
            }
        }
    }
    return result;
}

__device__ float k_q4_scale(uint32_t index, const uint8_t *scales, bool minimum) {
    if (index < 4) {
        return static_cast<float>(minimum ? (scales[index + 4] & 63u) : (scales[index] & 63u));
    }
    return static_cast<float>(minimum
        ? ((scales[index + 4] >> 4) | ((scales[index] >> 6) << 4))
        : ((scales[index + 4] & 0x0fu) | ((scales[index - 4] >> 6) << 4)));
}

__device__ float k_dot_q4_k(
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    constexpr int block_bytes = 144;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int block = 0; block < columns / 256; ++block) {
        const uint8_t *packed = encoded + block * block_bytes;
        const float d = k_f16(k_u16(packed));
        const float minimum = k_f16(k_u16(packed + 2));
        const uint8_t *scales = packed + 4;
        const uint8_t *quants = packed + 16;
        for (int chunk = 0; chunk < 4; ++chunk) {
            const uint8_t *values = quants + chunk * 32;
            const uint32_t scale_index = static_cast<uint32_t>(chunk * 2);
            const float scale0 = d * k_q4_scale(scale_index, scales, false);
            const float scale1 = d * k_q4_scale(scale_index + 1, scales, false);
            const float min0 = minimum * k_q4_scale(scale_index, scales, true);
            const float min1 = minimum * k_q4_scale(scale_index + 1, scales, true);
            const int input_base = block * 256 + chunk * 64;
            for (int lane = 0; lane < 32; ++lane) {
                result += input[input_base + lane] *
                    (scale0 * static_cast<float>(values[lane] & 0x0f) - min0);
                result += input[input_base + lane + 32] *
                    (scale1 * static_cast<float>(values[lane] >> 4) - min1);
            }
        }
    }
    return result;
}

__device__ float k_dot_q5_k(
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    constexpr int block_bytes = 176;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int block = 0; block < columns / 256; ++block) {
        const uint8_t *packed = encoded + block * block_bytes;
        const float d = k_f16(k_u16(packed));
        const float minimum = k_f16(k_u16(packed + 2));
        const uint8_t *scales = packed + 4;
        const uint8_t *high = packed + 16;
        const uint8_t *quants = packed + 48;
        for (int chunk = 0; chunk < 4; ++chunk) {
            const uint8_t *values = quants + chunk * 32;
            const uint32_t scale_index = static_cast<uint32_t>(chunk * 2);
            const float scale0 = d * k_q4_scale(scale_index, scales, false);
            const float scale1 = d * k_q4_scale(scale_index + 1, scales, false);
            const float min0 = minimum * k_q4_scale(scale_index, scales, true);
            const float min1 = minimum * k_q4_scale(scale_index + 1, scales, true);
            const uint8_t mask_low = static_cast<uint8_t>(1u << (chunk * 2));
            const uint8_t mask_high = static_cast<uint8_t>(2u << (chunk * 2));
            const int input_base = block * 256 + chunk * 64;
            for (int lane = 0; lane < 32; ++lane) {
                const float low_bits = static_cast<float>(values[lane] & 0x0f) +
                    ((high[lane] & mask_low) != 0 ? 16.0f : 0.0f);
                const float high_bits = static_cast<float>(values[lane] >> 4) +
                    ((high[lane] & mask_high) != 0 ? 16.0f : 0.0f);
                result += input[input_base + lane] * (scale0 * low_bits - min0);
                result += input[input_base + lane + 32] * (scale1 * high_bits - min1);
            }
        }
    }
    return result;
}

__device__ float k_dot_q6_k(
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    constexpr int block_bytes = 210;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int block = 0; block < columns / 256; ++block) {
        const uint8_t *packed = encoded + block * block_bytes;
        // GGML block_q6_K: ql[128], qh[64], scales[16], d[2].
        const uint8_t *ql = packed;
        const uint8_t *qh = packed + 128;
        const uint8_t *scales = packed + 192;
        const float d = k_f16(k_u16(packed + 208));
        for (int half = 0; half < 2; ++half) {
            const uint8_t *low = ql + half * 64;
            const uint8_t *high = qh + half * 32;
            const uint8_t *scale_group = scales + half * 8;
            const int input_base = block * 256 + half * 128;
            for (int lane = 0; lane < 32; ++lane) {
                const int group = lane / 16;
                const int q1 = static_cast<int>((low[lane] & 0x0f) | ((high[lane] & 3u) << 4)) - 32;
                const int q2 = static_cast<int>((low[lane + 32] & 0x0f) | (((high[lane] >> 2) & 3u) << 4)) - 32;
                const int q3 = static_cast<int>((low[lane] >> 4) | (((high[lane] >> 4) & 3u) << 4)) - 32;
                const int q4 = static_cast<int>((low[lane + 32] >> 4) | (((high[lane] >> 6) & 3u) << 4)) - 32;
                result += input[input_base + lane] * d * static_cast<float>(static_cast<int8_t>(scale_group[group])) * q1;
                result += input[input_base + lane + 32] * d * static_cast<float>(static_cast<int8_t>(scale_group[group + 2])) * q2;
                result += input[input_base + lane + 64] * d * static_cast<float>(static_cast<int8_t>(scale_group[group + 4])) * q3;
                result += input[input_base + lane + 96] * d * static_cast<float>(static_cast<int8_t>(scale_group[group + 6])) * q4;
            }
        }
    }
    return result;
}

__device__ float k_value_q6_k(
    const uint8_t *matrix,
    int row,
    int column,
    int columns
) {
    constexpr int block_bytes = 210;
    const int block = column / 256;
    const int offset = column % 256;
    const int half = offset / 128;
    const int within_half = offset % 128;
    const int lane = within_half % 32;
    const int quarter = within_half / 32;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *packed = matrix + row * row_bytes + block * block_bytes;
    const uint8_t *ql = packed;
    const uint8_t *qh = packed + 128;
    const uint8_t *scales = packed + 192;
    const uint8_t *low = ql + half * 64;
    const uint8_t *high = qh + half * 32;
    const int scale_index = half * 8 + (quarter * 2) + (lane / 16);
    const int8_t scale = static_cast<int8_t>(scales[scale_index]);
    uint8_t packed_low = 0;
    uint8_t packed_high = high[lane];
    if (quarter == 0) {
        packed_low = low[lane] & 0x0f;
        packed_high &= 0x03;
    } else if (quarter == 1) {
        packed_low = low[lane + 32] & 0x0f;
        packed_high = (packed_high >> 2) & 0x03;
    } else if (quarter == 2) {
        packed_low = low[lane] >> 4;
        packed_high = (packed_high >> 4) & 0x03;
    } else {
        packed_low = low[lane + 32] >> 4;
        packed_high = (packed_high >> 6) & 0x03;
    }
    const int quantized = static_cast<int>(packed_low | (packed_high << 4)) - 32;
    const float d = k_f16(k_u16(packed + 208));
    return d * static_cast<float>(scale) * static_cast<float>(quantized);
}

__device__ float k_value_q2_k(
    const uint8_t *matrix,
    int row,
    int column,
    int columns
) {
    constexpr int block_bytes = 84;
    const int block = column / 256;
    const int offset = column % 256;
    const int chunk = offset / 128;
    const int within_chunk = offset % 128;
    const int group = within_chunk / 32;
    const int lane = within_chunk % 32;
    const int scale_index = chunk * 8 + group * 2 + lane / 16;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *packed = matrix + row * row_bytes + block * block_bytes;
    const uint8_t *scales = packed;
    const uint8_t *values = packed + 16 + chunk * 32;
    const uint8_t scale_byte = scales[scale_index];
    const float d = k_f16(k_u16(packed + 80));
    const float scale = d * static_cast<float>(scale_byte & 0x0f);
    const float minimum = k_f16(k_u16(packed + 82)) *
        static_cast<float>(scale_byte >> 4);
    return scale * static_cast<float>((values[lane] >> (group * 2)) & 3u) - minimum;
}

__device__ float k_value_q3_k(
    const uint8_t *matrix,
    int row,
    int column,
    int columns
) {
    constexpr int block_bytes = 110;
    const int block = column / 256;
    const int offset = column % 256;
    const int chunk = offset / 128;
    const int within_chunk = offset % 128;
    const int group = within_chunk / 32;
    const int lane = within_chunk % 32;
    const int scale_index = chunk * 8 + group * 2 + lane / 16;
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *packed = matrix + row * row_bytes + block * block_bytes;
    const uint8_t *high = packed;
    const uint8_t *values = packed + 32 + chunk * 32;
    uint32_t aux[4];
    for (int word = 0; word < 4; ++word) {
        const uint8_t *bytes = packed + 96 + word * 4;
        aux[word] = static_cast<uint32_t>(k_u16(bytes)) |
            (static_cast<uint32_t>(k_u16(bytes + 2)) << 16);
    }
    constexpr uint32_t mask1 = 0x03030303u;
    constexpr uint32_t mask2 = 0x0f0f0f0fu;
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & mask2) | (((tmp >> 4) & mask1) << 4);
    aux[3] = ((aux[1] >> 4) & mask2) | (((tmp >> 6) & mask1) << 4);
    aux[0] = (aux[0] & mask2) | ((tmp & mask1) << 4);
    aux[1] = (aux[1] & mask2) | (((tmp >> 2) & mask1) << 4);
    const uint8_t scale_byte = static_cast<uint8_t>(aux[scale_index / 4] >>
        ((scale_index % 4) * 8));
    const float d = k_f16(k_u16(packed + 108));
    const int high_value = (high[lane] & (1u << (chunk * 4 + group))) != 0 ? 0 : 4;
    const int quantized = static_cast<int>((values[lane] >> (group * 2)) & 3u) - high_value;
    return d * static_cast<float>(static_cast<int8_t>(scale_byte) - 32) *
        static_cast<float>(quantized);
}

__device__ float k_value_q4_k(
    const uint8_t *matrix,
    int row,
    int column,
    int columns
) {
    constexpr int block_bytes = 144;
    const int block = column / 256;
    const int offset = column % 256;
    const int chunk = offset / 64;
    const int within_chunk = offset % 64;
    const int lane = within_chunk % 32;
    const bool high_nibble = within_chunk >= 32;
    const int scale_index = chunk * 2 + (high_nibble ? 1 : 0);
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *packed = matrix + row * row_bytes + block * block_bytes;
    const float d = k_f16(k_u16(packed));
    const float minimum = k_f16(k_u16(packed + 2));
    const uint8_t *scales = packed + 4;
    const uint8_t *values = packed + 16 + chunk * 32;
    const float scale = d * k_q4_scale(scale_index, scales, false);
    const float min = minimum * k_q4_scale(scale_index, scales, true);
    const uint8_t quantized = high_nibble ? values[lane] >> 4 : values[lane] & 0x0f;
    return scale * static_cast<float>(quantized) - min;
}

__device__ float k_value_q5_k(
    const uint8_t *matrix,
    int row,
    int column,
    int columns
) {
    constexpr int block_bytes = 176;
    const int block = column / 256;
    const int offset = column % 256;
    const int chunk = offset / 64;
    const int within_chunk = offset % 64;
    const int lane = within_chunk % 32;
    const bool high_nibble = within_chunk >= 32;
    const int scale_index = chunk * 2 + (high_nibble ? 1 : 0);
    const int row_bytes = (columns / 256) * block_bytes;
    const uint8_t *packed = matrix + row * row_bytes + block * block_bytes;
    const float d = k_f16(k_u16(packed));
    const float minimum = k_f16(k_u16(packed + 2));
    const uint8_t *scales = packed + 4;
    const uint8_t *high = packed + 16;
    const uint8_t *values = packed + 48 + chunk * 32;
    const float scale = d * k_q4_scale(scale_index, scales, false);
    const float min = minimum * k_q4_scale(scale_index, scales, true);
    const uint8_t mask = static_cast<uint8_t>((high_nibble ? 2u : 1u) << (chunk * 2));
    const float quantized =
        static_cast<float>(high_nibble ? values[lane] >> 4 : values[lane] & 0x0f) +
        ((high[lane] & mask) != 0 ? 16.0f : 0.0f);
    return scale * quantized - min;
}

__device__ float k_value(
    uint32_t dtype,
    const uint8_t *matrix,
    int row,
    int column,
    int columns
) {
    if (dtype == 10) return k_value_q2_k(matrix, row, column, columns);
    if (dtype == 11) return k_value_q3_k(matrix, row, column, columns);
    if (dtype == 12) return k_value_q4_k(matrix, row, column, columns);
    if (dtype == 13) return k_value_q5_k(matrix, row, column, columns);
    return k_value_q6_k(matrix, row, column, columns);
}

__device__ float k_dot(
    uint32_t dtype,
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    if (dtype == 10) {
        return k_dot_q2_k(matrix, row, columns, input);
    }
    if (dtype == 11) {
        return k_dot_q3_k(matrix, row, columns, input);
    }
    if (dtype == 12) {
        return k_dot_q4_k(matrix, row, columns, input);
    }
    return dtype == 13
        ? k_dot_q5_k(matrix, row, columns, input)
        : k_dot_q6_k(matrix, row, columns, input);
}

bool kquant_payload_geometry(
    uint32_t dtype,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    std::size_t *gate_up_bytes,
    std::size_t *total_bytes
) {
    if ((dtype != 10 && dtype != 11 && dtype != 12 && dtype != 13 && dtype != 14) ||
        hidden_width == 0 || intermediate_width == 0 ||
        hidden_width % 256 != 0 || intermediate_width % 256 != 0 ||
        gate_up_bytes == nullptr || total_bytes == nullptr) {
        return false;
    }
    const std::size_t block_bytes = dtype == 10 ? 84 : dtype == 11 ? 110 :
        dtype == 12 ? 144 : dtype == 13 ? 176 : 210;
    const std::size_t hidden_blocks = hidden_width / 256;
    const std::size_t intermediate_blocks = intermediate_width / 256;
    if (hidden_blocks > std::numeric_limits<std::size_t>::max() / block_bytes ||
        intermediate_blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
        return false;
    }
    const std::size_t gate_up_row_bytes = hidden_blocks * block_bytes;
    const std::size_t down_row_bytes = intermediate_blocks * block_bytes;
    if (intermediate_width > std::numeric_limits<std::size_t>::max() / 2 ||
        2 * intermediate_width > std::numeric_limits<std::size_t>::max() / gate_up_row_bytes ||
        hidden_width > std::numeric_limits<std::size_t>::max() / down_row_bytes) {
        return false;
    }
    *gate_up_bytes = 2 * intermediate_width * gate_up_row_bytes;
    const std::size_t down_bytes = hidden_width * down_row_bytes;
    if (*gate_up_bytes > std::numeric_limits<std::size_t>::max() - kKQuantPayloadHeaderBytes ||
        kKQuantPayloadHeaderBytes + *gate_up_bytes >
            std::numeric_limits<std::size_t>::max() - down_bytes) {
        return false;
    }
    *total_bytes = kKQuantPayloadHeaderBytes + *gate_up_bytes + down_bytes;
    return true;
}

bool iquant_payload_geometry(
    uint32_t dtype,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    std::size_t *gate_up_bytes,
    std::size_t *total_bytes
) {
    if ((dtype != 18 && dtype != 21 && dtype != 22 && dtype != 23) ||
        hidden_width == 0 || intermediate_width == 0 ||
        hidden_width % 256 != 0 || intermediate_width % 256 != 0 ||
        gate_up_bytes == nullptr || total_bytes == nullptr) {
        return false;
    }
    const std::size_t block_bytes = dtype == 18 ? 98 : dtype == 21 ? 110 :
        dtype == 22 ? 82 : 136;
    const std::size_t hidden_blocks = hidden_width / 256;
    const std::size_t intermediate_blocks = intermediate_width / 256;
    if (hidden_blocks > std::numeric_limits<std::size_t>::max() / block_bytes ||
        intermediate_blocks > std::numeric_limits<std::size_t>::max() / block_bytes) {
        return false;
    }
    const std::size_t gate_up_row_bytes = hidden_blocks * block_bytes;
    const std::size_t down_row_bytes = intermediate_blocks * block_bytes;
    if (intermediate_width > std::numeric_limits<std::size_t>::max() / 2 ||
        2 * intermediate_width > std::numeric_limits<std::size_t>::max() / gate_up_row_bytes ||
        hidden_width > std::numeric_limits<std::size_t>::max() / down_row_bytes) {
        return false;
    }
    *gate_up_bytes = 2 * intermediate_width * gate_up_row_bytes;
    const std::size_t down_bytes = hidden_width * down_row_bytes;
    if (*gate_up_bytes > std::numeric_limits<std::size_t>::max() - kScalarPayloadHeaderBytes ||
        kScalarPayloadHeaderBytes + *gate_up_bytes >
            std::numeric_limits<std::size_t>::max() - down_bytes) {
        return false;
    }
    *total_bytes = kScalarPayloadHeaderBytes + *gate_up_bytes + down_bytes;
    return true;
}

// IQ usa codebooks GGML por superbloco. Estas funções acessam um peso pela
// posição lógica, preservando a mesma ordem que dequantize_row_* usa no CPU.
__device__ __forceinline__ float iq2s_value(const uint8_t *matrix, int index) {
    constexpr int block_bytes = 82;
    const int block_index = index / 256;
    const int within = index % 256;
    const block_iq2_s *block = reinterpret_cast<const block_iq2_s *>(
        matrix + block_index * block_bytes);
    const int group = within / 32;
    const int lane = within % 32;
    const int packed_group = lane / 8;
    const int element = lane % 8;
    const uint8_t q = block->qs[group * 4 + packed_group];
    const int grid_index = q | ((static_cast<int>(block->qh[group]) <<
        (8 - 2 * packed_group)) & 0x300);
    const uint64_t grid = iq2s_grid[grid_index];
    const uint8_t signs = block->qs[32 + group * 4 + packed_group];
    const uint8_t code = static_cast<uint8_t>((grid >> (8 * element)) & 0xffu);
    const float scale = __half2float(block->d) *
        (0.5f + static_cast<float>((block->scales[group] >>
            (4 * (packed_group / 2))) & 0x0f)) * 0.25f;
    return scale * static_cast<float>(code) *
        ((signs & kmask_iq2xs[element]) != 0 ? -1.0f : 1.0f);
}

__device__ __forceinline__ float iq3s_value(const uint8_t *matrix, int index) {
    constexpr int block_bytes = 110;
    const int block_index = index / 256;
    const int within = index % 256;
    const block_iq3_s *block = reinterpret_cast<const block_iq3_s *>(
        matrix + block_index * block_bytes);
    const int group = within / 32;
    const int lane = within % 32;
    const int packed_group = lane / 8;
    const int element = lane % 8;
    const uint8_t *qs = block->qs + group * 8;
    const int pair = element / 4;
    const int in_pair = element % 4;
    const int q = qs[packed_group * 2 + pair];
    const int high_shift = pair == 0 ? (8 - 2 * packed_group) :
        (7 - 2 * packed_group);
    const int grid_index = q | ((static_cast<int>(block->qh[group]) <<
        high_shift) & 0x100);
    const uint32_t grid = iq3s_grid[grid_index];
    const uint8_t signs = block->signs[group * 4 + packed_group];
    const uint8_t code = static_cast<uint8_t>((grid >> (8 * in_pair)) & 0xffu);
    const float scale = __half2float(block->d) *
        (1.0f + 2.0f * static_cast<float>((block->scales[group / 2] >>
            (4 * (group % 2))) & 0x0f));
    return scale * static_cast<float>(code) *
        ((signs & kmask_iq2xs[element]) != 0 ? -1.0f : 1.0f);
}

__device__ __forceinline__ float iq3xxs_value(const uint8_t *matrix, int index) {
    constexpr int block_bytes = 98;
    const int block_index = index / 256;
    const int within = index % 256;
    const uint8_t *block = matrix + block_index * block_bytes;
    const int group = within / 32;
    const int in_group = within % 32;
    const int il = in_group / 8;
    const int element = in_group % 8;
    const uint8_t *q3 = block + 2 + group * 8;
    const uint8_t grid_index = q3[2 * il + element / 4];
    const uint32_t grid = iq3xxs_grid[grid_index];
    const uint32_t aux32 =
        static_cast<uint32_t>(block[66 + group * 4]) |
        (static_cast<uint32_t>(block[67 + group * 4]) << 8) |
        (static_cast<uint32_t>(block[68 + group * 4]) << 16) |
        (static_cast<uint32_t>(block[69 + group * 4]) << 24);
    const float d = __half2float(__ushort_as_half(
        static_cast<uint16_t>(block[0]) |
        (static_cast<uint16_t>(block[1]) << 8))) *
        (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    const int shift = 8 * (element % 4);
    const uint8_t code = static_cast<uint8_t>((grid >> shift) & 0xffu);
    return d * static_cast<float>(code) *
        ((signs & kmask_iq2xs[element]) != 0 ? -1.0f : 1.0f);
}

__device__ __forceinline__ float iq4xs_value(const uint8_t *matrix, int index) {
    constexpr int block_bytes = 136;
    const int block_index = index / 256;
    const int within = index % 256;
    const block_iq4_xs *block = reinterpret_cast<const block_iq4_xs *>(
        matrix + block_index * block_bytes);
    const int group = within / 32;
    const int lane = within % 32;
    const uint8_t packed = block->qs[group * 16 + lane % 16];
    const int code = lane < 16 ? packed & 0x0f : packed >> 4;
    const int scale_code = ((block->scales_l[group / 2] >>
        (4 * (group % 2))) & 0x0f) |
        (((block->scales_h >> (2 * group)) & 0x03) << 4);
    return __half2float(block->d) * static_cast<float>(scale_code - 32) *
        static_cast<float>(kvalues_iq4nl[code]);
}

__device__ __forceinline__ float e8m0_to_f32(uint8_t value) {
    if (value < 2) {
        return __uint_as_float(0x00200000u << value);
    }
    return __uint_as_float(static_cast<uint32_t>(value - 1) << 23);
}

__device__ __forceinline__ float mxfp4_value(const uint8_t *matrix, int index) {
    constexpr int block_bytes = 17;
    const int block_index = index / 32;
    const int within = index % 32;
    const uint8_t *block = matrix + block_index * block_bytes;
    const uint8_t packed = block[1 + (within % 16)];
    const uint8_t nibble = within < 16 ? packed & 0x0f : packed >> 4;
    return e8m0_to_f32(block[0]) * fp4_value(nibble);
}

__device__ float mxfp4_dot(
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    const int row_bytes = (columns / 32) * 17;
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int index = 0; index < columns; ++index) {
        result += mxfp4_value(encoded, index) * input[index];
    }
    return result;
}

__device__ __forceinline__ float iq_value(
    uint32_t dtype,
    const uint8_t *encoded,
    int index
) {
    if (dtype == 18) return iq3xxs_value(encoded, index);
    if (dtype == 21) return iq3s_value(encoded, index);
    if (dtype == 22) return iq2s_value(encoded, index);
    return iq4xs_value(encoded, index);
}

__device__ float iq_dot(
    uint32_t dtype,
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    const int row_bytes = (dtype == 18 ? 98 : dtype == 21 ? 110 : dtype == 22 ? 82 : 136) *
        (columns / 256);
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int index = 0; index < columns; ++index) {
        result += iq_value(dtype, encoded, index) * input[index];
    }
    return result;
}

// IQ3_XXS stores 256 values per superblock. One warp covers one superblock:
// each lane owns the eight values of one (group, il) slice. This keeps the
// q3/codebook metadata reusable and avoids paying for four warps and a shared
// memory tree reduction for a single output row. The result is a partial sum
// for the calling lane.
__device__ __forceinline__ float iq3xxs_warp_partial_dot(
    const uint8_t *matrix,
    int columns,
    const float *input
) {
    constexpr int block_elements = 256;
    constexpr int block_bytes = 98;
    const int lane = threadIdx.x & 31;
    const int group = lane / 4;
    const int il = lane % 4;
    float result = 0.0f;
    for (int block_index = 0; block_index < columns / block_elements; ++block_index) {
        const uint8_t *block = matrix + block_index * block_bytes;
        const uint8_t *q3 = block + 2 + group * 8;
        const uint32_t aux32 = static_cast<uint32_t>(block[66 + group * 4]) |
            (static_cast<uint32_t>(block[67 + group * 4]) << 8) |
            (static_cast<uint32_t>(block[68 + group * 4]) << 16) |
            (static_cast<uint32_t>(block[69 + group * 4]) << 24);
        const float d = __half2float(__ushort_as_half(
            static_cast<uint16_t>(block[0]) |
            (static_cast<uint16_t>(block[1]) << 8))) *
            (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        const uint32_t grid_lo = iq3xxs_grid[q3[2 * il]];
        const uint32_t grid_hi = iq3xxs_grid[q3[2 * il + 1]];
        const int column_base = block_index * block_elements + group * 32 + il * 8;
        for (int element = 0; element < 8; ++element) {
            const uint32_t grid = element < 4 ? grid_lo : grid_hi;
            const int grid_element = element & 3;
            const uint8_t code = static_cast<uint8_t>((grid >> (8 * grid_element)) & 0xffu);
            result += d * static_cast<float>(code) *
                ((signs & kmask_iq2xs[element]) != 0 ? -1.0f : 1.0f) *
                input[column_base + element];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    return result;
}

// Gate and up are adjacent matrices in the resident payload and consume the
// same hidden row. Compute both accumulators in one warp so the hidden values
// and the route metadata are fetched once instead of once per projection.
__device__ __forceinline__ void iq3xxs_warp_partial_dot_gate_up(
    const uint8_t *gate_matrix,
    const uint8_t *up_matrix,
    int columns,
    const float *input,
    float *gate_result,
    float *up_result
) {
    constexpr int block_elements = 256;
    constexpr int block_bytes = 98;
    const int lane = threadIdx.x & 31;
    const int group = lane / 4;
    const int il = lane % 4;
    float gate = 0.0f;
    float up = 0.0f;
    for (int block_index = 0; block_index < columns / block_elements; ++block_index) {
        const uint8_t *gate_block = gate_matrix + block_index * block_bytes;
        const uint8_t *up_block = up_matrix + block_index * block_bytes;
        const uint8_t *gate_q3 = gate_block + 2 + group * 8;
        const uint8_t *up_q3 = up_block + 2 + group * 8;
        const uint32_t gate_aux32 = static_cast<uint32_t>(gate_block[66 + group * 4]) |
            (static_cast<uint32_t>(gate_block[67 + group * 4]) << 8) |
            (static_cast<uint32_t>(gate_block[68 + group * 4]) << 16) |
            (static_cast<uint32_t>(gate_block[69 + group * 4]) << 24);
        const uint32_t up_aux32 = static_cast<uint32_t>(up_block[66 + group * 4]) |
            (static_cast<uint32_t>(up_block[67 + group * 4]) << 8) |
            (static_cast<uint32_t>(up_block[68 + group * 4]) << 16) |
            (static_cast<uint32_t>(up_block[69 + group * 4]) << 24);
        const float gate_d = __half2float(__ushort_as_half(
            static_cast<uint16_t>(gate_block[0]) |
            (static_cast<uint16_t>(gate_block[1]) << 8))) *
            (0.5f + static_cast<float>(gate_aux32 >> 28)) * 0.5f;
        const float up_d = __half2float(__ushort_as_half(
            static_cast<uint16_t>(up_block[0]) |
            (static_cast<uint16_t>(up_block[1]) << 8))) *
            (0.5f + static_cast<float>(up_aux32 >> 28)) * 0.5f;
        const uint8_t gate_signs = ksigns_iq2xs[(gate_aux32 >> (7 * il)) & 127];
        const uint8_t up_signs = ksigns_iq2xs[(up_aux32 >> (7 * il)) & 127];
        const uint32_t gate_grid_lo = iq3xxs_grid[gate_q3[2 * il]];
        const uint32_t gate_grid_hi = iq3xxs_grid[gate_q3[2 * il + 1]];
        const uint32_t up_grid_lo = iq3xxs_grid[up_q3[2 * il]];
        const uint32_t up_grid_hi = iq3xxs_grid[up_q3[2 * il + 1]];
        const int column_base = block_index * block_elements + group * 32 + il * 8;
        for (int element = 0; element < 8; ++element) {
            const uint32_t gate_grid = element < 4 ? gate_grid_lo : gate_grid_hi;
            const uint32_t up_grid = element < 4 ? up_grid_lo : up_grid_hi;
            const int grid_element = element & 3;
            const uint8_t gate_code = static_cast<uint8_t>(
                (gate_grid >> (8 * grid_element)) & 0xffu);
            const uint8_t up_code = static_cast<uint8_t>(
                (up_grid >> (8 * grid_element)) & 0xffu);
            const float hidden_value = input[column_base + element];
            gate += gate_d * static_cast<float>(gate_code) *
                ((gate_signs & kmask_iq2xs[element]) != 0 ? -1.0f : 1.0f) * hidden_value;
            up += up_d * static_cast<float>(up_code) *
                ((up_signs & kmask_iq2xs[element]) != 0 ? -1.0f : 1.0f) * hidden_value;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    *gate_result = gate;
    *up_result = up;
}

// llama.cpp's MMVQ path quantizes the activation to q8_1 and evaluates the
// IQ3_XXS codebook with dp4a. Keep the same representation here so the
// external callback does not fall back to scalar float/codebook arithmetic.
__device__ __forceinline__ float iq3xxs_warp_partial_dot_q8_1(
    const uint8_t *matrix,
    int columns,
    const block_q8_1 *input
) {
    constexpr int block_elements = 256;
    constexpr int q8_blocks_per_block = block_elements / QK8_1;
    constexpr int q8_chunks_per_warp = q8_blocks_per_block;
    constexpr int blocks_per_iteration = 32 / q8_chunks_per_warp;
    const int lane = threadIdx.x & 31;
    const int q8_chunk = lane % q8_chunks_per_warp;
    float result = 0.0f;
    for (int block_index = lane / q8_chunks_per_warp;
         block_index < columns / block_elements;
         block_index += blocks_per_iteration) {
        result += vec_dot_iq3_xxs_q8_1(
            matrix,
            input + block_index * q8_blocks_per_block,
            block_index,
            q8_chunk * 2);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        result += __shfl_down_sync(0xffffffffu, result, offset);
    }
    return result;
}

__device__ __forceinline__ void iq3xxs_warp_partial_dot_q8_1_gate_up(
    const uint8_t *gate_matrix,
    const uint8_t *up_matrix,
    int columns,
    const block_q8_1 *input,
    float *gate_result,
    float *up_result
) {
    constexpr int block_elements = 256;
    constexpr int q8_blocks_per_block = block_elements / QK8_1;
    constexpr int q8_chunks_per_warp = q8_blocks_per_block;
    constexpr int blocks_per_iteration = 32 / q8_chunks_per_warp;
    const int lane = threadIdx.x & 31;
    const int q8_chunk = lane % q8_chunks_per_warp;
    float gate = 0.0f;
    float up = 0.0f;
    for (int block_index = lane / q8_chunks_per_warp;
         block_index < columns / block_elements;
         block_index += blocks_per_iteration) {
        const block_q8_1 *input_block = input + block_index * q8_blocks_per_block;
        gate += vec_dot_iq3_xxs_q8_1(
            gate_matrix, input_block, block_index, q8_chunk * 2);
        up += vec_dot_iq3_xxs_q8_1(
            up_matrix, input_block, block_index, q8_chunk * 2);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    *gate_result = gate;
    *up_result = up;
}

// Kept for the other IQ layouts, whose metadata mapping is not identical to
// IQ3_XXS. They still use one block per row and a shared-memory reduction.
__device__ float iq3xxs_cooperative_partial_dot(
    const uint8_t *matrix,
    int columns,
    const float *input
) {
    constexpr int block_elements = 256;
    constexpr int block_bytes = 98;
    float result = 0.0f;
    const int lane_pair = threadIdx.x * 2;
    for (int block_index = 0; block_index < columns / block_elements; ++block_index) {
        const uint8_t *block = matrix + block_index * block_bytes;
        const uint8_t *q3 = block + 2;
        const int group = lane_pair / 32;
        const int in_group = lane_pair % 32;
        const int il = in_group / 8;
        const int element = in_group % 8;
        const uint32_t aux32 = static_cast<uint32_t>(block[66 + group * 4]) |
            (static_cast<uint32_t>(block[67 + group * 4]) << 8) |
            (static_cast<uint32_t>(block[68 + group * 4]) << 16) |
            (static_cast<uint32_t>(block[69 + group * 4]) << 24);
        const float d = __half2float(__ushort_as_half(
            static_cast<uint16_t>(block[0]) |
            (static_cast<uint16_t>(block[1]) << 8))) *
            (0.5f + static_cast<float>(aux32 >> 28)) * 0.5f;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
        for (int offset = 0; offset < 2; ++offset) {
            const int value_index = element + offset;
            const uint8_t grid_index = q3[group * 8 + 2 * il + value_index / 4];
            const uint32_t grid = iq3xxs_grid[grid_index];
            const int shift = 8 * (value_index % 4);
            const uint8_t code = static_cast<uint8_t>((grid >> shift) & 0xffu);
            const int column = block_index * block_elements + lane_pair + offset;
            result += d * static_cast<float>(code) *
                ((signs & kmask_iq2xs[value_index]) != 0 ? -1.0f : 1.0f) * input[column];
        }
    }
    return result;
}

__device__ float iq_cooperative_partial_dot(
    uint32_t dtype,
    const uint8_t *matrix,
    int columns,
    const float *input
) {
    if (dtype == 18) {
        return iq3xxs_cooperative_partial_dot(matrix, columns, input);
    }
    float result = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        result += iq_value(dtype, matrix, column) * input[column];
    }
    return result;
}

__device__ float q4_0_value(const uint8_t *encoded, int column) {
    constexpr int block_elements = 32;
    constexpr int block_bytes = 18;
    const uint8_t *block = encoded + (column / block_elements) * block_bytes;
    const int within = column % block_elements;
    const uint8_t packed = block[2 + (within % 16)];
    const uint8_t nibble = within < 16 ? packed & 0x0f : packed >> 4;
    const float scale = __half2float(__ushort_as_half(
        static_cast<uint16_t>(block[0]) |
        (static_cast<uint16_t>(block[1]) << 8)));
    return scale * (static_cast<float>(nibble) - 8.0f);
}

__device__ float q4_0_dot(
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    constexpr int block_elements = 32;
    constexpr int block_bytes = 18;
    const int row_bytes = (columns / block_elements) * block_bytes;
    const uint8_t *encoded = matrix + row * row_bytes;
    float result = 0.0f;
    for (int index = 0; index < columns; ++index) {
        result += q4_0_value(encoded, index) * input[index];
    }
    return result;
}

__device__ float heterogeneous_quant_dot(
    const CudaHeterogeneousTensor &tensor,
    int row,
    int columns,
    const float *input,
    float scale,
    float input_scale
) {
    if (tensor.dtype == 2) {
        return q4_0_dot(tensor.data, row, columns, input) * scale * input_scale;
    }
    if (tensor.dtype == 10 || tensor.dtype == 11 || tensor.dtype == 12 ||
        tensor.dtype == 13 || tensor.dtype == 14) {
        return k_dot(tensor.dtype, tensor.data, row, columns, input) * scale * input_scale;
    }
    if (tensor.dtype == 39) {
        return mxfp4_dot(tensor.data, row, columns, input) * scale * input_scale;
    }
    return iq_dot(tensor.dtype, tensor.data, row, columns, input) * scale * input_scale;
}

__global__ void kquant_gate_up_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const float *hidden_values,
    float *gate_up_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows_per_pair = 2 * intermediate;
    const int pair = index / rows_per_pair;
    const int row = index % rows_per_pair;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t *payload = payloads[pair];
    const float *input = hidden_values + static_cast<std::size_t>(token_ids[pair]) * hidden;
    gate_up_values[pair * rows_per_pair + row] = k_dot(
        dtype, payload + kKQuantPayloadHeaderBytes, row, hidden, input) *
        payload_scale(payload, row < intermediate ? 0 : 1) *
        payload_scale(payload, row < intermediate ? 3 : 4);
}

__global__ void kquant_silu_kernel(
    const float *gate_up_values,
    int pair_count,
    int intermediate,
    float *activated
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = index / intermediate;
    const int row = index % intermediate;
    if (pair >= pair_count) {
        return;
    }
    const float gate = gate_up_values[pair * 2 * intermediate + row];
    const float up = gate_up_values[pair * 2 * intermediate + intermediate + row];
    const float gate_exp = gate < 0.0f ? expf(gate) : expf(-gate);
    const float silu = gate == -CUDART_INF_F
        ? 0.0f
        : gate < 0.0f
            ? gate * gate_exp / (1.0f + gate_exp)
            : gate / (1.0f + gate_exp);
    activated[pair * intermediate + row] = silu * up;
}

// One warp quantizes one contiguous activation block, matching llama.cpp's
// block_q8_1 input expected by its IQ3_XXS MMVQ decoder. Rows are independent
// so the same kernel serves hidden tokens and per-route activated vectors.
__global__ void quantize_q8_1_rows_kernel(
    const float *input,
    int rows,
    int columns,
    block_q8_1 *output
) {
    constexpr int elements_per_block = QK8_1;
    const int blocks_per_row = columns / elements_per_block;
    const int block_index = blockIdx.x;
    const int row = block_index / blocks_per_row;
    const int column_block = block_index % blocks_per_row;
    if (row >= rows) return;

    const int element = threadIdx.x;
    const float value = input[
        static_cast<std::size_t>(row) * columns + column_block * elements_per_block + element];
    float amax = fabsf(value);
    float sum = value;
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, offset));
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    // The reduction result is valid only in lane zero. Every lane must use
    // that same delta when writing the shared q8 block; using the lane-local
    // partial maximum silently corrupts the activation scale and the dot
    // product even though the kernel launch itself succeeds.
    const float row_amax = __shfl_sync(0xffffffffu, amax, 0);
    const float d = row_amax / 127.0f;
    const int8_t quantized = row_amax == 0.0f
        ? 0
        : static_cast<int8_t>(roundf(value / d));
    block_q8_1 *block = &output[
        static_cast<std::size_t>(row) * blocks_per_row + column_block];
    block->qs[element] = quantized;
    if (element == 0) {
        block->ds = make_half2(d, sum);
    }
}

__global__ void kquant_down_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    int pair_count,
    int hidden,
    int intermediate,
    std::size_t gate_up_len,
    const float *activated,
    float *pair_output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = index / hidden;
    const int row = index % hidden;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t *payload = payloads[pair];
    pair_output[pair * hidden + row] = k_dot(
        dtype,
        payload + kKQuantPayloadHeaderBytes + gate_up_len,
        row,
        intermediate,
        activated + pair * intermediate) * payload_scale(payload, 2) *
        payload_scale(payload, 5);
}

// Each block computes one output row. The scalar kernel above assigned one
// thread to each row, which made a real prefill batch launch hundreds of
// thousands of long serial dot products. Keep the route layout unchanged,
// but let 128 threads share every dot product for all supported K-quants.
__global__ void kquant_gate_up_cooperative_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const float *hidden_values,
    float *gate_up_values
) {
    const int row_index = blockIdx.x;
    const int rows_per_pair = 2 * intermediate;
    const int pair = row_index / rows_per_pair;
    const int row = row_index % rows_per_pair;
    if (pair >= pair_count) return;
    const uint8_t *matrix = payloads[pair] + kKQuantPayloadHeaderBytes;
    const float *input = hidden_values + static_cast<std::size_t>(token_ids[pair]) * hidden;
    float value = 0.0f;
    for (int column = threadIdx.x; column < hidden; column += blockDim.x) {
        value += k_value(dtype, matrix, row, column, hidden) * input[column];
    }
    __shared__ float partials[128];
    partials[threadIdx.x] = value;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) partials[threadIdx.x] += partials[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate_up_values[pair * rows_per_pair + row] = partials[0] *
            payload_scale(payloads[pair], row < intermediate ? 0 : 1) *
            payload_scale(payloads[pair], row < intermediate ? 3 : 4);
    }
}

__global__ void kquant_down_cooperative_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    int pair_count,
    int hidden,
    int intermediate,
    std::size_t gate_up_len,
    const float *activated,
    float *pair_output
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / hidden;
    const int row = row_index % hidden;
    if (pair >= pair_count) return;
    const uint8_t *payload = payloads[pair];
    const uint8_t *matrix = payload + kKQuantPayloadHeaderBytes + gate_up_len;
    const float *input = activated + pair * intermediate;
    float value = 0.0f;
    for (int column = threadIdx.x; column < intermediate; column += blockDim.x) {
        value += k_value(dtype, matrix, row, column, intermediate) * input[column];
    }
    __shared__ float partials[128];
    partials[threadIdx.x] = value;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) partials[threadIdx.x] += partials[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pair_output[pair * hidden + row] = partials[0] *
            payload_scale(payload, 2) * payload_scale(payload, 5);
    }
}

__device__ __forceinline__ float plain_value(
    uint32_t dtype,
    const uint8_t *matrix,
    std::size_t index
) {
    if (dtype == 0) {
        uint32_t bits = static_cast<uint32_t>(matrix[index * 4]);
        bits |= static_cast<uint32_t>(matrix[index * 4 + 1]) << 8;
        bits |= static_cast<uint32_t>(matrix[index * 4 + 2]) << 16;
        bits |= static_cast<uint32_t>(matrix[index * 4 + 3]) << 24;
        return __uint_as_float(bits);
    }
    const std::size_t offset = index * 2;
    const uint16_t bits = static_cast<uint16_t>(matrix[offset]) |
        (static_cast<uint16_t>(matrix[offset + 1]) << 8);
    if (dtype == 1) {
        return __half2float(__ushort_as_half(bits));
    }
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ float plain_dot(
    uint32_t dtype,
    const uint8_t *matrix,
    int row,
    int columns,
    const float *input
) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * columns;
    float result = 0.0f;
    for (int column = 0; column < columns; ++column) {
        result += plain_value(dtype, matrix, row_offset + column) * input[column];
    }
    return result;
}

__global__ void plain_gate_up_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const float *hidden_values,
    float *gate_up_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows_per_pair = 2 * intermediate;
    const int pair = index / rows_per_pair;
    const int row = index % rows_per_pair;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t *payload = payloads[pair];
    const uint32_t gate_up_len = static_cast<uint32_t>(payload[0]) |
        (static_cast<uint32_t>(payload[1]) << 8) |
        (static_cast<uint32_t>(payload[2]) << 16) |
        (static_cast<uint32_t>(payload[3]) << 24);
    const float *input = hidden_values + static_cast<std::size_t>(token_ids[pair]) * hidden;
    gate_up_values[pair * rows_per_pair + row] = plain_dot(
        dtype, payload + kScalarPayloadHeaderBytes, row, hidden, input);
    (void)gate_up_len;
}

__global__ void plain_down_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    int pair_count,
    int hidden,
    int intermediate,
    const float *activated,
    float *pair_output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = index / hidden;
    const int row = index % hidden;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t *payload = payloads[pair];
    const uint32_t gate_up_len = static_cast<uint32_t>(payload[0]) |
        (static_cast<uint32_t>(payload[1]) << 8) |
        (static_cast<uint32_t>(payload[2]) << 16) |
        (static_cast<uint32_t>(payload[3]) << 24);
    pair_output[pair * hidden + row] = plain_dot(
        dtype,
        payload + kScalarPayloadHeaderBytes + gate_up_len,
        row,
        intermediate,
        activated + pair * intermediate);
}

__global__ void iquant_gate_up_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const float *hidden_values,
    float *gate_up_values
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows_per_pair = 2 * intermediate;
    const int pair = index / rows_per_pair;
    const int row = index % rows_per_pair;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t *payload = payloads[pair];
    const uint32_t gate_up_len = static_cast<uint32_t>(payload[0]) |
        (static_cast<uint32_t>(payload[1]) << 8) |
        (static_cast<uint32_t>(payload[2]) << 16) |
        (static_cast<uint32_t>(payload[3]) << 24);
    const float *input = hidden_values + static_cast<std::size_t>(token_ids[pair]) * hidden;
    gate_up_values[pair * rows_per_pair + row] = iq_dot(
        dtype, payload + kScalarPayloadHeaderBytes, row, hidden, input);
    (void)gate_up_len;
}

__global__ void iquant_down_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    int pair_count,
    int hidden,
    int intermediate,
    const float *activated,
    float *pair_output
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = index / hidden;
    const int row = index % hidden;
    if (pair >= pair_count) {
        return;
    }
    const uint8_t *payload = payloads[pair];
    const uint32_t gate_up_len = static_cast<uint32_t>(payload[0]) |
        (static_cast<uint32_t>(payload[1]) << 8) |
        (static_cast<uint32_t>(payload[2]) << 16) |
        (static_cast<uint32_t>(payload[3]) << 24);
    pair_output[pair * hidden + row] = iq_dot(
        dtype,
        payload + kScalarPayloadHeaderBytes + gate_up_len,
        row,
        intermediate,
        activated + pair * intermediate);
}

// IQ3_XXS (o formato medido na telemetria), IQ3_S, IQ2_S e IQ4_XS precisam
// do mesmo tratamento cooperativo que os K-quants. O kernel antigo atribuía
// um dot product inteiro a cada thread: para cada linha, 128 threads faziam
// o mesmo loop serial. Um bloco agora compartilha cada redução.
__global__ void iquant_gate_up_cooperative_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const float *hidden_values,
    float *gate_up_values
) {
    const int row_index = blockIdx.x;
    const int rows_per_pair = 2 * intermediate;
    const int pair = row_index / rows_per_pair;
    const int row = row_index % rows_per_pair;
    if (pair >= pair_count) return;
    const std::size_t block_bytes = dtype == 18 ? 98 : dtype == 21 ? 110 :
        dtype == 22 ? 82 : 136;
    const std::size_t row_bytes = static_cast<std::size_t>(hidden / 256) * block_bytes;
    const uint8_t *matrix = payloads[pair] + kScalarPayloadHeaderBytes;
    const uint8_t *encoded = matrix + static_cast<std::size_t>(row) * row_bytes;
    const float *input = hidden_values + static_cast<std::size_t>(token_ids[pair]) * hidden;
    const float value = iq_cooperative_partial_dot(dtype, encoded, hidden, input);
    __shared__ float partials[128];
    partials[threadIdx.x] = value;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) partials[threadIdx.x] += partials[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate_up_values[pair * rows_per_pair + row] = partials[0];
    }
}

__global__ void iquant_gate_up_iq3xxs_warp_route_kernel(
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const float *hidden_values,
    float *gate_up_values
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / intermediate;
    const int row = row_index % intermediate;
    if (pair >= pair_count) return;
    const std::size_t row_bytes = static_cast<std::size_t>(hidden / 256) * 98;
    const uint8_t *matrix = payloads[pair] + kScalarPayloadHeaderBytes;
    const uint8_t *gate = matrix + static_cast<std::size_t>(row) * row_bytes;
    const uint8_t *up = matrix + static_cast<std::size_t>(row + intermediate) * row_bytes;
    const float *input = hidden_values + static_cast<std::size_t>(token_ids[pair]) * hidden;
    float gate_value = 0.0f;
    float up_value = 0.0f;
    iq3xxs_warp_partial_dot_gate_up(
        gate, up, hidden, input, &gate_value, &up_value);
    if (threadIdx.x == 0) {
        const int output_offset = pair * 2 * intermediate + row;
        gate_up_values[output_offset] = gate_value;
        gate_up_values[output_offset + intermediate] = up_value;
    }
}

__global__ void iquant_gate_up_iq3xxs_q8_warp_route_kernel(
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const block_q8_1 *hidden_values,
    float *gate_up_values
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / intermediate;
    const int row = row_index % intermediate;
    if (pair >= pair_count) return;
    const std::size_t row_bytes = static_cast<std::size_t>(hidden / 256) * 98;
    const uint8_t *matrix = payloads[pair] + kScalarPayloadHeaderBytes;
    const uint8_t *gate = matrix + static_cast<std::size_t>(row) * row_bytes;
    const uint8_t *up = matrix + static_cast<std::size_t>(row + intermediate) * row_bytes;
    const block_q8_1 *input = hidden_values +
        static_cast<std::size_t>(token_ids[pair]) * (hidden / QK8_1);
    float gate_value = 0.0f;
    float up_value = 0.0f;
    iq3xxs_warp_partial_dot_q8_1_gate_up(
        gate, up, hidden, input, &gate_value, &up_value);
    if (threadIdx.x == 0) {
        const int output_offset = pair * 2 * intermediate + row;
        gate_up_values[output_offset] = gate_value;
        gate_up_values[output_offset + intermediate] = up_value;
    }
}

__global__ void iquant_down_cooperative_route_kernel(
    uint32_t dtype,
    const uint8_t * const *payloads,
    int pair_count,
    int hidden,
    int intermediate,
    std::size_t gate_up_len,
    const float *activated,
    float *pair_output
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / hidden;
    const int row = row_index % hidden;
    if (pair >= pair_count) return;
    const std::size_t block_bytes = dtype == 18 ? 98 : dtype == 21 ? 110 :
        dtype == 22 ? 82 : 136;
    const std::size_t row_bytes = static_cast<std::size_t>(intermediate / 256) * block_bytes;
    const uint8_t *matrix = payloads[pair] + kScalarPayloadHeaderBytes + gate_up_len;
    const uint8_t *encoded = matrix + static_cast<std::size_t>(row) * row_bytes;
    const float *input = activated + static_cast<std::size_t>(pair) * intermediate;
    const float value = iq_cooperative_partial_dot(dtype, encoded, intermediate, input);
    __shared__ float partials[128];
    partials[threadIdx.x] = value;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) partials[threadIdx.x] += partials[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pair_output[pair * hidden + row] = partials[0];
    }
}

__global__ void iquant_down_iq3xxs_warp_route_kernel(
    const uint8_t * const *payloads,
    int pair_count,
    int hidden,
    int intermediate,
    std::size_t gate_up_len,
    const float *activated,
    float *pair_output
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / hidden;
    const int row = row_index % hidden;
    if (pair >= pair_count) return;
    const std::size_t row_bytes = static_cast<std::size_t>(intermediate / 256) * 98;
    const uint8_t *matrix = payloads[pair] + kScalarPayloadHeaderBytes + gate_up_len;
    const uint8_t *encoded = matrix + static_cast<std::size_t>(row) * row_bytes;
    const float *input = activated + static_cast<std::size_t>(pair) * intermediate;
    const float value = iq3xxs_warp_partial_dot(encoded, intermediate, input);
    if (threadIdx.x == 0) {
        pair_output[pair * hidden + row] = value;
    }
}

__global__ void iquant_down_iq3xxs_q8_warp_route_kernel(
    const uint8_t * const *payloads,
    int pair_count,
    int hidden,
    int intermediate,
    std::size_t gate_up_len,
    const block_q8_1 *activated,
    float *pair_output
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / hidden;
    const int row = row_index % hidden;
    if (pair >= pair_count) return;
    const std::size_t row_bytes = static_cast<std::size_t>(intermediate / 256) * 98;
    const uint8_t *matrix = payloads[pair] + kScalarPayloadHeaderBytes + gate_up_len;
    const uint8_t *encoded = matrix + static_cast<std::size_t>(row) * row_bytes;
    const block_q8_1 *input = activated +
        static_cast<std::size_t>(pair) * (intermediate / QK8_1);
    const float value = iq3xxs_warp_partial_dot_q8_1(encoded, intermediate, input);
    if (threadIdx.x == 0) {
        pair_output[pair * hidden + row] = value;
    }
}

__device__ float heterogeneous_quant_partial_dot(
    const CudaHeterogeneousTensor &tensor,
    int row,
    const float *input,
    float scale,
    float input_scale
) {
    if (tensor.dtype == 2 || tensor.dtype == 10 || tensor.dtype == 11 || tensor.dtype == 12 ||
        tensor.dtype == 13 || tensor.dtype == 14 ||
        tensor.dtype == 18 || tensor.dtype == 21 || tensor.dtype == 22 ||
        tensor.dtype == 23 || tensor.dtype == 39) {
        float value = 0.0f;
        for (int column = threadIdx.x; column < tensor.columns; column += blockDim.x) {
            float weight;
            if (tensor.dtype == 2) {
                const std::size_t row_bytes = (tensor.columns / 32) * 18;
                weight = q4_0_value(tensor.data + row * row_bytes, column);
            } else if (tensor.dtype == 10 || tensor.dtype == 11 || tensor.dtype == 12 ||
                       tensor.dtype == 13 || tensor.dtype == 14) {
                weight = k_value(tensor.dtype, tensor.data, row, column, tensor.columns);
            } else if (tensor.dtype == 39) {
                const std::size_t row_bytes = (tensor.columns / 32) * 17;
                weight = mxfp4_value(tensor.data + row * row_bytes, column);
            } else {
                const std::size_t block_bytes = tensor.dtype == 18 ? 98 :
                    tensor.dtype == 21 ? 110 : tensor.dtype == 22 ? 82 : 136;
                const uint8_t *encoded = tensor.data + row * (tensor.columns / 256) * block_bytes;
                weight = tensor.dtype == 18
                    ? iq3xxs_value(encoded, column)
                    : tensor.dtype == 21
                    ? iq3s_value(encoded, column)
                    : tensor.dtype == 22 ? iq2s_value(encoded, column) : iq4xs_value(encoded, column);
            }
            value += weight * input[column];
        }
        return value * scale * input_scale;
    }
    return threadIdx.x == 0
        ? heterogeneous_quant_dot(tensor, row, tensor.columns, input, scale, input_scale)
        : 0.0f;
}

__global__ void heterogeneous_quant_gate_up_route_kernel(
    const CudaHeterogeneousExpert * experts,
    const uint32_t * token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const float * hidden_values,
    float * gate_up_values
) {
    const int rows_per_pair = 2 * intermediate;
    const int row_index = blockIdx.x;
    const int pair = row_index / rows_per_pair;
    const int row = row_index % rows_per_pair;
    if (pair >= pair_count) return;
    const CudaHeterogeneousExpert expert = experts[pair];
    const float * input = hidden_values + static_cast<std::size_t>(token_ids[pair]) * hidden;
    const int component_row = row % intermediate;
    float value;
    if (expert.gate_up.data != nullptr) {
        const bool is_up = row >= intermediate;
        value = heterogeneous_quant_partial_dot(
            expert.gate_up,
            is_up ? intermediate + component_row : component_row,
            input,
            is_up ? expert.gate_up.secondary_scale : expert.gate_up.scale,
            is_up ? expert.gate_up.secondary_input_scale : expert.gate_up.input_scale);
    } else {
        const bool is_up = row >= intermediate;
        const CudaHeterogeneousTensor &tensor = is_up ? expert.up : expert.gate;
        value = heterogeneous_quant_partial_dot(
            tensor, component_row, input, tensor.scale, tensor.input_scale);
    }
    __shared__ float gate_partials[128];
    gate_partials[threadIdx.x] = value;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) {
            gate_partials[threadIdx.x] += gate_partials[threadIdx.x + width];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate_up_values[pair * rows_per_pair + row] = gate_partials[0];
    }
}

__device__ float heterogeneous_activation(float value, int activation) {
    if (activation == 2) {
        if (value == -CUDART_INF_F) {
            return 0.0f;
        }
        return 0.5f * value *
            (1.0f + tanhf(0.7978845608028654f * value *
                (1.0f + 0.044715f * value * value)));
    }
    if (value == -CUDART_INF_F) {
        return 0.0f;
    }
    const float value_exp = value < 0.0f ? expf(value) : expf(-value);
    return value < 0.0f
        ? value * value_exp / (1.0f + value_exp)
        : value / (1.0f + value_exp);
}

__global__ void heterogeneous_activation_kernel(
    const float * gate_up_values,
    int pair_count,
    int intermediate,
    int activation,
    float * activated
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair = index / intermediate;
    const int row = index % intermediate;
    if (pair >= pair_count) {
        return;
    }
    const float * gate_up = gate_up_values + pair * 2 * intermediate;
    activated[pair * intermediate + row] =
        heterogeneous_activation(gate_up[row], activation) * gate_up[intermediate + row];
}

__global__ void heterogeneous_quant_down_route_kernel(
    const CudaHeterogeneousExpert * experts,
    int pair_count,
    int hidden,
    int intermediate,
    const float * activated,
    float * pair_output
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / hidden;
    const int row = row_index % hidden;
    if (pair >= pair_count) return;
    const CudaHeterogeneousExpert expert = experts[pair];
    const float *input = activated + pair * intermediate;
    const float value = heterogeneous_quant_partial_dot(
        expert.down, row, input, expert.down.scale, expert.down.input_scale);
    __shared__ float partials[128];
    partials[threadIdx.x] = value;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width) partials[threadIdx.x] += partials[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) pair_output[pair * hidden + row] = partials[0];
}

// The heterogeneous path must use the same MMVQ primitives as llama.cpp.
// Its tensors may use different quantizers per role, but every supported
// quantizer has a canonical q8_1 dot product in vecdotq.cuh. Keep the dtype
// dispatch outside the block loop and call each primitive directly: a device
// function pointer in that loop prevents the compiler from inlining the
// canonical dp4a/codebook implementation and was measurable on the mixed
// dtype Mistral route.
#define DEFINE_HETEROGENEOUS_Q8_DOT(NAME, QK_VALUE, QI_VALUE, VDR_VALUE, DOT_FN) \
__device__ __forceinline__ float heterogeneous_quant_partial_dot_q8_1_##NAME( \
    const CudaHeterogeneousTensor &tensor, \
    int row, \
    const block_q8_1 *input, \
    float scale, \
    float input_scale \
) { \
    constexpr int qk = QK_VALUE; \
    constexpr int qi = QI_VALUE; \
    constexpr int vdr = VDR_VALUE; \
    const int lane = threadIdx.x; \
    const int lanes_per_block = qi / vdr; \
    const int blocks_per_iteration = vdr * 32 / qi; \
    const int blocks_per_row = static_cast<int>(tensor.columns) / qk; \
    const std::size_t row_bytes = tensor.bytes / tensor.rows; \
    const uint8_t *row_data = tensor.data + static_cast<std::size_t>(row) * row_bytes; \
    float result = 0.0f; \
    for (int block_index = lane / lanes_per_block; \
         block_index < blocks_per_row; \
         block_index += blocks_per_iteration) { \
        const int q8_block = block_index * (qk / QK8_1); \
        const int q8_subblock = vdr * (lane % lanes_per_block); \
        result += DOT_FN( \
            row_data, \
            input + q8_block, \
            block_index, \
            q8_subblock); \
    } \
    for (int offset = 16; offset > 0; offset >>= 1) { \
        result += __shfl_down_sync(0xffffffffu, result, offset); \
    } \
    return result * scale * input_scale; \
}

DEFINE_HETEROGENEOUS_Q8_DOT(q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(q2_k, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(q3_k, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(q4_k, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(q5_k, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(q6_k, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(iq3_xxs, QK_K, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(iq3_s, QK_K, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(iq2_s, QK_K, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(iq4_xs, QK_K, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1)
DEFINE_HETEROGENEOUS_Q8_DOT(mxfp4, QK_MXFP4, QI_MXFP4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1)

__device__ __forceinline__ float heterogeneous_quant_partial_dot_q8_1_direct(
    const CudaHeterogeneousTensor &tensor,
    int row,
    const block_q8_1 *input,
    float scale,
    float input_scale
) {
    switch (tensor.dtype) {
        case 2:
            return heterogeneous_quant_partial_dot_q8_1_q4_0(
                tensor, row, input, scale, input_scale);
        case 10:
            return heterogeneous_quant_partial_dot_q8_1_q2_k(
                tensor, row, input, scale, input_scale);
        case 11:
            return heterogeneous_quant_partial_dot_q8_1_q3_k(
                tensor, row, input, scale, input_scale);
        case 12:
            return heterogeneous_quant_partial_dot_q8_1_q4_k(
                tensor, row, input, scale, input_scale);
        case 13:
            return heterogeneous_quant_partial_dot_q8_1_q5_k(
                tensor, row, input, scale, input_scale);
        case 14:
            return heterogeneous_quant_partial_dot_q8_1_q6_k(
                tensor, row, input, scale, input_scale);
        case 18:
            return heterogeneous_quant_partial_dot_q8_1_iq3_xxs(
                tensor, row, input, scale, input_scale);
        case 21:
            return heterogeneous_quant_partial_dot_q8_1_iq3_s(
                tensor, row, input, scale, input_scale);
        case 22:
            return heterogeneous_quant_partial_dot_q8_1_iq2_s(
                tensor, row, input, scale, input_scale);
        case 23:
            return heterogeneous_quant_partial_dot_q8_1_iq4_xs(
                tensor, row, input, scale, input_scale);
        case 39:
            return heterogeneous_quant_partial_dot_q8_1_mxfp4(
                tensor, row, input, scale, input_scale);
        default:
            return 0.0f;
    }
}

#undef DEFINE_HETEROGENEOUS_Q8_DOT

__global__ void heterogeneous_quant_gate_up_route_q8_kernel(
    const CudaHeterogeneousExpert *experts,
    const uint32_t *token_ids,
    int pair_count,
    int intermediate,
    int hidden,
    const block_q8_1 *hidden_values,
    float *gate_up_values
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / intermediate;
    const int row = row_index % intermediate;
    if (pair >= pair_count) return;
    const CudaHeterogeneousExpert expert = experts[pair];
    const block_q8_1 *input = hidden_values +
        static_cast<std::size_t>(token_ids[pair]) * (hidden / QK8_1);
    float gate;
    float up;
    if (expert.gate_up.data != nullptr) {
        gate = heterogeneous_quant_partial_dot_q8_1_direct(
            expert.gate_up, row, input,
            expert.gate_up.scale, expert.gate_up.input_scale);
        up = heterogeneous_quant_partial_dot_q8_1_direct(
            expert.gate_up, row + intermediate, input,
            expert.gate_up.secondary_scale, expert.gate_up.secondary_input_scale);
    } else {
        gate = heterogeneous_quant_partial_dot_q8_1_direct(
            expert.gate, row, input, expert.gate.scale, expert.gate.input_scale);
        up = heterogeneous_quant_partial_dot_q8_1_direct(
            expert.up, row, input, expert.up.scale, expert.up.input_scale);
    }
    if (threadIdx.x == 0) {
        const int output_offset = pair * 2 * intermediate + row;
        gate_up_values[output_offset] = gate;
        gate_up_values[output_offset + intermediate] = up;
    }
}

__global__ void heterogeneous_quant_down_route_q8_kernel(
    const CudaHeterogeneousExpert *experts,
    int pair_count,
    int hidden,
    int intermediate,
    const block_q8_1 *activated,
    float *pair_output
) {
    const int row_index = blockIdx.x;
    const int pair = row_index / hidden;
    const int row = row_index % hidden;
    if (pair >= pair_count) return;
    const block_q8_1 *input = activated +
        static_cast<std::size_t>(pair) * (intermediate / QK8_1);
    const CudaHeterogeneousTensor down = experts[pair].down;
    const float value = heterogeneous_quant_partial_dot_q8_1_direct(
        down, row, input, down.scale, down.input_scale);
    if (threadIdx.x == 0) {
        pair_output[pair * hidden + row] = value;
    }
}

int cuda_status(cudaError_t error) {
    return error == cudaSuccess ? 0 : static_cast<int>(error);
}

bool first_nonfinite(
    const float *values,
    std::size_t count,
    std::size_t *index,
    std::size_t *nonfinite_count
) {
    std::size_t found = 0;
    for (std::size_t current = 0; current < count; ++current) {
        if (!std::isfinite(values[current])) {
            if (found == 0 && index != nullptr) {
                *index = current;
            }
            found += 1;
        }
    }
    if (nonfinite_count != nullptr) {
        *nonfinite_count = found;
    }
    return found != 0;
}

bool device_has_nonfinite(
    const float *device_values,
    std::size_t count,
    std::vector<float> &host_values,
    int *copy_status,
    std::size_t *index,
    std::size_t *nonfinite_count
) {
    host_values.resize(count);
    *copy_status = cuda_status(cudaMemcpy(
        host_values.data(),
        device_values,
        count * sizeof(float),
        cudaMemcpyDeviceToHost));
    return *copy_status == 0 && first_nonfinite(
        host_values.data(), count, index, nonfinite_count);
}

struct Nvfp4RouteSlot {
    cudaStream_t stream = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_end = nullptr;
    cudaEvent_t gate_start = nullptr;
    cudaEvent_t gate_end = nullptr;
    cudaEvent_t down_start = nullptr;
    cudaEvent_t down_end = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_end = nullptr;
    float * d_hidden = nullptr;
    float * d_hidden_scales = nullptr;
    int8_t * d_hidden_qs = nullptr;
    const uint8_t ** d_payloads = nullptr;
    uint32_t * d_token_ids = nullptr;
    float * d_weights = nullptr;
    std::size_t * d_token_offsets = nullptr;
    float * d_gate_up_values = nullptr;
    float * d_activated = nullptr;
    float * d_activated_scales = nullptr;
    int8_t * d_activated_qs = nullptr;
    float * d_pair_output = nullptr;
    float * d_output = nullptr;
    float * h_hidden = nullptr;
    const uint8_t ** h_payloads = nullptr;
    uint32_t * h_token_ids = nullptr;
    float * h_weights = nullptr;
    std::size_t * h_token_offsets = nullptr;
    float * h_output = nullptr;
};

struct Nvfp4RoutePipeline {
    int device_index;
    std::size_t hidden_width;
    std::size_t intermediate_width;
    std::size_t token_capacity = 0;
    std::size_t route_capacity = 0;
    std::size_t next_slot = 0;
    Nvfp4RouteSlot slots[2];
};

void destroy_route_events(Nvfp4RouteSlot *slot) {
    if (slot->d2h_end != nullptr) cudaEventDestroy(slot->d2h_end);
    if (slot->d2h_start != nullptr) cudaEventDestroy(slot->d2h_start);
    if (slot->down_end != nullptr) cudaEventDestroy(slot->down_end);
    if (slot->down_start != nullptr) cudaEventDestroy(slot->down_start);
    if (slot->gate_end != nullptr) cudaEventDestroy(slot->gate_end);
    if (slot->gate_start != nullptr) cudaEventDestroy(slot->gate_start);
    if (slot->h2d_end != nullptr) cudaEventDestroy(slot->h2d_end);
    if (slot->h2d_start != nullptr) cudaEventDestroy(slot->h2d_start);
    slot->d2h_end = nullptr;
    slot->d2h_start = nullptr;
    slot->down_end = nullptr;
    slot->down_start = nullptr;
    slot->gate_end = nullptr;
    slot->gate_start = nullptr;
    slot->h2d_end = nullptr;
    slot->h2d_start = nullptr;
}

void release_route_slot(Nvfp4RouteSlot *slot) {
    if (slot->stream != nullptr) {
        cudaStreamSynchronize(slot->stream);
    }
    destroy_route_events(slot);
    if (slot->d_output != nullptr) cudaFree(slot->d_output);
    if (slot->d_pair_output != nullptr) cudaFree(slot->d_pair_output);
    if (slot->d_activated_qs != nullptr) cudaFree(slot->d_activated_qs);
    if (slot->d_activated_scales != nullptr) cudaFree(slot->d_activated_scales);
    if (slot->d_activated != nullptr) cudaFree(slot->d_activated);
    if (slot->d_gate_up_values != nullptr) cudaFree(slot->d_gate_up_values);
    if (slot->d_token_offsets != nullptr) cudaFree(slot->d_token_offsets);
    if (slot->d_weights != nullptr) cudaFree(slot->d_weights);
    if (slot->d_token_ids != nullptr) cudaFree(slot->d_token_ids);
    if (slot->d_payloads != nullptr) cudaFree(slot->d_payloads);
    if (slot->d_hidden_qs != nullptr) cudaFree(slot->d_hidden_qs);
    if (slot->d_hidden_scales != nullptr) cudaFree(slot->d_hidden_scales);
    if (slot->d_hidden != nullptr) cudaFree(slot->d_hidden);
    if (slot->h_output != nullptr) cudaFreeHost(slot->h_output);
    if (slot->h_token_offsets != nullptr) cudaFreeHost(slot->h_token_offsets);
    if (slot->h_weights != nullptr) cudaFreeHost(slot->h_weights);
    if (slot->h_token_ids != nullptr) cudaFreeHost(slot->h_token_ids);
    if (slot->h_payloads != nullptr) cudaFreeHost(slot->h_payloads);
    if (slot->h_hidden != nullptr) cudaFreeHost(slot->h_hidden);
    if (slot->stream != nullptr) cudaStreamDestroy(slot->stream);
    *slot = {};
}

int allocate_route_slot(
    Nvfp4RouteSlot *slot,
    std::size_t hidden,
    std::size_t intermediate,
    std::size_t tokens,
    std::size_t routes
) {
    const std::size_t hidden_values = tokens * hidden;
    const std::size_t hidden_scales = tokens * (hidden / 32);
    const std::size_t gate_up_values = 2 * intermediate * routes;
    const std::size_t activated_values = intermediate * routes;
    const std::size_t activated_scales = routes * (intermediate / 32);
    const std::size_t pair_output = hidden * routes;

    int result = cuda_status(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_hidden),
        hidden_values * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_hidden_scales),
        hidden_scales * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_hidden_qs),
        hidden_values * sizeof(int8_t)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_payloads),
        routes * sizeof(*slot->d_payloads)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_token_ids),
        routes * sizeof(*slot->d_token_ids)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_weights),
        routes * sizeof(*slot->d_weights)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_token_offsets),
        (tokens + 1) * sizeof(*slot->d_token_offsets)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_gate_up_values),
        gate_up_values * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_activated),
        activated_values * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_activated_scales),
        activated_scales * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_activated_qs),
        activated_values * sizeof(int8_t)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_pair_output),
        pair_output * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&slot->d_output),
        hidden_values * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMallocHost(reinterpret_cast<void **>(&slot->h_hidden),
        hidden_values * sizeof(float)));
    if (result != 0) return result;
    result = cuda_status(cudaMallocHost(reinterpret_cast<void **>(&slot->h_payloads),
        routes * sizeof(*slot->h_payloads)));
    if (result != 0) return result;
    result = cuda_status(cudaMallocHost(reinterpret_cast<void **>(&slot->h_token_ids),
        routes * sizeof(*slot->h_token_ids)));
    if (result != 0) return result;
    result = cuda_status(cudaMallocHost(reinterpret_cast<void **>(&slot->h_weights),
        routes * sizeof(*slot->h_weights)));
    if (result != 0) return result;
    result = cuda_status(cudaMallocHost(reinterpret_cast<void **>(&slot->h_token_offsets),
        (tokens + 1) * sizeof(*slot->h_token_offsets)));
    if (result != 0) return result;
    return cuda_status(cudaMallocHost(reinterpret_cast<void **>(&slot->h_output),
        hidden_values * sizeof(float)));
}

int ensure_route_pipeline_capacity(
    Nvfp4RoutePipeline *pipeline,
    std::size_t tokens,
    std::size_t routes
) {
    if (pipeline->token_capacity >= tokens && pipeline->route_capacity >= routes) {
        return 0;
    }
    release_route_slot(&pipeline->slots[0]);
    release_route_slot(&pipeline->slots[1]);
    pipeline->token_capacity = tokens;
    pipeline->route_capacity = routes;
    int result = allocate_route_slot(&pipeline->slots[0], pipeline->hidden_width,
        pipeline->intermediate_width, tokens, routes);
    if (result != 0) {
        release_route_slot(&pipeline->slots[0]);
        pipeline->token_capacity = 0;
        pipeline->route_capacity = 0;
        return result;
    }
    result = allocate_route_slot(&pipeline->slots[1], pipeline->hidden_width,
        pipeline->intermediate_width, tokens, routes);
    if (result != 0) {
        release_route_slot(&pipeline->slots[0]);
        release_route_slot(&pipeline->slots[1]);
        pipeline->token_capacity = 0;
        pipeline->route_capacity = 0;
    }
    return result;
}

int ensure_route_timing_events(Nvfp4RouteSlot *slot) {
    int result = cuda_status(cudaEventCreate(&slot->h2d_start));
    if (result != 0) return result;
    result = cuda_status(cudaEventCreate(&slot->h2d_end));
    if (result != 0) return result;
    result = cuda_status(cudaEventCreate(&slot->gate_start));
    if (result != 0) return result;
    result = cuda_status(cudaEventCreate(&slot->gate_end));
    if (result != 0) return result;
    result = cuda_status(cudaEventCreate(&slot->down_start));
    if (result != 0) return result;
    result = cuda_status(cudaEventCreate(&slot->down_end));
    if (result != 0) return result;
    result = cuda_status(cudaEventCreate(&slot->d2h_start));
    if (result != 0) return result;
    return cuda_status(cudaEventCreate(&slot->d2h_end));
}

void copy_route_chunk_to_host(
    Nvfp4RouteSlot *slot,
    const uint8_t * const *payloads,
    const uint32_t *token_ids,
    const float *weights,
    const std::size_t *token_offsets,
    const float *hidden,
    std::size_t token_start,
    std::size_t chunk_tokens,
    std::size_t route_start,
    std::size_t route_count,
    std::size_t hidden_width
) {
    std::memcpy(slot->h_hidden, hidden + token_start * hidden_width,
        chunk_tokens * hidden_width * sizeof(float));
    for (std::size_t route = 0; route < route_count; ++route) {
        const std::size_t source = route_start + route;
        slot->h_payloads[route] = payloads[source];
        slot->h_token_ids[route] = static_cast<uint32_t>(token_ids[source] - token_start);
        slot->h_weights[route] = weights[source];
    }
    for (std::size_t token = 0; token <= chunk_tokens; ++token) {
        slot->h_token_offsets[token] = token_offsets[token_start + token] - route_start;
    }
}

int enqueue_route_chunk(
    Nvfp4RouteSlot *slot,
    std::size_t chunk_tokens,
    std::size_t route_count,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    CudaMoeCallTiming *timing
) {
    int result = 0;
    if (timing != nullptr && slot->h2d_start == nullptr) {
        result = ensure_route_timing_events(slot);
        if (result != 0) return result;
    }
    if (timing != nullptr) {
        result = cuda_status(cudaEventRecord(slot->h2d_start, slot->stream));
        if (result != 0) return result;
    }
    result = cuda_status(cudaMemcpyAsync(slot->d_hidden, slot->h_hidden,
        chunk_tokens * hidden_width * sizeof(float), cudaMemcpyHostToDevice, slot->stream));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpyAsync(slot->d_payloads, slot->h_payloads,
        route_count * sizeof(*slot->d_payloads), cudaMemcpyHostToDevice, slot->stream));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpyAsync(slot->d_token_ids, slot->h_token_ids,
        route_count * sizeof(*slot->d_token_ids), cudaMemcpyHostToDevice, slot->stream));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpyAsync(slot->d_weights, slot->h_weights,
        route_count * sizeof(*slot->d_weights), cudaMemcpyHostToDevice, slot->stream));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpyAsync(slot->d_token_offsets, slot->h_token_offsets,
        (chunk_tokens + 1) * sizeof(*slot->d_token_offsets), cudaMemcpyHostToDevice, slot->stream));
    if (result != 0) return result;
    if (timing != nullptr) {
        result = cuda_status(cudaEventRecord(slot->h2d_end, slot->stream));
        if (result != 0) return result;
        result = cuda_status(cudaEventRecord(slot->gate_start, slot->stream));
        if (result != 0) return result;
    }

    const int block_size = 128;
    const auto gate_launch_start = Clock::now();
    quantize_q8_0_kernel<<<chunk_tokens * ((hidden_width + 31) / 32), 32, 0, slot->stream>>>(
        slot->d_hidden, static_cast<int>(hidden_width), slot->d_hidden_scales,
        slot->d_hidden_qs, static_cast<int>(chunk_tokens));
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    gate_up_route_batch_resident_kernel<<<
        (2 * intermediate_width * route_count + block_size - 1) / block_size,
        block_size, 0, slot->stream>>>(
            slot->d_payloads, slot->d_token_ids, static_cast<int>(route_count),
            static_cast<int>(intermediate_width), static_cast<int>(hidden_width),
            slot->d_hidden_qs, slot->d_hidden_scales, slot->d_gate_up_values);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    if (timing != nullptr) {
        result = cuda_status(cudaEventRecord(slot->gate_end, slot->stream));
        if (result != 0) return result;
        result = cuda_status(cudaEventRecord(slot->down_start, slot->stream));
        if (result != 0) return result;
    }
    const auto down_launch_start = Clock::now();
    geglu_kernel<<<(intermediate_width * route_count + block_size - 1) / block_size,
        block_size, 0, slot->stream>>>(
            slot->d_gate_up_values, static_cast<int>(route_count),
            static_cast<int>(intermediate_width), slot->d_activated);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    quantize_q8_0_kernel<<<((intermediate_width + 31) / 32) * route_count, 32, 0, slot->stream>>>(
        slot->d_activated, static_cast<int>(intermediate_width), slot->d_activated_scales,
        slot->d_activated_qs, static_cast<int>(route_count));
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    down_route_batch_resident_kernel<<<
        (hidden_width * route_count + block_size - 1) / block_size,
        block_size, 0, slot->stream>>>(
            slot->d_payloads, static_cast<int>(route_count), static_cast<int>(hidden_width),
            static_cast<int>(intermediate_width), slot->d_activated_qs,
            slot->d_activated_scales, slot->d_pair_output);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    aggregate_route_batch_kernel<<<
        (chunk_tokens * hidden_width + block_size - 1) / block_size,
        block_size, 0, slot->stream>>>(
            slot->d_pair_output, slot->d_weights, slot->d_token_offsets,
            static_cast<int>(chunk_tokens), static_cast<int>(hidden_width), slot->d_output);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    if (timing != nullptr) {
        timing->launch_ms += elapsed_ms(gate_launch_start, Clock::now());
        result = cuda_status(cudaEventRecord(slot->down_end, slot->stream));
        if (result != 0) return result;
    }
    if (timing != nullptr) {
        result = cuda_status(cudaEventRecord(slot->d2h_start, slot->stream));
        if (result != 0) return result;
    }
    result = cuda_status(cudaMemcpyAsync(slot->h_output, slot->d_output,
        chunk_tokens * hidden_width * sizeof(float), cudaMemcpyDeviceToHost, slot->stream));
    if (result != 0) return result;
    if (timing != nullptr) {
        result = cuda_status(cudaEventRecord(slot->d2h_end, slot->stream));
    }
    return result;
}

int collect_route_chunk(
    Nvfp4RouteSlot *slot,
    std::size_t chunk_tokens,
    std::size_t route_count,
    std::size_t hidden_width,
    CudaMoeCallTiming *timing
) {
    const auto sync_start = Clock::now();
    int result = cuda_status(cudaStreamSynchronize(slot->stream));
    if (result != 0) return result;
    const double sync_ms = elapsed_ms(sync_start, Clock::now());
    if (timing != nullptr) {
        float h2d_ms = 0.0f;
        float gate_ms = 0.0f;
        float down_ms = 0.0f;
        float d2h_ms = 0.0f;
        result = cuda_status(cudaEventElapsedTime(&h2d_ms, slot->h2d_start, slot->h2d_end));
        if (result != 0) return result;
        result = cuda_status(cudaEventElapsedTime(&gate_ms, slot->gate_start, slot->gate_end));
        if (result != 0) return result;
        result = cuda_status(cudaEventElapsedTime(&down_ms, slot->down_start, slot->down_end));
        if (result != 0) return result;
        result = cuda_status(cudaEventElapsedTime(&d2h_ms, slot->d2h_start, slot->d2h_end));
        if (result != 0) return result;
        timing->h2d_ms += h2d_ms;
        timing->h2d_bytes += static_cast<uint64_t>(
            chunk_tokens * hidden_width * sizeof(float) +
            route_count * sizeof(*slot->d_payloads) +
            route_count * sizeof(*slot->d_token_ids) +
            route_count * sizeof(*slot->d_weights) +
            (chunk_tokens + 1) * sizeof(*slot->d_token_offsets));
        const double chunk_kernel_ms = static_cast<double>(gate_ms + down_ms);
        const double chunk_transfer_ms = static_cast<double>(h2d_ms + d2h_ms);
        timing->kernel_ms += chunk_kernel_ms;
        timing->kernel_span_ms = std::max(timing->kernel_span_ms, chunk_kernel_ms);
        timing->d2h_ms += d2h_ms;
        timing->d2h_bytes += static_cast<uint64_t>(
            chunk_tokens * hidden_width * sizeof(float));
        timing->transfer_span_ms = std::max(timing->transfer_span_ms, chunk_transfer_ms);
        timing->sync_ms += std::max(0.0, sync_ms - chunk_kernel_ms);
    }
    return 0;
}

} // namespace

extern "C" int pom_moe_kquant_forward_route_batch_resident(
    int device_index,
    uint32_t dtype,
    const uint8_t * const *payloads,
    std::size_t payload_count,
    const std::size_t *payload_lengths,
    const uint32_t *token_ids,
    const float *weights,
    const std::size_t *token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float *hidden,
    std::size_t hidden_len,
    float *output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    int *nonfinite_stage,
    std::size_t *nonfinite_index,
    std::size_t *nonfinite_count,
    CudaMoeCallTiming *timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (nonfinite_stage != nullptr) {
        *nonfinite_stage = 0;
    }
    if (nonfinite_index != nullptr) {
        *nonfinite_index = 0;
    }
    if (nonfinite_count != nullptr) {
        *nonfinite_count = 0;
    }
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(payload_count);
        timing->kernel_launches = 4;
        timing->sync_count = 1;
    }
    if ((dtype != 10 && dtype != 11 && dtype != 12 && dtype != 13 && dtype != 14) ||
        payloads == nullptr || payload_lengths == nullptr ||
        token_ids == nullptr || weights == nullptr || token_offsets == nullptr ||
        hidden == nullptr || output == nullptr || payload_count == 0 || token_count == 0 ||
        hidden_width == 0 || intermediate_width == 0 || hidden_width % 256 != 0 ||
        intermediate_width % 256 != 0 ||
        payload_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        token_count > std::numeric_limits<std::size_t>::max() / hidden_width ||
        hidden_width > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        intermediate_width > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        token_offsets_len != token_count + 1 || hidden_len != token_count * hidden_width ||
        output_len != hidden_len || token_offsets[0] != 0 ||
        token_offsets[token_count] != payload_count) {
        return 1;
    }
    std::size_t expected_gate_up_bytes = 0;
    std::size_t expected_payload_bytes = 0;
    if (!kquant_payload_geometry(
            dtype,
            hidden_width,
            intermediate_width,
            &expected_gate_up_bytes,
            &expected_payload_bytes)) {
        return 1;
    }
    for (std::size_t pair = 0; pair < payload_count; ++pair) {
        if (payloads[pair] == nullptr || token_ids[pair] >= token_count ||
            !std::isfinite(weights[pair])) {
            return 1;
        }
        if (payload_lengths[pair] != expected_payload_bytes) {
            return 2;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) {
            return 1;
        }
    }

    const int pair_count = static_cast<int>(payload_count);
    const int hidden_i = static_cast<int>(hidden_width);
    const int intermediate_i = static_cast<int>(intermediate_width);
    const int block_size = 128;
    const std::size_t hidden_bytes = hidden_width * token_count * sizeof(float);
    const std::size_t gate_up_bytes = 2 * intermediate_width * payload_count * sizeof(float);
    const std::size_t activated_bytes = intermediate_width * payload_count * sizeof(float);
    const std::size_t pair_output_bytes = hidden_width * payload_count * sizeof(float);

    float *d_hidden = nullptr;
    const uint8_t **d_payloads = nullptr;
    uint32_t *d_token_ids = nullptr;
    float *d_weights = nullptr;
    std::size_t *d_token_offsets = nullptr;
    float *d_gate_up = nullptr;
    float *d_activated = nullptr;
    float *d_pair_output = nullptr;
    float *d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    int result = 0;

    do {
        const auto alloc_start = Clock::now();
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_payloads),
            payload_count * sizeof(*d_payloads)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_ids),
            payload_count * sizeof(*d_token_ids)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_weights),
            payload_count * sizeof(*d_weights)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_offsets),
            token_offsets_len * sizeof(*d_token_offsets)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up), gate_up_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated), activated_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_pair_output), pair_output_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output), hidden_bytes));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->alloc_ms = elapsed_ms(alloc_start, Clock::now());
            result = cuda_status(cudaEventCreate(&start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&end));
            if (result != 0) break;
        }

        const auto h2d_start = Clock::now();
        result = cuda_status(cudaMemcpy(d_hidden, hidden, hidden_bytes, cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_payloads, payloads,
            payload_count * sizeof(*d_payloads), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_ids, token_ids,
            payload_count * sizeof(*d_token_ids), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_weights, weights,
            payload_count * sizeof(*d_weights), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_offsets, token_offsets,
            token_offsets_len * sizeof(*d_token_offsets), cudaMemcpyHostToDevice));
        if (timing != nullptr) {
            timing->h2d_ms = elapsed_ms(h2d_start, Clock::now());
            timing->h2d_bytes = static_cast<uint64_t>(
                hidden_bytes + payload_count * sizeof(*d_payloads) +
                payload_count * sizeof(*d_token_ids) + payload_count * sizeof(*d_weights) +
                token_offsets_len * sizeof(*d_token_offsets));
        }
        if (result != 0) break;

        const auto launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(start));
            if (result != 0) break;
        }
        kquant_gate_up_route_kernel<<<
            (2 * intermediate_i * pair_count + block_size - 1) / block_size,
            block_size>>>(dtype, d_payloads, d_token_ids, pair_count, intermediate_i,
                hidden_i, d_hidden, d_gate_up);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        kquant_silu_kernel<<<
            (intermediate_i * pair_count + block_size - 1) / block_size,
            block_size>>>(d_gate_up, pair_count, intermediate_i, d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        kquant_down_route_kernel<<<
            (hidden_i * pair_count + block_size - 1) / block_size,
            block_size>>>(dtype, d_payloads, pair_count, hidden_i, intermediate_i,
                expected_gate_up_bytes, d_activated, d_pair_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        aggregate_route_batch_kernel<<<
            (static_cast<int>(token_count) * hidden_i + block_size - 1) / block_size,
            block_size>>>(d_pair_output, d_weights, d_token_offsets,
                static_cast<int>(token_count), hidden_i, d_output);
        result = cuda_status(cudaGetLastError());
        if (timing != nullptr) {
            timing->launch_ms = elapsed_ms(launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(end));
            if (result != 0) break;
        }
        const auto sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, start, end));
            if (result != 0) break;
            timing->kernel_ms = kernel_ms;
            timing->sync_ms = std::max(0.0, elapsed_ms(sync_start, Clock::now()) -
                static_cast<double>(kernel_ms));
        }

        const auto d2h_start = Clock::now();
        result = cuda_status(cudaMemcpy(output, d_output, hidden_bytes, cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_start, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(hidden_bytes);
        }
        if (result == 0 && first_nonfinite(
                output,
                hidden_bytes / sizeof(float),
                nonfinite_index,
                nonfinite_count)) {
            std::vector<float> stage_values;
            int copy_status = 0;
            if (device_has_nonfinite(
                    d_gate_up,
                    gate_up_bytes / sizeof(float),
                    stage_values,
                    &copy_status,
                    nullptr,
                    nullptr)) {
                if (nonfinite_stage != nullptr) *nonfinite_stage = 1;
                result = 0;
                break;
            }
            if (copy_status != 0) {
                result = copy_status;
                break;
            }
            if (device_has_nonfinite(
                    d_activated,
                    activated_bytes / sizeof(float),
                    stage_values,
                    &copy_status,
                    nullptr,
                    nullptr)) {
                if (nonfinite_stage != nullptr) *nonfinite_stage = 2;
                result = 0;
                break;
            }
            if (copy_status != 0) {
                result = copy_status;
                break;
            }
            if (device_has_nonfinite(
                    d_pair_output,
                    pair_output_bytes / sizeof(float),
                    stage_values,
                    &copy_status,
                    nullptr,
                    nullptr)) {
                if (nonfinite_stage != nullptr) *nonfinite_stage = 3;
                result = 0;
                break;
            }
            if (copy_status != 0) {
                result = copy_status;
                break;
            }
            if (nonfinite_stage != nullptr) *nonfinite_stage = 4;
        }
    } while (false);

    if (start != nullptr) cudaEventDestroy(start);
    if (end != nullptr) cudaEventDestroy(end);
    const auto free_start = Clock::now();
    cudaFree(d_output);
    cudaFree(d_pair_output);
    cudaFree(d_activated);
    cudaFree(d_gate_up);
    cudaFree(d_token_offsets);
    cudaFree(d_weights);
    cudaFree(d_token_ids);
    cudaFree(d_payloads);
    cudaFree(d_hidden);
    if (timing != nullptr) {
        timing->free_ms = elapsed_ms(free_start, Clock::now());
    }
    return result;
}

extern "C" int pom_moe_plain_forward_route_batch_resident(
    int device_index,
    uint32_t dtype,
    const uint8_t * const *payloads,
    std::size_t payload_count,
    const uint32_t *token_ids,
    const float *weights,
    const std::size_t *token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float *hidden,
    std::size_t hidden_len,
    float *output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    CudaMoeCallTiming *timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(payload_count);
        timing->kernel_launches = 4;
        timing->sync_count = 1;
    }
    if ((dtype != 0 && dtype != 1 && dtype != 30) || payloads == nullptr ||
        token_ids == nullptr || weights == nullptr || token_offsets == nullptr ||
        hidden == nullptr || output == nullptr || payload_count == 0 || token_count == 0 ||
        hidden_width == 0 || intermediate_width == 0 ||
        token_offsets_len != token_count + 1 || hidden_len != token_count * hidden_width ||
        output_len != hidden_len || token_offsets[0] != 0 ||
        token_offsets[token_count] != payload_count ||
        payload_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        token_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        hidden_width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        intermediate_width > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
        return 1;
    }

    for (std::size_t pair = 0; pair < payload_count; ++pair) {
        if (payloads[pair] == nullptr || token_ids[pair] >= token_count ||
            !std::isfinite(weights[pair])) {
            return 1;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) {
            return 1;
        }
    }

    const int pair_count = static_cast<int>(payload_count);
    const int token_count_i = static_cast<int>(token_count);
    const int hidden_i = static_cast<int>(hidden_width);
    const int intermediate_i = static_cast<int>(intermediate_width);
    const int block_size = 128;
    if (token_count > std::numeric_limits<std::size_t>::max() / hidden_width ||
        hidden_width * token_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        payload_count > std::numeric_limits<std::size_t>::max() /
            (2 * intermediate_width * sizeof(float)) ||
        payload_count > std::numeric_limits<std::size_t>::max() /
            (intermediate_width * sizeof(float)) ||
        payload_count > std::numeric_limits<std::size_t>::max() /
            (hidden_width * sizeof(float))) {
        return 1;
    }
    const std::size_t hidden_bytes = hidden_width * token_count * sizeof(float);
    const std::size_t gate_up_bytes = 2 * intermediate_width * payload_count * sizeof(float);
    const std::size_t activated_bytes = intermediate_width * payload_count * sizeof(float);
    const std::size_t pair_output_bytes = hidden_width * payload_count * sizeof(float);

    float *d_hidden = nullptr;
    const uint8_t **d_payloads = nullptr;
    uint32_t *d_token_ids = nullptr;
    float *d_weights = nullptr;
    std::size_t *d_token_offsets = nullptr;
    float *d_gate_up = nullptr;
    float *d_activated = nullptr;
    float *d_pair_output = nullptr;
    float *d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    int result = 0;

    do {
        const auto alloc_start = Clock::now();
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_payloads),
            payload_count * sizeof(*d_payloads)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_ids),
            payload_count * sizeof(*d_token_ids)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_weights),
            payload_count * sizeof(*d_weights)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_offsets),
            token_offsets_len * sizeof(*d_token_offsets)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up), gate_up_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated), activated_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_pair_output), pair_output_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output), hidden_bytes));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->alloc_ms = elapsed_ms(alloc_start, Clock::now());
            result = cuda_status(cudaEventCreate(&start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&end));
            if (result != 0) break;
        }

        const auto h2d_start = Clock::now();
        result = cuda_status(cudaMemcpy(d_hidden, hidden, hidden_bytes, cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_payloads, payloads,
            payload_count * sizeof(*d_payloads), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_ids, token_ids,
            payload_count * sizeof(*d_token_ids), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_weights, weights,
            payload_count * sizeof(*d_weights), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_offsets, token_offsets,
            token_offsets_len * sizeof(*d_token_offsets), cudaMemcpyHostToDevice));
        if (timing != nullptr) {
            timing->h2d_ms = elapsed_ms(h2d_start, Clock::now());
            timing->h2d_bytes = static_cast<uint64_t>(
                hidden_bytes + payload_count * sizeof(*d_payloads) +
                payload_count * sizeof(*d_token_ids) + payload_count * sizeof(*d_weights) +
                token_offsets_len * sizeof(*d_token_offsets));
        }
        if (result != 0) break;

        const auto launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(start));
            if (result != 0) break;
        }
        plain_gate_up_route_kernel<<<
            (2 * intermediate_i * pair_count + block_size - 1) / block_size,
            block_size>>>(dtype, d_payloads, d_token_ids, pair_count, intermediate_i,
                hidden_i, d_hidden, d_gate_up);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        kquant_silu_kernel<<<
            (intermediate_i * pair_count + block_size - 1) / block_size,
            block_size>>>(d_gate_up, pair_count, intermediate_i, d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        plain_down_route_kernel<<<
            (hidden_i * pair_count + block_size - 1) / block_size,
            block_size>>>(dtype, d_payloads, pair_count, hidden_i, intermediate_i,
                d_activated, d_pair_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        aggregate_route_batch_kernel<<<
            (static_cast<int>(token_count) * hidden_i + block_size - 1) / block_size,
            block_size>>>(d_pair_output, d_weights, d_token_offsets, token_count_i,
                hidden_i, d_output);
        result = cuda_status(cudaGetLastError());
        if (timing != nullptr) {
            timing->launch_ms = elapsed_ms(launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(end));
            if (result != 0) break;
        }
        const auto sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, start, end));
            if (result != 0) break;
            timing->kernel_ms = kernel_ms;
            timing->sync_ms = std::max(0.0, elapsed_ms(sync_start, Clock::now()) -
                static_cast<double>(kernel_ms));
        }

        const auto d2h_start = Clock::now();
        result = cuda_status(cudaMemcpy(output, d_output, hidden_bytes, cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_start, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(hidden_bytes);
        }
    } while (false);

    if (start != nullptr) cudaEventDestroy(start);
    if (end != nullptr) cudaEventDestroy(end);
    const auto free_start = Clock::now();
    cudaFree(d_output);
    cudaFree(d_pair_output);
    cudaFree(d_activated);
    cudaFree(d_gate_up);
    cudaFree(d_token_offsets);
    cudaFree(d_weights);
    cudaFree(d_token_ids);
    cudaFree(d_payloads);
    cudaFree(d_hidden);
    if (timing != nullptr) {
        timing->free_ms = elapsed_ms(free_start, Clock::now());
    }
    return result;
}

extern "C" int pom_moe_iquant_forward_route_batch_resident(
    int device_index,
    uint32_t dtype,
    const uint8_t * const *payloads,
    std::size_t payload_count,
    const uint32_t *token_ids,
    const float *weights,
    const std::size_t *token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float *hidden,
    std::size_t hidden_len,
    float *output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    CudaMoeCallTiming *timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(payload_count);
        timing->kernel_launches = 4;
        timing->sync_count = 1;
    }
    if ((dtype != 21 && dtype != 22 && dtype != 23) || payloads == nullptr ||
        token_ids == nullptr || weights == nullptr || token_offsets == nullptr ||
        hidden == nullptr || output == nullptr || payload_count == 0 ||
        token_count == 0 || hidden_width == 0 || intermediate_width == 0 ||
        hidden_width % 256 != 0 || intermediate_width % 256 != 0 ||
        token_offsets_len != token_count + 1 || hidden_len != token_count * hidden_width ||
        output_len != hidden_len || token_offsets[0] != 0 ||
        token_offsets[token_count] != payload_count) {
        return 1;
    }
    for (std::size_t pair = 0; pair < payload_count; ++pair) {
        if (payloads[pair] == nullptr || token_ids[pair] >= token_count ||
            !std::isfinite(weights[pair])) {
            return 1;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) {
            return 1;
        }
    }

    const int pair_count = static_cast<int>(payload_count);
    const int hidden_i = static_cast<int>(hidden_width);
    const int intermediate_i = static_cast<int>(intermediate_width);
    const int block_size = 128;
    const std::size_t hidden_bytes = hidden_width * token_count * sizeof(float);
    const std::size_t gate_up_bytes = 2 * intermediate_width * payload_count * sizeof(float);
    const std::size_t activated_bytes = intermediate_width * payload_count * sizeof(float);
    const std::size_t pair_output_bytes = hidden_width * payload_count * sizeof(float);

    float *d_hidden = nullptr;
    const uint8_t **d_payloads = nullptr;
    uint32_t *d_token_ids = nullptr;
    float *d_weights = nullptr;
    std::size_t *d_token_offsets = nullptr;
    float *d_gate_up = nullptr;
    float *d_activated = nullptr;
    float *d_pair_output = nullptr;
    float *d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    int result = 0;

    do {
        const auto alloc_start = Clock::now();
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_payloads),
            payload_count * sizeof(*d_payloads)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_ids),
            payload_count * sizeof(*d_token_ids)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_weights),
            payload_count * sizeof(*d_weights)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_offsets),
            token_offsets_len * sizeof(*d_token_offsets)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up), gate_up_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated), activated_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_pair_output), pair_output_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output), hidden_bytes));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->alloc_ms = elapsed_ms(alloc_start, Clock::now());
            result = cuda_status(cudaEventCreate(&start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&end));
            if (result != 0) break;
        }

        const auto h2d_start = Clock::now();
        result = cuda_status(cudaMemcpy(d_hidden, hidden, hidden_bytes, cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_payloads, payloads,
            payload_count * sizeof(*d_payloads), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_ids, token_ids,
            payload_count * sizeof(*d_token_ids), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_weights, weights,
            payload_count * sizeof(*d_weights), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_offsets, token_offsets,
            token_offsets_len * sizeof(*d_token_offsets), cudaMemcpyHostToDevice));
        if (timing != nullptr) {
            timing->h2d_ms = elapsed_ms(h2d_start, Clock::now());
            timing->h2d_bytes = static_cast<uint64_t>(
                hidden_bytes + payload_count * sizeof(*d_payloads) +
                payload_count * sizeof(*d_token_ids) + payload_count * sizeof(*d_weights) +
                token_offsets_len * sizeof(*d_token_offsets));
        }
        if (result != 0) break;

        const auto launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(start));
            if (result != 0) break;
        }
        if (dtype == 18) {
            iquant_gate_up_iq3xxs_warp_route_kernel<<<
                intermediate_i * pair_count, 32>>>(
                    d_payloads, d_token_ids, pair_count, intermediate_i,
                    hidden_i, d_hidden, d_gate_up);
        } else {
            iquant_gate_up_cooperative_route_kernel<<<
                2 * intermediate_i * pair_count, block_size>>>(
                    dtype, d_payloads, d_token_ids, pair_count, intermediate_i,
                    hidden_i, d_hidden, d_gate_up);
        }
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        kquant_silu_kernel<<<
            (intermediate_i * pair_count + block_size - 1) / block_size,
            block_size>>>(d_gate_up, pair_count, intermediate_i, d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        if (dtype == 18) {
            iquant_down_iq3xxs_warp_route_kernel<<<
                hidden_i * pair_count, 32>>>(
                    d_payloads, pair_count, hidden_i, intermediate_i,
                    gate_up_bytes, d_activated, d_pair_output);
        } else {
            iquant_down_cooperative_route_kernel<<<
                hidden_i * pair_count, block_size>>>(
                    dtype, d_payloads, pair_count, hidden_i, intermediate_i,
                    gate_up_bytes, d_activated, d_pair_output);
        }
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        aggregate_route_batch_kernel<<<
            (static_cast<int>(token_count) * hidden_i + block_size - 1) / block_size,
            block_size>>>(d_pair_output, d_weights, d_token_offsets,
                static_cast<int>(token_count), hidden_i, d_output);
        result = cuda_status(cudaGetLastError());
        if (timing != nullptr) {
            timing->launch_ms = elapsed_ms(launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(end));
            if (result != 0) break;
        }
        const auto sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, start, end));
            if (result != 0) break;
            timing->kernel_ms = kernel_ms;
            timing->sync_ms = std::max(0.0, elapsed_ms(sync_start, Clock::now()) -
                static_cast<double>(kernel_ms));
        }

        const auto d2h_start = Clock::now();
        result = cuda_status(cudaMemcpy(output, d_output, hidden_bytes, cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_start, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(hidden_bytes);
        }
    } while (false);

    if (start != nullptr) cudaEventDestroy(start);
    if (end != nullptr) cudaEventDestroy(end);
    const auto free_start = Clock::now();
    cudaFree(d_output);
    cudaFree(d_pair_output);
    cudaFree(d_activated);
    cudaFree(d_gate_up);
    cudaFree(d_token_offsets);
    cudaFree(d_weights);
    cudaFree(d_token_ids);
    cudaFree(d_payloads);
    cudaFree(d_hidden);
    if (timing != nullptr) {
        timing->free_ms = elapsed_ms(free_start, Clock::now());
    }
    return result;
}

extern "C" int pom_moe_heterogeneous_quant_forward_route_batch_resident(
    int device_index,
    const CudaHeterogeneousExpert * experts,
    std::size_t expert_count,
    const uint32_t * token_ids,
    const float * weights,
    const std::size_t * token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float * hidden,
    std::size_t hidden_len,
    float * output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    int activation,
    int *nonfinite_stage,
    std::size_t *nonfinite_index,
    std::size_t *nonfinite_count,
    CudaMoeCallTiming * timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (nonfinite_stage != nullptr) {
        *nonfinite_stage = 0;
    }
    if (nonfinite_index != nullptr) {
        *nonfinite_index = 0;
    }
    if (nonfinite_count != nullptr) {
        *nonfinite_count = 0;
    }
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(expert_count);
        timing->kernel_launches = 4;
        timing->sync_count = 1;
    }
    if (experts == nullptr || token_ids == nullptr || weights == nullptr ||
        token_offsets == nullptr || hidden == nullptr || output == nullptr ||
        expert_count == 0 || token_count == 0 || hidden_width == 0 ||
        intermediate_width == 0 || token_offsets_len != token_count + 1 ||
        hidden_len != token_count * hidden_width || output_len != hidden_len ||
        token_offsets[0] != 0 || token_offsets[token_count] != expert_count ||
        (activation != 1 && activation != 2)) {
        return 1;
    }

    const auto valid_dtype = [](uint32_t dtype) {
        return dtype == 2 || dtype == 10 || dtype == 11 || dtype == 12 || dtype == 13 || dtype == 14 || dtype == 18 || dtype == 21 ||
            dtype == 22 || dtype == 23 || dtype == 39;
    };
    const auto row_bytes = [&](uint32_t dtype, std::size_t columns) -> std::size_t {
        if (dtype == 2) {
            return (columns / 32) * 18;
        }
        if (dtype == 39) {
            return (columns / 32) * 17;
        }
        const std::size_t block_bytes = dtype == 10 ? 84 : dtype == 11 ? 110 :
            dtype == 12 ? 144 : dtype == 13 ? 176 : dtype == 14 ? 210 : dtype == 18 ? 98 :
            dtype == 21 ? 110 : dtype == 22 ? 82 : 136;
        return (columns / 256) * block_bytes;
    };
    const auto valid_tensor = [&](const CudaHeterogeneousTensor & tensor,
                                  std::size_t rows,
                                  std::size_t columns) {
        if (tensor.data == nullptr || !valid_dtype(tensor.dtype) ||
            tensor.rows != rows || tensor.columns != columns ||
            columns == 0 ||
            ((tensor.dtype == 2 || tensor.dtype == 39) ? columns % 32 != 0 : columns % 256 != 0) ||
            !std::isfinite(tensor.scale) || !std::isfinite(tensor.input_scale) ||
            !std::isfinite(tensor.secondary_scale) ||
            !std::isfinite(tensor.secondary_input_scale)) {
            return false;
        }
        return tensor.bytes == rows * row_bytes(tensor.dtype, columns);
    };
    for (std::size_t pair = 0; pair < expert_count; ++pair) {
        const auto & expert = experts[pair];
        if (token_ids[pair] >= token_count || !std::isfinite(weights[pair])) {
            return 1;
        }
        if (!valid_tensor(expert.down, hidden_width, intermediate_width)) {
            return 1;
        }
        const bool fused = expert.gate_up.data != nullptr;
        if (fused) {
            if (expert.gate.data != nullptr || expert.up.data != nullptr ||
                !valid_tensor(expert.gate_up, 2 * intermediate_width, hidden_width)) {
                return 1;
            }
        } else if (!valid_tensor(expert.gate, intermediate_width, hidden_width) ||
                   !valid_tensor(expert.up, intermediate_width, hidden_width)) {
            return 1;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) {
            return 1;
        }
    }

    const int pair_count = static_cast<int>(expert_count);
    const int hidden_i = static_cast<int>(hidden_width);
    const int intermediate_i = static_cast<int>(intermediate_width);
    const int block_size = 128;
    const std::size_t hidden_bytes = hidden_len * sizeof(float);
    const std::size_t gate_up_bytes = 2 * intermediate_width * expert_count * sizeof(float);
    const std::size_t activated_bytes = intermediate_width * expert_count * sizeof(float);
    const std::size_t pair_output_bytes = hidden_width * expert_count * sizeof(float);

    CudaHeterogeneousExpert * d_experts = nullptr;
    float * d_hidden = nullptr;
    uint32_t * d_token_ids = nullptr;
    float * d_weights = nullptr;
    std::size_t * d_token_offsets = nullptr;
    float * d_gate_up = nullptr;
    float * d_activated = nullptr;
    float * d_pair_output = nullptr;
    float * d_output = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    int result = 0;

    do {
        const auto alloc_start = Clock::now();
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_experts),
            expert_count * sizeof(*d_experts)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_ids),
            expert_count * sizeof(*d_token_ids)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_weights),
            expert_count * sizeof(*d_weights)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_offsets),
            token_offsets_len * sizeof(*d_token_offsets)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up), gate_up_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated), activated_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_pair_output), pair_output_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output), hidden_bytes));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->alloc_ms = elapsed_ms(alloc_start, Clock::now());
            result = cuda_status(cudaEventCreate(&start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&end));
            if (result != 0) break;
        }

        const auto h2d_start = Clock::now();
        result = cuda_status(cudaMemcpy(d_experts, experts,
            expert_count * sizeof(*d_experts), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_hidden, hidden, hidden_bytes, cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_ids, token_ids,
            expert_count * sizeof(*d_token_ids), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_weights, weights,
            expert_count * sizeof(*d_weights), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_offsets, token_offsets,
            token_offsets_len * sizeof(*d_token_offsets), cudaMemcpyHostToDevice));
        if (timing != nullptr) {
            timing->h2d_ms = elapsed_ms(h2d_start, Clock::now());
            timing->h2d_bytes = static_cast<uint64_t>(
                expert_count * sizeof(*d_experts) + hidden_bytes +
                expert_count * sizeof(*d_token_ids) + expert_count * sizeof(*d_weights) +
                token_offsets_len * sizeof(*d_token_offsets));
        }
        if (result != 0) break;

        const auto launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(start));
            if (result != 0) break;
        }
        heterogeneous_quant_gate_up_route_kernel<<<
            2 * intermediate_i * pair_count,
            block_size>>>(d_experts, d_token_ids, pair_count, intermediate_i,
                hidden_i, d_hidden, d_gate_up);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        heterogeneous_activation_kernel<<<
            (intermediate_i * pair_count + block_size - 1) / block_size,
            block_size>>>(d_gate_up, pair_count, intermediate_i, activation, d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        heterogeneous_quant_down_route_kernel<<<
            hidden_i * pair_count,
            block_size>>>(d_experts, pair_count, hidden_i, intermediate_i,
                d_activated, d_pair_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        aggregate_route_batch_kernel<<<
            (static_cast<int>(token_count) * hidden_i + block_size - 1) / block_size,
            block_size>>>(d_pair_output, d_weights, d_token_offsets,
                static_cast<int>(token_count), hidden_i, d_output);
        result = cuda_status(cudaGetLastError());
        if (timing != nullptr) {
            timing->launch_ms = elapsed_ms(launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(end));
            if (result != 0) break;
        }
        const auto sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, start, end));
            if (result != 0) break;
            timing->kernel_ms = kernel_ms;
            timing->sync_ms = std::max(0.0, elapsed_ms(sync_start, Clock::now()) -
                static_cast<double>(kernel_ms));
        }

        const auto d2h_start = Clock::now();
        result = cuda_status(cudaMemcpy(output, d_output, hidden_bytes, cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_start, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(hidden_bytes);
        }
        if (result == 0 && first_nonfinite(
                output,
                hidden_bytes / sizeof(float),
                nonfinite_index,
                nonfinite_count)) {
            std::vector<float> stage_values;
            int copy_status = 0;
            if (device_has_nonfinite(
                    d_gate_up,
                    gate_up_bytes / sizeof(float),
                    stage_values,
                    &copy_status,
                    nullptr,
                    nullptr)) {
                if (nonfinite_stage != nullptr) *nonfinite_stage = 1;
                result = 0;
                break;
            }
            if (copy_status != 0) {
                result = copy_status;
                break;
            }
            if (device_has_nonfinite(
                    d_activated,
                    activated_bytes / sizeof(float),
                    stage_values,
                    &copy_status,
                    nullptr,
                    nullptr)) {
                if (nonfinite_stage != nullptr) *nonfinite_stage = 2;
                result = 0;
                break;
            }
            if (copy_status != 0) {
                result = copy_status;
                break;
            }
            if (device_has_nonfinite(
                    d_pair_output,
                    pair_output_bytes / sizeof(float),
                    stage_values,
                    &copy_status,
                    nullptr,
                    nullptr)) {
                if (nonfinite_stage != nullptr) *nonfinite_stage = 3;
                result = 0;
                break;
            }
            if (copy_status != 0) {
                result = copy_status;
                break;
            }
            if (nonfinite_stage != nullptr) *nonfinite_stage = 4;
        }
    } while (false);

    if (start != nullptr) cudaEventDestroy(start);
    if (end != nullptr) cudaEventDestroy(end);
    const auto free_start = Clock::now();
    cudaFree(d_output);
    cudaFree(d_pair_output);
    cudaFree(d_activated);
    cudaFree(d_gate_up);
    cudaFree(d_token_offsets);
    cudaFree(d_weights);
    cudaFree(d_token_ids);
    cudaFree(d_hidden);
    cudaFree(d_experts);
    if (timing != nullptr) {
        timing->free_ms = elapsed_ms(free_start, Clock::now());
    }
    return result;
}

// O callback MoE também é síncrono para o llama.cpp, mas as alocações de
// trabalho não precisam ser. Este pipeline mantém os buffers temporários por
// device e só cresce quando um lote maior aparece. Os descritores continuam
// sendo copiados a cada rota porque os experts residentes podem mudar, porém
// o peso apontado por eles nunca volta para a CPU.
struct HeterogeneousRoutePipeline {
    int device_index;
    std::size_t hidden_width;
    std::size_t intermediate_width;
    std::size_t token_capacity = 0;
    std::size_t route_capacity = 0;
    CudaHeterogeneousExpert *d_experts = nullptr;
    float *d_hidden = nullptr;
    block_q8_1 *d_hidden_q8 = nullptr;
    uint32_t *d_token_ids = nullptr;
    float *d_weights = nullptr;
    uint32_t *d_route_to_pair = nullptr;
    std::size_t *d_token_offsets = nullptr;
    float *d_gate_up = nullptr;
    float *d_activated = nullptr;
    block_q8_1 *d_activated_q8 = nullptr;
    float *d_pair_output = nullptr;
    float *d_output = nullptr;
    // The llama.cpp callback owns a regular host tensor. Copying directly to
    // pageable memory makes every synchronous D2H transfer pay the CUDA
    // staging cost. Keep one pinned staging buffer per device pipeline and
    // copy the small final result to the callback buffer after the transfer.
    float *h_output = nullptr;
};

void release_heterogeneous_route_pipeline(HeterogeneousRoutePipeline *pipeline) {
    if (pipeline == nullptr) return;
    cudaFreeHost(pipeline->h_output);
    cudaFree(pipeline->d_output);
    cudaFree(pipeline->d_pair_output);
    cudaFree(pipeline->d_activated);
    cudaFree(pipeline->d_activated_q8);
    cudaFree(pipeline->d_gate_up);
    cudaFree(pipeline->d_token_offsets);
    cudaFree(pipeline->d_route_to_pair);
    cudaFree(pipeline->d_weights);
    cudaFree(pipeline->d_token_ids);
    cudaFree(pipeline->d_hidden);
    cudaFree(pipeline->d_hidden_q8);
    cudaFree(pipeline->d_experts);
    pipeline->d_output = nullptr;
    pipeline->h_output = nullptr;
    pipeline->d_pair_output = nullptr;
    pipeline->d_activated = nullptr;
    pipeline->d_activated_q8 = nullptr;
    pipeline->d_gate_up = nullptr;
    pipeline->d_token_offsets = nullptr;
    pipeline->d_route_to_pair = nullptr;
    pipeline->d_weights = nullptr;
    pipeline->d_token_ids = nullptr;
    pipeline->d_hidden = nullptr;
    pipeline->d_hidden_q8 = nullptr;
    pipeline->d_experts = nullptr;
    pipeline->token_capacity = 0;
    pipeline->route_capacity = 0;
}

int ensure_heterogeneous_route_pipeline_capacity(
    HeterogeneousRoutePipeline *pipeline,
    std::size_t tokens,
    std::size_t routes
) {
    if (pipeline->token_capacity >= tokens && pipeline->route_capacity >= routes) {
        return 0;
    }
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    if (tokens == 0 || routes == 0 ||
        pipeline->hidden_width % QK8_1 != 0 ||
        pipeline->intermediate_width % QK8_1 != 0 ||
        tokens > max_size / pipeline->hidden_width ||
        routes > max_size / sizeof(CudaHeterogeneousExpert) ||
        routes > max_size / sizeof(uint32_t) ||
        routes > max_size / sizeof(float) ||
        routes > max_size / pipeline->intermediate_width ||
        routes > max_size / pipeline->hidden_width ||
        pipeline->intermediate_width > max_size / 2 ||
        tokens + 1 < tokens) {
        return 1;
    }
    const std::size_t hidden_elements = tokens * pipeline->hidden_width;
    const std::size_t hidden_q8_blocks = hidden_elements / QK8_1;
    const std::size_t gate_up_elements = 2 * pipeline->intermediate_width * routes;
    const std::size_t activated_elements = pipeline->intermediate_width * routes;
    const std::size_t activated_q8_blocks = activated_elements / QK8_1;
    const std::size_t pair_output_elements = pipeline->hidden_width * routes;
    if (hidden_elements > max_size / sizeof(float) ||
        gate_up_elements > max_size / sizeof(float) ||
        hidden_q8_blocks > max_size / sizeof(block_q8_1) ||
        activated_elements > max_size / sizeof(float) ||
        activated_q8_blocks > max_size / sizeof(block_q8_1) ||
        pair_output_elements > max_size / sizeof(float) ||
        tokens + 1 > max_size / sizeof(std::size_t)) {
        return 1;
    }

    release_heterogeneous_route_pipeline(pipeline);
    pipeline->token_capacity = tokens;
    pipeline->route_capacity = routes;
    const std::size_t hidden_bytes = hidden_elements * sizeof(float);
    const std::size_t gate_up_bytes = gate_up_elements * sizeof(float);
    const std::size_t activated_bytes = activated_elements * sizeof(float);
    const std::size_t pair_output_bytes = pair_output_elements * sizeof(float);
    const std::size_t offset_bytes = (tokens + 1) * sizeof(std::size_t);
    int result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_experts),
        routes * sizeof(*pipeline->d_experts)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_hidden), hidden_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_hidden_q8),
        hidden_q8_blocks * sizeof(*pipeline->d_hidden_q8)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_token_ids), routes * sizeof(*pipeline->d_token_ids)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_weights), routes * sizeof(*pipeline->d_weights)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_route_to_pair),
        routes * sizeof(*pipeline->d_route_to_pair)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_token_offsets), offset_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_gate_up), gate_up_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_activated), activated_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_activated_q8),
        activated_q8_blocks * sizeof(*pipeline->d_activated_q8)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_pair_output), pair_output_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_output), hidden_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMallocHost(
        reinterpret_cast<void **>(&pipeline->h_output), hidden_bytes));
    if (result != 0) goto fail;
    return 0;

fail:
    release_heterogeneous_route_pipeline(pipeline);
    return result;
}

extern "C" int pom_moe_heterogeneous_route_pipeline_create(
    int device_index,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    void **out
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (out == nullptr || hidden_width == 0 || intermediate_width == 0 ||
        hidden_width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        intermediate_width > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return 1;
    }
    *out = nullptr;
    auto *pipeline = new (std::nothrow) HeterogeneousRoutePipeline{
        device_index,
        hidden_width,
        intermediate_width,
    };
    if (pipeline == nullptr) return 2;
    *out = pipeline;
    return 0;
}

extern "C" int pom_moe_heterogeneous_route_pipeline_run(
    void *handle,
    const CudaHeterogeneousExpert * experts,
    std::size_t expert_count,
    const uint32_t * token_ids,
    const float * weights,
    const uint32_t * route_to_pair,
    std::size_t route_count,
    const std::size_t * token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float * hidden,
    std::size_t hidden_len,
    float * output,
    std::size_t output_len,
    int activation,
    int *nonfinite_stage,
    std::size_t *nonfinite_index,
    std::size_t *nonfinite_count,
    CudaMoeCallTiming * timing
) {
    auto *pipeline = static_cast<HeterogeneousRoutePipeline *>(handle);
    if (pipeline == nullptr) return 1;
    const ScopedMoeCudaDevice device(pipeline->device_index);
    if (!device.ready()) return 1;
    if (nonfinite_stage != nullptr) *nonfinite_stage = 0;
    if (nonfinite_index != nullptr) *nonfinite_index = 0;
    if (nonfinite_count != nullptr) *nonfinite_count = 0;
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(expert_count);
        timing->kernel_launches = 6;
        // cudaMemcpy(DeviceToHost) below is the completion barrier. An extra
        // cudaDeviceSynchronize would wait for unrelated work on this device.
        timing->sync_count = 0;
    }
    const auto validation_started = Clock::now();
    if (experts == nullptr || token_ids == nullptr || weights == nullptr ||
        route_to_pair == nullptr ||
        token_offsets == nullptr || hidden == nullptr || output == nullptr ||
        expert_count == 0 || route_count == 0 || token_count == 0 ||
        expert_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        route_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        token_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        token_offsets_len != token_count + 1 || hidden_len != token_count * pipeline->hidden_width ||
        output_len != hidden_len || token_offsets[0] != 0 ||
        token_offsets[token_count] != route_count ||
        token_count + 1 < token_count) {
        return 1;
    }
    const auto valid_dtype = [](uint32_t dtype) {
        return dtype == 2 || dtype == 10 || dtype == 11 || dtype == 12 || dtype == 13 ||
            dtype == 14 || dtype == 18 || dtype == 21 || dtype == 22 || dtype == 23 || dtype == 39;
    };
    const auto row_bytes = [&](uint32_t dtype, std::size_t columns) -> std::size_t {
        if (dtype == 2) return (columns / 32) * 18;
        if (dtype == 39) return (columns / 32) * 17;
        const std::size_t block_bytes = dtype == 10 ? 84 : dtype == 11 ? 110 :
            dtype == 12 ? 144 : dtype == 13 ? 176 : dtype == 14 ? 210 : dtype == 18 ? 98 :
            dtype == 21 ? 110 : dtype == 22 ? 82 : 136;
        return (columns / 256) * block_bytes;
    };
    const auto valid_tensor = [&](const CudaHeterogeneousTensor & tensor,
                                  std::size_t rows,
                                  std::size_t columns) {
        if (tensor.data == nullptr || !valid_dtype(tensor.dtype) || tensor.rows != rows ||
            tensor.columns != columns || columns == 0 ||
            ((tensor.dtype == 2 || tensor.dtype == 39) ? columns % 32 != 0 : columns % 256 != 0) ||
            !std::isfinite(tensor.scale) || !std::isfinite(tensor.input_scale) ||
            !std::isfinite(tensor.secondary_scale) ||
            !std::isfinite(tensor.secondary_input_scale)) {
            return false;
        }
        return tensor.bytes == rows * row_bytes(tensor.dtype, columns);
    };
    for (std::size_t pair = 0; pair < expert_count; ++pair) {
        const auto & expert = experts[pair];
        if (token_ids[pair] >= token_count || !std::isfinite(weights[pair]) ||
            !valid_tensor(expert.down, pipeline->hidden_width, pipeline->intermediate_width)) {
            return 1;
        }
        const bool fused = expert.gate_up.data != nullptr;
        if (fused) {
            if (expert.gate.data != nullptr || expert.up.data != nullptr ||
                !valid_tensor(expert.gate_up, 2 * pipeline->intermediate_width,
                    pipeline->hidden_width)) {
                return 1;
            }
        } else if (!valid_tensor(expert.gate, pipeline->intermediate_width, pipeline->hidden_width) ||
                   !valid_tensor(expert.up, pipeline->intermediate_width, pipeline->hidden_width)) {
            return 1;
        }
    }
    for (std::size_t route = 0; route < route_count; ++route) {
        if (route_to_pair[route] >= expert_count) return 1;
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) return 1;
    }
    if (timing != nullptr) {
        timing->payload_validation_ms = elapsed_ms(validation_started, Clock::now());
    }

    const auto allocation_started = Clock::now();
    const std::size_t old_token_capacity = pipeline->token_capacity;
    const std::size_t old_route_capacity = pipeline->route_capacity;
    const int capacity_result = ensure_heterogeneous_route_pipeline_capacity(
        pipeline, token_count, std::max(expert_count, route_count));
    if (timing != nullptr &&
        (old_token_capacity != pipeline->token_capacity || old_route_capacity != pipeline->route_capacity)) {
        timing->alloc_ms = elapsed_ms(allocation_started, Clock::now());
    }
    if (capacity_result != 0) return capacity_result;

    const int pair_count = static_cast<int>(expert_count);
    const int token_count_i = static_cast<int>(token_count);
    const int hidden_i = static_cast<int>(pipeline->hidden_width);
    const int intermediate_i = static_cast<int>(pipeline->intermediate_width);
    const int block_size = 128;
    const int warp_size = 32;
    const std::size_t hidden_bytes = hidden_len * sizeof(float);
    const std::size_t gate_up_bytes = 2 * pipeline->intermediate_width * expert_count * sizeof(float);
    const std::size_t activated_bytes = pipeline->intermediate_width * expert_count * sizeof(float);
    const std::size_t pair_output_bytes = pipeline->hidden_width * expert_count * sizeof(float);

    const auto h2d_started = Clock::now();
    int result = cuda_status(cudaMemcpy(
        pipeline->d_experts, experts,
        expert_count * sizeof(*pipeline->d_experts), cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_hidden, hidden, hidden_bytes, cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_token_ids, token_ids,
        expert_count * sizeof(*pipeline->d_token_ids), cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_weights, weights,
        route_count * sizeof(*pipeline->d_weights), cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_route_to_pair, route_to_pair,
        route_count * sizeof(*pipeline->d_route_to_pair), cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_token_offsets, token_offsets,
        token_offsets_len * sizeof(*pipeline->d_token_offsets), cudaMemcpyHostToDevice));
    if (timing != nullptr) {
        timing->h2d_ms = elapsed_ms(h2d_started, Clock::now());
        timing->h2d_bytes = static_cast<uint64_t>(
            expert_count * sizeof(*pipeline->d_experts) + hidden_bytes +
            expert_count * sizeof(*pipeline->d_token_ids) +
            route_count * sizeof(*pipeline->d_weights) +
            route_count * sizeof(*pipeline->d_route_to_pair) +
            token_offsets_len * sizeof(*pipeline->d_token_offsets));
    }
    if (result != 0) return result;

    const auto launch_started = Clock::now();
    quantize_q8_1_rows_kernel<<<
        token_count_i * (hidden_i / QK8_1), warp_size>>>(
            pipeline->d_hidden, token_count_i, hidden_i, pipeline->d_hidden_q8);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    heterogeneous_quant_gate_up_route_q8_kernel<<<
        intermediate_i * pair_count, warp_size>>>(
            pipeline->d_experts, pipeline->d_token_ids, pair_count, intermediate_i,
            hidden_i, pipeline->d_hidden_q8, pipeline->d_gate_up);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    heterogeneous_activation_kernel<<<
        (intermediate_i * pair_count + block_size - 1) / block_size,
        block_size>>>(pipeline->d_gate_up, pair_count, intermediate_i, activation,
            pipeline->d_activated);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    quantize_q8_1_rows_kernel<<<
        pair_count * (intermediate_i / QK8_1), warp_size>>>(
            pipeline->d_activated, pair_count, intermediate_i, pipeline->d_activated_q8);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    heterogeneous_quant_down_route_q8_kernel<<<
        hidden_i * pair_count, warp_size>>>(
            pipeline->d_experts, pair_count, hidden_i, intermediate_i,
            pipeline->d_activated_q8, pipeline->d_pair_output);
    result = cuda_status(cudaGetLastError());
    if (result != 0) return result;
    aggregate_route_batch_deduplicated_kernel<<<
        (token_count_i * hidden_i + block_size - 1) / block_size,
        block_size>>>(pipeline->d_pair_output, pipeline->d_weights,
            pipeline->d_route_to_pair, pipeline->d_token_offsets,
            token_count_i, hidden_i, pipeline->d_output);
    result = cuda_status(cudaGetLastError());
    if (timing != nullptr) timing->launch_ms = elapsed_ms(launch_started, Clock::now());
    if (result != 0) return result;
    const auto d2h_started = Clock::now();
    result = cuda_status(cudaMemcpy(
        pipeline->h_output, pipeline->d_output, hidden_bytes, cudaMemcpyDeviceToHost));
    if (timing != nullptr) {
        timing->d2h_ms = elapsed_ms(d2h_started, Clock::now());
        timing->d2h_bytes = static_cast<uint64_t>(hidden_bytes);
    }
    if (result != 0) return result;
    std::memcpy(output, pipeline->h_output, hidden_bytes);
    if (timing != nullptr) {
        // A chamada pode ocorrer durante a captura de grafo do llama.cpp.
        // Eventos gravados pelo hook retornam cudaErrorInvalidResourceHandle
        // (400) nesse contexto. A cópia D2H é a barreira de conclusão, então
        // o intervalo de parede até ela mantém o diagnóstico sem tocar nos
        // recursos da captura.
        timing->kernel_ms = elapsed_ms(launch_started, d2h_started);
        timing->kernel_span_ms = timing->kernel_ms;
        timing->transfer_span_ms = timing->h2d_ms + timing->d2h_ms;
    }
    if (first_nonfinite(
            output, hidden_bytes / sizeof(float), nonfinite_index, nonfinite_count)) {
        std::vector<float> stage_values;
        int copy_status = 0;
        if (device_has_nonfinite(
                pipeline->d_gate_up, gate_up_bytes / sizeof(float), stage_values,
                &copy_status, nullptr, nullptr)) {
            if (nonfinite_stage != nullptr) *nonfinite_stage = 1;
        } else if (copy_status != 0) {
            result = copy_status;
        } else if (device_has_nonfinite(
                pipeline->d_activated, activated_bytes / sizeof(float), stage_values,
                &copy_status, nullptr, nullptr)) {
            if (nonfinite_stage != nullptr) *nonfinite_stage = 2;
        } else if (copy_status != 0) {
            result = copy_status;
        } else if (device_has_nonfinite(
                pipeline->d_pair_output, pair_output_bytes / sizeof(float), stage_values,
                &copy_status, nullptr, nullptr)) {
            if (nonfinite_stage != nullptr) *nonfinite_stage = 3;
        } else if (copy_status != 0) {
            result = copy_status;
        } else if (nonfinite_stage != nullptr) {
            *nonfinite_stage = 4;
        }
    }
    return result;
}

extern "C" void pom_moe_heterogeneous_route_pipeline_destroy(void *handle) {
    auto *pipeline = static_cast<HeterogeneousRoutePipeline *>(handle);
    if (pipeline == nullptr) return;
    const ScopedMoeCudaDevice device(pipeline->device_index);
    if (!device.ready()) return;
    release_heterogeneous_route_pipeline(pipeline);
    delete pipeline;
}

extern "C" int pom_moe_nvfp4_forward_batch(
    int device_index,
    const uint8_t * payload,
    std::size_t payload_len,
    std::size_t payload_count,
    const float * hidden,
    std::size_t hidden_len,
    float * output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    CudaMoeCallTiming * timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    const uint64_t expert_calls = timing != nullptr ? static_cast<uint64_t>(payload_count) : 0;
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = expert_calls;
        timing->kernel_launches = 5;
        timing->sync_count = 2;
    }
    if (payload == nullptr || hidden == nullptr || output == nullptr || payload_count == 0 ||
        hidden_width == 0 || intermediate_width == 0 ||
        hidden_width % kBlockElements != 0 || intermediate_width % kBlockElements != 0 ||
        hidden_len != hidden_width || output_len != hidden_width * payload_count ||
        payload_len < kPayloadHeaderBytes) {
        return 1;
    }

    const std::size_t gate_up_expected = 2 * intermediate_width * (hidden_width / kBlockElements) * kBlockBytes;
    const std::size_t down_expected = hidden_width * (intermediate_width / kBlockElements) * kBlockBytes;
    for (std::size_t expert = 0; expert < payload_count; ++expert) {
        const auto * header = payload + expert * payload_len;
        const std::size_t gate_up_len = static_cast<std::size_t>(header[0]) |
            (static_cast<std::size_t>(header[1]) << 8) |
            (static_cast<std::size_t>(header[2]) << 16) |
            (static_cast<std::size_t>(header[3]) << 24);
        const std::size_t down_len = static_cast<std::size_t>(header[4]) |
            (static_cast<std::size_t>(header[5]) << 8) |
            (static_cast<std::size_t>(header[6]) << 16) |
            (static_cast<std::size_t>(header[7]) << 24);
        if (gate_up_len != gate_up_expected || down_len != down_expected ||
            gate_up_len > payload_len - kPayloadHeaderBytes ||
            down_len != payload_len - kPayloadHeaderBytes - gate_up_len) {
            return 1;
        }
    }
    float * d_hidden = nullptr;
    float * d_hidden_scales = nullptr;
    int8_t * d_hidden_qs = nullptr;
    uint8_t * d_payloads = nullptr;
    float * d_gate_up_values = nullptr;
    float * d_activated = nullptr;
    float * d_activated_scales = nullptr;
    int8_t * d_activated_qs = nullptr;
    float * d_output = nullptr;
    cudaEvent_t gate_start = nullptr;
    cudaEvent_t gate_end = nullptr;
    cudaEvent_t down_start = nullptr;
    cudaEvent_t down_end = nullptr;
    int result = 0;

    do {
        const auto alloc_start = Clock::now();
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_width * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden_scales), hidden_width / 32 * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden_qs), hidden_width * sizeof(int8_t)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_payloads), payload_len * payload_count));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up_values),
            2 * intermediate_width * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated),
            intermediate_width * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated_scales),
            intermediate_width / 32 * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated_qs),
            intermediate_width * payload_count * sizeof(int8_t)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output),
            hidden_width * payload_count * sizeof(float)));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->alloc_ms = elapsed_ms(alloc_start, Clock::now());
        }

        if (timing != nullptr) {
            result = cuda_status(cudaEventCreate(&gate_start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&gate_end));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&down_start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&down_end));
            if (result != 0) break;
        }

        const auto h2d_start = Clock::now();
        result = cuda_status(cudaMemcpy(d_hidden, hidden, hidden_width * sizeof(float), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_payloads, payload,
            payload_len * payload_count, cudaMemcpyHostToDevice));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->h2d_ms = elapsed_ms(h2d_start, Clock::now());
            timing->h2d_bytes = static_cast<uint64_t>(hidden_width * sizeof(float) +
                payload_len * payload_count);
        }

        const int block_size = 128;
        const auto gate_launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(gate_start));
            if (result != 0) break;
        }
        quantize_q8_0_kernel<<<(hidden_width + 31) / 32, 32>>>(
            d_hidden, static_cast<int>(hidden_width), d_hidden_scales, d_hidden_qs, 1);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        gate_up_kernel<<<(2 * intermediate_width * payload_count + block_size - 1) / block_size, block_size>>>(
            d_payloads, payload_len, static_cast<int>(payload_count),
            static_cast<int>(intermediate_width), static_cast<int>(hidden_width),
            d_hidden_qs, d_hidden_scales, d_gate_up_values);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        if (timing != nullptr) {
            timing->launch_ms += elapsed_ms(gate_launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(gate_end));
            if (result != 0) break;
        }
        const auto gate_sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, gate_start, gate_end));
            if (result != 0) break;
            timing->kernel_ms += kernel_ms;
            timing->sync_ms += std::max(0.0, elapsed_ms(gate_sync_start, Clock::now()) - static_cast<double>(kernel_ms));
        }

        const auto down_launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(down_start));
            if (result != 0) break;
        }
        geglu_kernel<<<(intermediate_width * payload_count + block_size - 1) / block_size, block_size>>>(
            d_gate_up_values, static_cast<int>(payload_count),
            static_cast<int>(intermediate_width), d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        quantize_q8_0_kernel<<<((intermediate_width + 31) / 32) * payload_count, 32>>>(
            d_activated, static_cast<int>(intermediate_width), d_activated_scales,
            d_activated_qs, static_cast<int>(payload_count));
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        down_kernel<<<(hidden_width * payload_count + block_size - 1) / block_size, block_size>>>(
            d_payloads, payload_len, static_cast<int>(payload_count),
            static_cast<int>(hidden_width), static_cast<int>(intermediate_width),
            d_activated_qs, d_activated_scales, d_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        if (timing != nullptr) {
            timing->launch_ms += elapsed_ms(down_launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(down_end));
            if (result != 0) break;
        }
        const auto down_sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, down_start, down_end));
            if (result != 0) break;
            timing->kernel_ms += kernel_ms;
            timing->sync_ms += std::max(0.0, elapsed_ms(down_sync_start, Clock::now()) - static_cast<double>(kernel_ms));
        }
        const auto d2h_start = Clock::now();
        result = cuda_status(cudaMemcpy(output, d_output,
            hidden_width * payload_count * sizeof(float), cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_start, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(hidden_width * payload_count * sizeof(float));
        }
    } while (false);

    if (gate_start != nullptr) cudaEventDestroy(gate_start);
    if (gate_end != nullptr) cudaEventDestroy(gate_end);
    if (down_start != nullptr) cudaEventDestroy(down_start);
    if (down_end != nullptr) cudaEventDestroy(down_end);
    const auto free_start = Clock::now();
    cudaFree(d_output);
    cudaFree(d_activated_qs);
    cudaFree(d_activated_scales);
    cudaFree(d_activated);
    cudaFree(d_gate_up_values);
    cudaFree(d_payloads);
    cudaFree(d_hidden_qs);
    cudaFree(d_hidden_scales);
    cudaFree(d_hidden);
    if (timing != nullptr) {
        timing->free_ms = elapsed_ms(free_start, Clock::now());
    }
    return result;
}

extern "C" int pom_moe_nvfp4_forward_batch_resident(
    int device_index,
    const uint8_t * const * payloads,
    std::size_t payload_count,
    const float * hidden,
    std::size_t hidden_len,
    float * output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    CudaMoeCallTiming * timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(payload_count);
        timing->kernel_launches = 5;
        timing->sync_count = 2;
    }
    if (payloads == nullptr || hidden == nullptr || output == nullptr || payload_count == 0 ||
        hidden_width == 0 || intermediate_width == 0 ||
        hidden_width % kBlockElements != 0 || intermediate_width % kBlockElements != 0 ||
        hidden_len != hidden_width || output_len != hidden_width * payload_count) {
        return 1;
    }
    for (std::size_t expert = 0; expert < payload_count; ++expert) {
        if (payloads[expert] == nullptr) {
            return 1;
        }
    }

    float * d_hidden = nullptr;
    float * d_hidden_scales = nullptr;
    int8_t * d_hidden_qs = nullptr;
    const uint8_t ** d_payloads = nullptr;
    float * d_gate_up_values = nullptr;
    float * d_activated = nullptr;
    float * d_activated_scales = nullptr;
    int8_t * d_activated_qs = nullptr;
    float * d_output = nullptr;
    cudaEvent_t gate_start = nullptr;
    cudaEvent_t gate_end = nullptr;
    cudaEvent_t down_start = nullptr;
    cudaEvent_t down_end = nullptr;
    int result = 0;

    do {
        const auto alloc_start = Clock::now();
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_width * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden_scales), hidden_width / 32 * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden_qs), hidden_width * sizeof(int8_t)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_payloads), payload_count * sizeof(*d_payloads)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up_values),
            2 * intermediate_width * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated),
            intermediate_width * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated_scales),
            intermediate_width / 32 * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated_qs),
            intermediate_width * payload_count * sizeof(int8_t)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output),
            hidden_width * payload_count * sizeof(float)));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->alloc_ms = elapsed_ms(alloc_start, Clock::now());
            result = cuda_status(cudaEventCreate(&gate_start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&gate_end));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&down_start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&down_end));
            if (result != 0) break;
        }

        const auto h2d_start = Clock::now();
        result = cuda_status(cudaMemcpy(d_hidden, hidden, hidden_width * sizeof(float), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_payloads, payloads,
            payload_count * sizeof(*d_payloads), cudaMemcpyHostToDevice));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->h2d_ms = elapsed_ms(h2d_start, Clock::now());
            timing->h2d_bytes = static_cast<uint64_t>(hidden_width * sizeof(float) +
                payload_count * sizeof(*d_payloads));
        }

        const int block_size = 128;
        const auto gate_launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(gate_start));
            if (result != 0) break;
        }
        quantize_q8_0_kernel<<<(hidden_width + 31) / 32, 32>>>(
            d_hidden, static_cast<int>(hidden_width), d_hidden_scales, d_hidden_qs, 1);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        gate_up_resident_kernel<<<(2 * intermediate_width * payload_count + block_size - 1) / block_size, block_size>>>(
            d_payloads, static_cast<int>(payload_count),
            static_cast<int>(intermediate_width), static_cast<int>(hidden_width),
            d_hidden_qs, d_hidden_scales, d_gate_up_values);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        if (timing != nullptr) {
            timing->launch_ms += elapsed_ms(gate_launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(gate_end));
            if (result != 0) break;
        }
        const auto gate_sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, gate_start, gate_end));
            if (result != 0) break;
            timing->kernel_ms += kernel_ms;
            timing->sync_ms += std::max(0.0, elapsed_ms(gate_sync_start, Clock::now()) - static_cast<double>(kernel_ms));
        }

        const auto down_launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(down_start));
            if (result != 0) break;
        }
        geglu_kernel<<<(intermediate_width * payload_count + block_size - 1) / block_size, block_size>>>(
            d_gate_up_values, static_cast<int>(payload_count),
            static_cast<int>(intermediate_width), d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        quantize_q8_0_kernel<<<((intermediate_width + 31) / 32) * payload_count, 32>>>(
            d_activated, static_cast<int>(intermediate_width), d_activated_scales,
            d_activated_qs, static_cast<int>(payload_count));
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        down_resident_kernel<<<(hidden_width * payload_count + block_size - 1) / block_size, block_size>>>(
            d_payloads, static_cast<int>(payload_count),
            static_cast<int>(hidden_width), static_cast<int>(intermediate_width),
            d_activated_qs, d_activated_scales, d_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        if (timing != nullptr) {
            timing->launch_ms += elapsed_ms(down_launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(down_end));
            if (result != 0) break;
        }
        const auto down_sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, down_start, down_end));
            if (result != 0) break;
            timing->kernel_ms += kernel_ms;
            timing->sync_ms += std::max(0.0, elapsed_ms(down_sync_start, Clock::now()) - static_cast<double>(kernel_ms));
        }
        const auto d2h_start = Clock::now();
        result = cuda_status(cudaMemcpy(output, d_output,
            hidden_width * payload_count * sizeof(float), cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_start, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(hidden_width * payload_count * sizeof(float));
        }
    } while (false);

    if (gate_start != nullptr) cudaEventDestroy(gate_start);
    if (gate_end != nullptr) cudaEventDestroy(gate_end);
    if (down_start != nullptr) cudaEventDestroy(down_start);
    if (down_end != nullptr) cudaEventDestroy(down_end);
    const auto free_start = Clock::now();
    cudaFree(d_output);
    cudaFree(d_activated_qs);
    cudaFree(d_activated_scales);
    cudaFree(d_activated);
    cudaFree(d_gate_up_values);
    cudaFree(d_payloads);
    cudaFree(d_hidden_qs);
    cudaFree(d_hidden_scales);
    cudaFree(d_hidden);
    if (timing != nullptr) {
        timing->free_ms = elapsed_ms(free_start, Clock::now());
    }
    return result;
}

extern "C" int pom_moe_nvfp4_forward_route_batch_resident(
    int device_index,
    const uint8_t * const * payloads,
    std::size_t payload_count,
    const uint32_t * token_ids,
    const float * weights,
    const std::size_t * token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float * hidden,
    std::size_t hidden_len,
    float * output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    CudaMoeCallTiming * timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(payload_count);
        timing->kernel_launches = 6;
        timing->sync_count = 2;
    }
    if (payloads == nullptr || token_ids == nullptr || weights == nullptr ||
        token_offsets == nullptr || hidden == nullptr || output == nullptr ||
        payload_count == 0 || token_count == 0 || hidden_width == 0 ||
        intermediate_width == 0 || hidden_width % kBlockElements != 0 ||
        intermediate_width % kBlockElements != 0 || token_offsets_len != token_count + 1 ||
        hidden_len != token_count * hidden_width || output_len != hidden_len ||
        token_offsets[0] != 0 || token_offsets[token_count] != payload_count) {
        return 1;
    }
    for (std::size_t pair = 0; pair < payload_count; ++pair) {
        if (payloads[pair] == nullptr || token_ids[pair] >= token_count ||
            !std::isfinite(weights[pair])) {
            return 1;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) {
            return 1;
        }
    }

    float * d_hidden = nullptr;
    float * d_hidden_scales = nullptr;
    int8_t * d_hidden_qs = nullptr;
    const uint8_t ** d_payloads = nullptr;
    uint32_t * d_token_ids = nullptr;
    float * d_weights = nullptr;
    std::size_t * d_token_offsets = nullptr;
    float * d_gate_up_values = nullptr;
    float * d_activated = nullptr;
    float * d_activated_scales = nullptr;
    int8_t * d_activated_qs = nullptr;
    float * d_pair_output = nullptr;
    float * d_output = nullptr;
    cudaEvent_t gate_start = nullptr;
    cudaEvent_t gate_end = nullptr;
    cudaEvent_t down_start = nullptr;
    cudaEvent_t down_end = nullptr;
    int result = 0;

    do {
        const auto alloc_start = Clock::now();
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden),
            token_count * hidden_width * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden_scales),
            token_count * (hidden_width / 32) * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden_qs),
            token_count * hidden_width * sizeof(int8_t)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_payloads),
            payload_count * sizeof(*d_payloads)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_ids),
            payload_count * sizeof(*d_token_ids)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_weights),
            payload_count * sizeof(*d_weights)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_offsets),
            token_offsets_len * sizeof(*d_token_offsets)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up_values),
            2 * intermediate_width * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated),
            intermediate_width * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated_scales),
            payload_count * (intermediate_width / 32) * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated_qs),
            intermediate_width * payload_count * sizeof(int8_t)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_pair_output),
            hidden_width * payload_count * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output),
            token_count * hidden_width * sizeof(float)));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->alloc_ms = elapsed_ms(alloc_start, Clock::now());
            result = cuda_status(cudaEventCreate(&gate_start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&gate_end));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&down_start));
            if (result != 0) break;
            result = cuda_status(cudaEventCreate(&down_end));
            if (result != 0) break;
        }

        const auto h2d_start = Clock::now();
        result = cuda_status(cudaMemcpy(d_hidden, hidden,
            token_count * hidden_width * sizeof(float), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_payloads, payloads,
            payload_count * sizeof(*d_payloads), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_ids, token_ids,
            payload_count * sizeof(*d_token_ids), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_weights, weights,
            payload_count * sizeof(*d_weights), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_offsets, token_offsets,
            token_offsets_len * sizeof(*d_token_offsets), cudaMemcpyHostToDevice));
        if (result != 0) break;
        if (timing != nullptr) {
            timing->h2d_ms = elapsed_ms(h2d_start, Clock::now());
            timing->h2d_bytes = static_cast<uint64_t>(
                token_count * hidden_width * sizeof(float) +
                payload_count * sizeof(*d_payloads) +
                payload_count * sizeof(*d_token_ids) +
                payload_count * sizeof(*d_weights) +
                token_offsets_len * sizeof(*d_token_offsets));
        }

        const int block_size = 128;
        const auto gate_launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(gate_start));
            if (result != 0) break;
        }
        quantize_q8_0_kernel<<<(token_count * ((hidden_width + 31) / 32)), 32>>>(
            d_hidden, static_cast<int>(hidden_width), d_hidden_scales, d_hidden_qs,
            static_cast<int>(token_count));
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        gate_up_route_batch_resident_kernel<<<
            (2 * intermediate_width * payload_count + block_size - 1) / block_size,
            block_size>>>(
                d_payloads, d_token_ids, static_cast<int>(payload_count),
                static_cast<int>(intermediate_width), static_cast<int>(hidden_width),
                d_hidden_qs, d_hidden_scales, d_gate_up_values);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        if (timing != nullptr) {
            timing->launch_ms += elapsed_ms(gate_launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(gate_end));
            if (result != 0) break;
        }
        const auto gate_sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, gate_start, gate_end));
            if (result != 0) break;
            timing->kernel_ms += kernel_ms;
            timing->sync_ms += std::max(0.0, elapsed_ms(gate_sync_start, Clock::now()) -
                static_cast<double>(kernel_ms));
        }

        const auto down_launch_start = Clock::now();
        if (timing != nullptr) {
            result = cuda_status(cudaEventRecord(down_start));
            if (result != 0) break;
        }
        geglu_kernel<<<(intermediate_width * payload_count + block_size - 1) / block_size,
            block_size>>>(
                d_gate_up_values, static_cast<int>(payload_count),
                static_cast<int>(intermediate_width), d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        quantize_q8_0_kernel<<<
            ((intermediate_width + 31) / 32) * payload_count, 32>>>(
                d_activated, static_cast<int>(intermediate_width), d_activated_scales,
                d_activated_qs, static_cast<int>(payload_count));
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        down_route_batch_resident_kernel<<<
            (hidden_width * payload_count + block_size - 1) / block_size, block_size>>>(
                d_payloads, static_cast<int>(payload_count), static_cast<int>(hidden_width),
                static_cast<int>(intermediate_width), d_activated_qs, d_activated_scales,
                d_pair_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        aggregate_route_batch_kernel<<<
            (token_count * hidden_width + block_size - 1) / block_size, block_size>>>(
                d_pair_output, d_weights, d_token_offsets, static_cast<int>(token_count),
                static_cast<int>(hidden_width), d_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        if (timing != nullptr) {
            timing->launch_ms += elapsed_ms(down_launch_start, Clock::now());
            result = cuda_status(cudaEventRecord(down_end));
            if (result != 0) break;
        }
        const auto down_sync_start = Clock::now();
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        if (timing != nullptr) {
            float kernel_ms = 0.0f;
            result = cuda_status(cudaEventElapsedTime(&kernel_ms, down_start, down_end));
            if (result != 0) break;
            timing->kernel_ms += kernel_ms;
            timing->sync_ms += std::max(0.0, elapsed_ms(down_sync_start, Clock::now()) -
                static_cast<double>(kernel_ms));
        }
        const auto d2h_start = Clock::now();
        result = cuda_status(cudaMemcpy(output, d_output,
            token_count * hidden_width * sizeof(float), cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_start, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(
                token_count * hidden_width * sizeof(float));
        }
    } while (false);

    if (gate_start != nullptr) cudaEventDestroy(gate_start);
    if (gate_end != nullptr) cudaEventDestroy(gate_end);
    if (down_start != nullptr) cudaEventDestroy(down_start);
    if (down_end != nullptr) cudaEventDestroy(down_end);
    const auto free_start = Clock::now();
    cudaFree(d_output);
    cudaFree(d_pair_output);
    cudaFree(d_activated_qs);
    cudaFree(d_activated_scales);
    cudaFree(d_activated);
    cudaFree(d_gate_up_values);
    cudaFree(d_token_offsets);
    cudaFree(d_weights);
    cudaFree(d_token_ids);
    cudaFree(d_payloads);
    cudaFree(d_hidden_qs);
    cudaFree(d_hidden_scales);
    cudaFree(d_hidden);
    if (timing != nullptr) {
        timing->free_ms = elapsed_ms(free_start, Clock::now());
    }
    return result;
}

extern "C" int pom_moe_nvfp4_route_pipeline_create(
    int device_index,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    void ** out
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (out == nullptr || hidden_width == 0 || intermediate_width == 0 ||
        hidden_width % kBlockElements != 0 || intermediate_width % kBlockElements != 0) {
        return 1;
    }
    *out = nullptr;
    auto * pipeline = new (std::nothrow) Nvfp4RoutePipeline{
        device_index,
        hidden_width,
        intermediate_width,
    };
    if (pipeline == nullptr) {
        return 2;
    }
    *out = pipeline;
    return 0;
}

extern "C" int pom_moe_nvfp4_route_pipeline_run(
    void * handle,
    const uint8_t * const * payloads,
    std::size_t payload_count,
    const uint32_t * token_ids,
    const float * weights,
    const std::size_t * token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float * hidden,
    std::size_t hidden_len,
    float * output,
    std::size_t output_len,
    CudaMoeCallTiming * timing
) {
    auto * pipeline = static_cast<Nvfp4RoutePipeline *>(handle);
    if (pipeline == nullptr) return 1;
    const ScopedMoeCudaDevice device(pipeline->device_index);
    if (!device.ready()) return 1;
    if (payloads == nullptr || token_ids == nullptr ||
        weights == nullptr || token_offsets == nullptr || hidden == nullptr ||
        output == nullptr || payload_count == 0 || token_count == 0 ||
        token_offsets_len != token_count + 1 ||
        hidden_len != token_count * pipeline->hidden_width ||
        output_len != hidden_len || token_offsets[0] != 0 ||
        token_offsets[token_count] != payload_count) {
        return 1;
    }
    for (std::size_t route = 0; route < payload_count; ++route) {
        if (payloads[route] == nullptr || token_ids[route] >= token_count ||
            !std::isfinite(weights[route])) {
            return 1;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) {
            return 1;
        }
    }
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(payload_count);
    }

    int result = ensure_route_pipeline_capacity(pipeline, token_count, payload_count);
    if (result != 0) return result;

    const std::size_t split = token_count / 2;
    const bool two_chunks = token_count > 1 && token_offsets[split] > token_offsets[0] &&
        token_offsets[token_count] > token_offsets[split];
    const std::size_t chunk_count = two_chunks ? 2 : 1;
    const std::size_t chunk_starts[2] = {0, split};
    const std::size_t chunk_ends[2] = {two_chunks ? split : token_count, token_count};
    const std::size_t first_slot = pipeline->next_slot;
    if (timing != nullptr) {
        timing->kernel_launches = static_cast<uint64_t>(6 * chunk_count);
        timing->sync_count = static_cast<uint64_t>(chunk_count);
        timing->buffer_switches = chunk_count > 1 ? chunk_count - 1 : 0;
    }

    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const std::size_t token_start = chunk_starts[chunk];
        const std::size_t token_end = chunk_ends[chunk];
        const std::size_t route_start = token_offsets[token_start];
        const std::size_t route_end = token_offsets[token_end];
        const std::size_t route_count = route_end - route_start;
        const std::size_t slot_index = (first_slot + chunk) % 2;
        for (std::size_t route = route_start; route < route_end; ++route) {
            if (token_ids[route] < token_start || token_ids[route] >= token_end) {
                return 1;
            }
        }
        auto * slot = &pipeline->slots[slot_index];
        copy_route_chunk_to_host(
            slot,
            payloads,
            token_ids,
            weights,
            token_offsets,
            hidden,
            token_start,
            token_end - token_start,
            route_start,
            route_count,
            pipeline->hidden_width);
        if (timing != nullptr) {
            timing->buffer_mask |= static_cast<uint64_t>(1) << slot_index;
        }
        result = enqueue_route_chunk(
            slot,
            token_end - token_start,
            route_count,
            pipeline->hidden_width,
            pipeline->intermediate_width,
            timing);
        if (result != 0) return result;
    }

    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const std::size_t token_start = chunk_starts[chunk];
        const std::size_t token_end = chunk_ends[chunk];
        const std::size_t route_count = token_offsets[token_end] - token_offsets[token_start];
        const std::size_t slot_index = (first_slot + chunk) % 2;
        auto * slot = &pipeline->slots[slot_index];
        result = collect_route_chunk(
            slot,
            token_end - token_start,
            route_count,
            pipeline->hidden_width,
            timing);
        if (result != 0) return result;
        std::memcpy(
            output + token_start * pipeline->hidden_width,
            slot->h_output,
            (token_end - token_start) * pipeline->hidden_width * sizeof(float));
    }
    pipeline->next_slot = (first_slot + chunk_count) % 2;
    return 0;
}

extern "C" void pom_moe_nvfp4_route_pipeline_destroy(void * handle) {
    auto * pipeline = static_cast<Nvfp4RoutePipeline *>(handle);
    if (pipeline == nullptr) return;
    const ScopedMoeCudaDevice device(pipeline->device_index);
    if (!device.ready()) return;
    release_route_slot(&pipeline->slots[0]);
    release_route_slot(&pipeline->slots[1]);
    delete pipeline;
}

// O callback MoE é síncrono: o llama.cpp espera o resultado antes de seguir
// para a próxima camada. Manter este workspace entre chamadas remove o ciclo
// cudaMalloc/cudaFree que antes acontecia uma vez por camada e tornava o
// prefill especialmente caro. Os payloads continuam residentes no cache de
// experts; só os ponteiros e a rota mudam a cada lote.
struct KQuantRoutePipeline {
    int device_index;
    std::size_t hidden_width;
    std::size_t intermediate_width;
    std::size_t token_capacity = 0;
    std::size_t route_capacity = 0;
    float *d_hidden = nullptr;
    block_q8_1 *d_hidden_q8 = nullptr;
    const uint8_t **d_payloads = nullptr;
    uint32_t *d_token_ids = nullptr;
    float *d_weights = nullptr;
    std::size_t *d_token_offsets = nullptr;
    float *d_gate_up = nullptr;
    float *d_activated = nullptr;
    block_q8_1 *d_activated_q8 = nullptr;
    float *d_pair_output = nullptr;
    float *d_output = nullptr;
};

void release_kquant_route_pipeline(KQuantRoutePipeline *pipeline) {
    if (pipeline == nullptr) return;
    cudaFree(pipeline->d_output);
    cudaFree(pipeline->d_pair_output);
    cudaFree(pipeline->d_activated);
    cudaFree(pipeline->d_activated_q8);
    cudaFree(pipeline->d_gate_up);
    cudaFree(pipeline->d_token_offsets);
    cudaFree(pipeline->d_weights);
    cudaFree(pipeline->d_token_ids);
    cudaFree(pipeline->d_payloads);
    cudaFree(pipeline->d_hidden);
    cudaFree(pipeline->d_hidden_q8);
    pipeline->d_output = nullptr;
    pipeline->d_pair_output = nullptr;
    pipeline->d_activated = nullptr;
    pipeline->d_activated_q8 = nullptr;
    pipeline->d_gate_up = nullptr;
    pipeline->d_token_offsets = nullptr;
    pipeline->d_weights = nullptr;
    pipeline->d_token_ids = nullptr;
    pipeline->d_payloads = nullptr;
    pipeline->d_hidden = nullptr;
    pipeline->d_hidden_q8 = nullptr;
    pipeline->token_capacity = 0;
    pipeline->route_capacity = 0;
}

int ensure_kquant_route_pipeline_capacity(
    KQuantRoutePipeline *pipeline,
    std::size_t tokens,
    std::size_t routes
) {
    if (pipeline->token_capacity >= tokens && pipeline->route_capacity >= routes) {
        return 0;
    }
    const auto fits_product = [](std::size_t lhs, std::size_t rhs) {
        return rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs;
    };
    const auto fits_product3 = [&](std::size_t first, std::size_t second, std::size_t third) {
        return fits_product(first, second) &&
            fits_product(first * second, third);
    };
    const auto fits_product4 = [&](std::size_t first, std::size_t second,
                                   std::size_t third, std::size_t fourth) {
        if (!fits_product(first, second)) return false;
        const std::size_t first_second = first * second;
        if (!fits_product(first_second, third)) return false;
        return fits_product(first_second * third, fourth);
    };
    const std::size_t hidden_q8_blocks = pipeline->hidden_width / QK8_1;
    const std::size_t activated_q8_blocks = pipeline->intermediate_width / QK8_1;
    if (tokens == 0 || routes == 0 ||
        tokens == std::numeric_limits<std::size_t>::max() ||
        !fits_product3(tokens, pipeline->hidden_width, sizeof(float)) ||
        !fits_product3(tokens, hidden_q8_blocks, sizeof(block_q8_1)) ||
        !fits_product4(2, pipeline->intermediate_width, routes, sizeof(float)) ||
        !fits_product3(routes, activated_q8_blocks, sizeof(block_q8_1)) ||
        !fits_product3(routes, pipeline->hidden_width, sizeof(float)) ||
        !fits_product3(routes, pipeline->intermediate_width, sizeof(float)) ||
        !fits_product(tokens + 1, sizeof(std::size_t))) {
        return 1;
    }
    release_kquant_route_pipeline(pipeline);
    pipeline->token_capacity = tokens;
    pipeline->route_capacity = routes;
    const std::size_t hidden_bytes = tokens * pipeline->hidden_width * sizeof(float);
    const std::size_t hidden_q8_bytes =
        tokens * (pipeline->hidden_width / QK8_1) * sizeof(block_q8_1);
    const std::size_t gate_up_bytes =
        2 * pipeline->intermediate_width * routes * sizeof(float);
    const std::size_t activated_bytes = pipeline->intermediate_width * routes * sizeof(float);
    const std::size_t activated_q8_bytes =
        routes * (pipeline->intermediate_width / QK8_1) * sizeof(block_q8_1);
    const std::size_t pair_output_bytes = pipeline->hidden_width * routes * sizeof(float);
    const std::size_t offset_bytes = (tokens + 1) * sizeof(std::size_t);

    int result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_hidden), hidden_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_hidden_q8), hidden_q8_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_payloads),
        routes * sizeof(*pipeline->d_payloads)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_token_ids),
        routes * sizeof(*pipeline->d_token_ids)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_weights),
        routes * sizeof(*pipeline->d_weights)));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_token_offsets),
        offset_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_gate_up),
        gate_up_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_activated),
        activated_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(
        reinterpret_cast<void **>(&pipeline->d_activated_q8), activated_q8_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_pair_output),
        pair_output_bytes));
    if (result != 0) goto fail;
    result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&pipeline->d_output), hidden_bytes));
    if (result != 0) goto fail;
    return 0;

fail:
    release_kquant_route_pipeline(pipeline);
    return result;
}

extern "C" int pom_moe_kquant_route_pipeline_create(
    int device_index,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    void **out
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (out == nullptr || hidden_width == 0 || intermediate_width == 0 ||
        hidden_width % 256 != 0 || intermediate_width % 256 != 0) {
        return 1;
    }
    *out = nullptr;
    auto *pipeline = new (std::nothrow) KQuantRoutePipeline{
        device_index,
        hidden_width,
        intermediate_width,
    };
    if (pipeline == nullptr) return 2;
    *out = pipeline;
    return 0;
}

extern "C" int pom_moe_kquant_route_pipeline_run(
    void *handle,
    uint32_t dtype,
    const uint8_t * const *payloads,
    std::size_t payload_count,
    const std::size_t *payload_lengths,
    const uint32_t *token_ids,
    const float *weights,
    const std::size_t *token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float *hidden,
    std::size_t hidden_len,
    float *output,
    std::size_t output_len,
    int *nonfinite_stage,
    std::size_t *nonfinite_index,
    std::size_t *nonfinite_count,
    CudaMoeCallTiming *timing
) {
    auto *pipeline = static_cast<KQuantRoutePipeline *>(handle);
    if (pipeline == nullptr) return 1;
    const ScopedMoeCudaDevice device(pipeline->device_index);
    if (!device.ready()) return 1;
    if (nonfinite_stage != nullptr) *nonfinite_stage = 0;
    if (nonfinite_index != nullptr) *nonfinite_index = 0;
    if (nonfinite_count != nullptr) *nonfinite_count = 0;
    if (timing != nullptr) {
        *timing = {};
        timing->expert_calls = static_cast<uint64_t>(payload_count);
        // IQ3_XXS follows llama.cpp's MMVQ contract and needs one q8_1
        // quantization pass for hidden and one for each activated route.
        timing->kernel_launches = dtype == 18 ? 6 : 4;
        timing->sync_count = 0;
    }
    const bool is_kquant = dtype == 10 || dtype == 11 || dtype == 12 ||
        dtype == 13 || dtype == 14;
    const bool is_iquant = dtype == 18 || dtype == 21 || dtype == 22 || dtype == 23;
    if ((!is_kquant && !is_iquant) ||
        payloads == nullptr || payload_lengths == nullptr || token_ids == nullptr ||
        weights == nullptr || token_offsets == nullptr || hidden == nullptr ||
        output == nullptr || payload_count == 0 || token_count == 0 ||
        payload_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        token_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        hidden_len != token_count * pipeline->hidden_width || output_len != hidden_len ||
        token_offsets_len != token_count + 1 || token_offsets[0] != 0 ||
        token_offsets[token_count] != payload_count) {
        return 1;
    }
    std::size_t expected_gate_up_bytes = 0;
    std::size_t expected_payload_bytes = 0;
    const bool valid_geometry = is_kquant
        ? kquant_payload_geometry(
            dtype, pipeline->hidden_width, pipeline->intermediate_width,
            &expected_gate_up_bytes, &expected_payload_bytes)
        : iquant_payload_geometry(
            dtype, pipeline->hidden_width, pipeline->intermediate_width,
            &expected_gate_up_bytes, &expected_payload_bytes);
    if (!valid_geometry) {
        return 1;
    }
    for (std::size_t pair = 0; pair < payload_count; ++pair) {
        if (payloads[pair] == nullptr || token_ids[pair] >= token_count ||
            !std::isfinite(weights[pair]) || payload_lengths[pair] != expected_payload_bytes) {
            return payload_lengths[pair] == expected_payload_bytes ? 1 : 2;
        }
    }
    for (std::size_t token = 0; token < token_count; ++token) {
        if (token_offsets[token] > token_offsets[token + 1]) return 1;
    }

    const auto allocation_started = Clock::now();
    const std::size_t old_token_capacity = pipeline->token_capacity;
    const std::size_t old_route_capacity = pipeline->route_capacity;
    const int capacity_result = ensure_kquant_route_pipeline_capacity(
        pipeline, token_count, payload_count);
    if (timing != nullptr &&
        (old_token_capacity != pipeline->token_capacity || old_route_capacity != pipeline->route_capacity)) {
        timing->alloc_ms = elapsed_ms(allocation_started, Clock::now());
    }
    if (capacity_result != 0) return capacity_result;

    const int pair_count = static_cast<int>(payload_count);
    const int token_count_i = static_cast<int>(token_count);
    const int hidden_i = static_cast<int>(pipeline->hidden_width);
    const int intermediate_i = static_cast<int>(pipeline->intermediate_width);
    const int block_size = 128;
    const std::size_t hidden_bytes =
        pipeline->hidden_width * token_count * sizeof(float);
    const std::size_t gate_up_bytes =
        2 * pipeline->intermediate_width * payload_count * sizeof(float);
    const std::size_t activated_bytes =
        pipeline->intermediate_width * payload_count * sizeof(float);
    const std::size_t pair_output_bytes =
        pipeline->hidden_width * payload_count * sizeof(float);

    const auto h2d_started = Clock::now();
    int result = cuda_status(cudaMemcpy(
        pipeline->d_hidden, hidden, hidden_bytes, cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_payloads, payloads,
        payload_count * sizeof(*pipeline->d_payloads), cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_token_ids, token_ids,
        payload_count * sizeof(*pipeline->d_token_ids), cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_weights, weights,
        payload_count * sizeof(*pipeline->d_weights), cudaMemcpyHostToDevice));
    if (result != 0) return result;
    result = cuda_status(cudaMemcpy(
        pipeline->d_token_offsets, token_offsets,
        token_offsets_len * sizeof(*pipeline->d_token_offsets), cudaMemcpyHostToDevice));
    if (timing != nullptr) {
        timing->h2d_ms = elapsed_ms(h2d_started, Clock::now());
        timing->h2d_bytes = static_cast<uint64_t>(
            hidden_bytes + payload_count * sizeof(*pipeline->d_payloads) +
            payload_count * sizeof(*pipeline->d_token_ids) +
            payload_count * sizeof(*pipeline->d_weights) +
            token_offsets_len * sizeof(*pipeline->d_token_offsets));
    }
    if (result != 0) return result;

    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    const auto launch_started = Clock::now();
    if (timing != nullptr) {
        result = cuda_status(cudaEventCreate(&start));
        if (result != 0) return result;
        result = cuda_status(cudaEventCreate(&end));
        if (result != 0) {
            cudaEventDestroy(start);
            return result;
        }
        result = cuda_status(cudaEventRecord(start));
        if (result != 0) goto finish;
    }
    if (dtype == 18) {
        quantize_q8_1_rows_kernel<<<
            token_count_i * (hidden_i / QK8_1), 32>>>(
                pipeline->d_hidden, token_count_i, hidden_i, pipeline->d_hidden_q8);
        result = cuda_status(cudaGetLastError());
        if (result != 0) goto finish;
    }
    if (is_kquant) {
        kquant_gate_up_cooperative_route_kernel<<<
            2 * intermediate_i * pair_count, block_size>>>(
                dtype, pipeline->d_payloads, pipeline->d_token_ids, pair_count,
                intermediate_i, hidden_i, pipeline->d_hidden, pipeline->d_gate_up);
    } else if (dtype == 18) {
        iquant_gate_up_iq3xxs_q8_warp_route_kernel<<<
            intermediate_i * pair_count, 32>>>(
                pipeline->d_payloads, pipeline->d_token_ids, pair_count,
                intermediate_i, hidden_i, pipeline->d_hidden_q8, pipeline->d_gate_up);
    } else {
        iquant_gate_up_cooperative_route_kernel<<<
            2 * intermediate_i * pair_count, block_size>>>(
                dtype, pipeline->d_payloads, pipeline->d_token_ids, pair_count,
                intermediate_i, hidden_i, pipeline->d_hidden, pipeline->d_gate_up);
    }
    result = cuda_status(cudaGetLastError());
    if (result != 0) goto finish;
    kquant_silu_kernel<<<
        (intermediate_i * pair_count + block_size - 1) / block_size,
        block_size>>>(pipeline->d_gate_up, pair_count, intermediate_i, pipeline->d_activated);
    result = cuda_status(cudaGetLastError());
    if (result != 0) goto finish;
    if (dtype == 18) {
        quantize_q8_1_rows_kernel<<<
            pair_count * (intermediate_i / QK8_1), 32>>>(
                pipeline->d_activated, pair_count, intermediate_i,
                pipeline->d_activated_q8);
        result = cuda_status(cudaGetLastError());
        if (result != 0) goto finish;
    }
    if (is_kquant) {
        kquant_down_cooperative_route_kernel<<<hidden_i * pair_count, block_size>>>(
            dtype, pipeline->d_payloads, pair_count, hidden_i, intermediate_i,
            expected_gate_up_bytes, pipeline->d_activated, pipeline->d_pair_output);
    } else if (dtype == 18) {
        iquant_down_iq3xxs_q8_warp_route_kernel<<<hidden_i * pair_count, 32>>>(
            pipeline->d_payloads, pair_count, hidden_i, intermediate_i,
            expected_gate_up_bytes, pipeline->d_activated_q8, pipeline->d_pair_output);
    } else {
        iquant_down_cooperative_route_kernel<<<hidden_i * pair_count, block_size>>>(
            dtype, pipeline->d_payloads, pair_count, hidden_i, intermediate_i,
            expected_gate_up_bytes, pipeline->d_activated, pipeline->d_pair_output);
    }
    result = cuda_status(cudaGetLastError());
    if (result != 0) goto finish;
    aggregate_route_batch_kernel<<<
        (static_cast<int>(token_count) * hidden_i + block_size - 1) / block_size,
        block_size>>>(pipeline->d_pair_output, pipeline->d_weights, pipeline->d_token_offsets,
            token_count_i, hidden_i, pipeline->d_output);
    result = cuda_status(cudaGetLastError());
    if (timing != nullptr) timing->launch_ms = elapsed_ms(launch_started, Clock::now());
    if (result != 0) goto finish;
    if (timing != nullptr) {
        result = cuda_status(cudaEventRecord(end));
        if (result != 0) goto finish;
    }
    {
        const auto d2h_started = Clock::now();
        result = cuda_status(cudaMemcpy(
            output, pipeline->d_output, hidden_bytes, cudaMemcpyDeviceToHost));
        if (timing != nullptr) {
            timing->d2h_ms = elapsed_ms(d2h_started, Clock::now());
            timing->d2h_bytes = static_cast<uint64_t>(hidden_bytes);
        }
    }
    if (result != 0) goto finish;
    if (timing != nullptr) {
        // A synchronous D2H copy is the completion barrier for the default
        // stream. There is no standalone cudaDeviceSynchronize on this hot
        // path; measuring the event after the copy preserves the kernel span.
        float kernel_ms = 0.0f;
        result = cuda_status(cudaEventElapsedTime(&kernel_ms, start, end));
        if (result != 0) goto finish;
        timing->kernel_ms = kernel_ms;
        timing->sync_ms = 0.0;
    }
    if (result == 0 && first_nonfinite(
            output, hidden_bytes / sizeof(float), nonfinite_index, nonfinite_count)) {
        std::vector<float> stage_values;
        int copy_status = 0;
        if (device_has_nonfinite(
                pipeline->d_gate_up, gate_up_bytes / sizeof(float), stage_values,
                &copy_status, nullptr, nullptr)) {
            if (nonfinite_stage != nullptr) *nonfinite_stage = 1;
        } else if (copy_status != 0) {
            result = copy_status;
        } else if (device_has_nonfinite(
                pipeline->d_activated, activated_bytes / sizeof(float), stage_values,
                &copy_status, nullptr, nullptr)) {
            if (nonfinite_stage != nullptr) *nonfinite_stage = 2;
        } else if (copy_status != 0) {
            result = copy_status;
        } else if (device_has_nonfinite(
                pipeline->d_pair_output, pair_output_bytes / sizeof(float), stage_values,
                &copy_status, nullptr, nullptr)) {
            if (nonfinite_stage != nullptr) *nonfinite_stage = 3;
        } else if (copy_status != 0) {
            result = copy_status;
        } else if (nonfinite_stage != nullptr) {
            *nonfinite_stage = 4;
        }
    }

finish:
    if (start != nullptr) cudaEventDestroy(start);
    if (end != nullptr) cudaEventDestroy(end);
    return result;
}

extern "C" void pom_moe_kquant_route_pipeline_destroy(void *handle) {
    auto *pipeline = static_cast<KQuantRoutePipeline *>(handle);
    if (pipeline == nullptr) return;
    const ScopedMoeCudaDevice device(pipeline->device_index);
    if (!device.ready()) return;
    release_kquant_route_pipeline(pipeline);
    delete pipeline;
}

// IQ e K-quant compartilham o mesmo workspace residente e o mesmo contrato
// de lote. Mantemos símbolos distintos para que a seleção Rust continue
// explícita por formato, sem duplicar a implementação ou o estado CUDA.
extern "C" int pom_moe_iquant_route_pipeline_create(
    int device_index,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    void **out
) {
    return pom_moe_kquant_route_pipeline_create(
        device_index, hidden_width, intermediate_width, out);
}

extern "C" int pom_moe_iquant_route_pipeline_run(
    void *handle,
    uint32_t dtype,
    const uint8_t * const *payloads,
    std::size_t payload_count,
    const std::size_t *payload_lengths,
    const uint32_t *token_ids,
    const float *weights,
    const std::size_t *token_offsets,
    std::size_t token_offsets_len,
    std::size_t token_count,
    const float *hidden,
    std::size_t hidden_len,
    float *output,
    std::size_t output_len,
    int *nonfinite_stage,
    std::size_t *nonfinite_index,
    std::size_t *nonfinite_count,
    CudaMoeCallTiming *timing
) {
    return pom_moe_kquant_route_pipeline_run(
        handle, dtype, payloads, payload_count, payload_lengths, token_ids, weights,
        token_offsets, token_offsets_len, token_count, hidden, hidden_len, output,
        output_len, nonfinite_stage, nonfinite_index, nonfinite_count, timing);
}

extern "C" void pom_moe_iquant_route_pipeline_destroy(void *handle) {
    pom_moe_kquant_route_pipeline_destroy(handle);
}

// Diagnóstico por estágio para comparar o decoder CUDA com o oráculo GGML.
// Mantém a mesma sequência do caminho de produção, mas devolve gate/up,
// ativação e down separadamente para localizar a primeira divergência.
extern "C" int pom_moe_iquant_trace_single_resident(
    int device_index,
    uint32_t dtype,
    const uint8_t * payload,
    const float * hidden,
    std::size_t hidden_len,
    float * gate_up,
    std::size_t gate_up_len,
    float * activated,
    std::size_t activated_len,
    float * output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if ((dtype != 18 && dtype != 21 && dtype != 22 && dtype != 23) || payload == nullptr ||
        hidden == nullptr || gate_up == nullptr || activated == nullptr ||
        output == nullptr || hidden_width == 0 || intermediate_width == 0 ||
        hidden_len != hidden_width || gate_up_len != 2 * intermediate_width * sizeof(float) ||
        activated_len != intermediate_width * sizeof(float) ||
        output_len != hidden_width * sizeof(float)) {
        return 1;
    }

    const int hidden_i = static_cast<int>(hidden_width);
    const int intermediate_i = static_cast<int>(intermediate_width);
    const int block_size = 128;
    const std::size_t gate_up_bytes = gate_up_len;
    const std::size_t activated_bytes = activated_len;
    const std::size_t output_bytes = output_len;
    const uint32_t token_id = 0;
    const uint8_t ** d_payloads = nullptr;
    uint32_t * d_token_ids = nullptr;
    float * d_hidden = nullptr;
    float * d_gate_up = nullptr;
    float * d_activated = nullptr;
    float * d_output = nullptr;
    int result = 0;

    do {
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_payloads), sizeof(*d_payloads)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_token_ids), sizeof(*d_token_ids)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_hidden), hidden_len * sizeof(float)));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_gate_up), gate_up_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_activated), activated_bytes));
        if (result != 0) break;
        result = cuda_status(cudaMalloc(reinterpret_cast<void **>(&d_output), output_bytes));
        if (result != 0) break;

        result = cuda_status(cudaMemcpy(d_payloads, &payload, sizeof(payload), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_token_ids, &token_id, sizeof(token_id), cudaMemcpyHostToDevice));
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(d_hidden, hidden, hidden_len * sizeof(float), cudaMemcpyHostToDevice));
        if (result != 0) break;

        iquant_gate_up_cooperative_route_kernel<<<
            2 * intermediate_i, block_size>>>(dtype, d_payloads, d_token_ids, 1,
                intermediate_i, hidden_i, d_hidden, d_gate_up);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(gate_up, d_gate_up, gate_up_bytes, cudaMemcpyDeviceToHost));
        if (result != 0) break;

        kquant_silu_kernel<<<
            (intermediate_i + block_size - 1) / block_size,
            block_size>>>(d_gate_up, 1, intermediate_i, d_activated);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(activated, d_activated, activated_bytes, cudaMemcpyDeviceToHost));
        if (result != 0) break;

        iquant_down_cooperative_route_kernel<<<
            hidden_i, block_size>>>(dtype, d_payloads, 1, hidden_i, intermediate_i,
                gate_up_bytes, d_activated, d_output);
        result = cuda_status(cudaGetLastError());
        if (result != 0) break;
        result = cuda_status(cudaDeviceSynchronize());
        if (result != 0) break;
        result = cuda_status(cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    } while (false);

    cudaFree(d_output);
    cudaFree(d_activated);
    cudaFree(d_gate_up);
    cudaFree(d_hidden);
    cudaFree(d_token_ids);
    cudaFree(d_payloads);
    return result;
}

extern "C" int pom_moe_device_count(int * device_count) {
    if (device_count == nullptr) return 1;
    return cuda_status(cudaGetDeviceCount(device_count));
}

extern "C" int pom_moe_device_free_bytes(int device_index, std::uint64_t * free_bytes) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (free_bytes == nullptr) {
        return 1;
    }
    std::size_t free = 0;
    std::size_t total = 0;
    const auto status = cudaMemGetInfo(&free, &total);
    if (status != cudaSuccess) {
        return cuda_status(status);
    }
    *free_bytes = static_cast<std::uint64_t>(free);
    return 0;
}

extern "C" int pom_moe_device_upload_many(
    int device_index,
    const std::uint8_t * const * payloads,
    const std::size_t * payload_lengths,
    std::size_t payload_count,
    void ** out
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    if (payloads == nullptr || payload_lengths == nullptr || out == nullptr ||
        payload_count == 0 ||
        payload_count > std::numeric_limits<std::size_t>::max() / sizeof(void *)) {
        return 1;
    }
    for (std::size_t index = 0; index < payload_count; ++index) {
        out[index] = nullptr;
        if (payloads[index] == nullptr || payload_lengths[index] == 0) return 1;
    }

    // Use the device memory pool so a cache eviction does not force the
    // legacy cudaFree synchronization on every expert. The copy itself stays
    // synchronous because callers may release a temporary mmap/Vec as soon
    // as this function returns; subsequent kernels still observe the same
    // default-stream ordering as before.
    int status = 0;
    for (std::size_t index = 0; index < payload_count; ++index) {
        status = cuda_status(cudaMallocAsync(&out[index], payload_lengths[index], nullptr));
        if (status != 0) goto fail;
    }
    for (std::size_t index = 0; index < payload_count; ++index) {
        status = cuda_status(cudaMemcpyAsync(
            out[index], payloads[index], payload_lengths[index], cudaMemcpyHostToDevice, nullptr));
        if (status != 0) goto fail;
    }
    status = cuda_status(cudaStreamSynchronize(nullptr));
    if (status == 0) return 0;

fail:
    for (std::size_t index = 0; index < payload_count; ++index) {
        if (out[index] != nullptr) {
            (void) cudaFreeAsync(out[index], nullptr);
            out[index] = nullptr;
        }
    }
    // Cleanup is an error-only path. Complete the queued frees so a caller
    // can immediately retry without retaining failed batch allocations.
    (void) cudaStreamSynchronize(nullptr);
    return status;
}

extern "C" int pom_moe_device_upload(
    int device_index,
    const std::uint8_t * payload,
    std::size_t payload_len,
    void ** out
) {
    if (out == nullptr) return 1;
    const std::uint8_t * payloads[] = {payload};
    const std::size_t payload_lengths[] = {payload_len};
    void * handles[] = {nullptr};
    const int status = pom_moe_device_upload_many(
        device_index, payloads, payload_lengths, 1, handles);
    if (status == 0) *out = handles[0];
    return status;
}

extern "C" void pom_moe_device_free(int device_index, void * handle) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return;
    if (handle != nullptr) {
        // All MoE kernels and the upload above use the device default stream.
        // Queueing the release there preserves ordering while avoiding a
        // device-wide host wait for every LRU eviction.
        (void) cudaFreeAsync(handle, nullptr);
    }
}

extern "C" int pom_moe_nvfp4_forward(
    int device_index,
    const uint8_t * payload,
    std::size_t payload_len,
    const float * hidden,
    std::size_t hidden_len,
    float * output,
    std::size_t output_len,
    std::size_t hidden_width,
    std::size_t intermediate_width,
    CudaMoeCallTiming * timing
) {
    const ScopedMoeCudaDevice device(device_index);
    if (!device.ready()) return 1;
    return pom_moe_nvfp4_forward_batch(
        device_index,
        payload,
        payload_len,
        1,
        hidden,
        hidden_len,
        output,
        output_len,
        hidden_width,
        intermediate_width,
        timing
    );
}
