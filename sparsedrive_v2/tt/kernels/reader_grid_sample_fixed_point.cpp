// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Bilinear grid_sample reader (interleaved, FLOAT32 grid, align_corners=False, zeros padding).
//
// Drop-in replacement for ttnn's reader_grid_sample_interleaved_start_id.cpp, paired with the
// same compute (pool/generic compute_pool_2d.cpp) and writer kernels. It differs in two ways:
//
//  * The sampling geometry is computed in Q16.16 integers instead of float. The data-movement
//    cores have no FPU, so every float op in the stock reader is a software routine; that
//    geometry was ~60% of the reader's time and the reader bounds the op.
//  * Grid sticks are fetched GRID_BLOCK at a time behind a single read barrier instead of one
//    barrier per point.

#include <stdint.h>
#include "api/compile_time_args.h"
#include <api/dataflow/dataflow_api.h>
#include "ttnn/operations/pool/device/kernels/pool_kernels_common.hpp"
#include "ttnn/operations/pool/grid_sample/device/kernels/grid_sample_reader_common.hpp"

namespace {

constexpr int32_t Q16_SHIFT = 16;
constexpr int32_t Q16_ONE = 1 << Q16_SHIFT;
constexpr int32_t Q16_HALF = 1 << (Q16_SHIFT - 1);
// Coordinates are saturated at +-2 (grid space): far enough that a clamped out-of-range point
// still maps outside the image, small enough that (g + 1) * size stays inside int32.
constexpr int32_t GRID_CLAMP_Q16 = 2 * Q16_ONE;

// fp32 bit pattern -> Q16.16, saturating at +-GRID_CLAMP_Q16. Integer ops only.
inline int32_t fp32_to_q16(uint32_t bits) {
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu);
    if (exp == 0) {
        return 0;  // zero / subnormal: below Q16.16 resolution
    }
    const uint32_t significand = 0x800000u | (bits & 0x7FFFFFu);  // 1.m, 24 bits
    const int32_t shift = exp - 134;                              // value * 2^16 = significand * 2^shift
    int32_t magnitude;
    if (shift >= 8) {
        magnitude = GRID_CLAMP_Q16;  // |value| >= 256, far past the clamp
    } else if (shift >= 0) {
        magnitude = static_cast<int32_t>(significand << shift);
    } else if (shift > -32) {
        magnitude = static_cast<int32_t>(significand >> (-shift));
    } else {
        magnitude = 0;
    }
    if (magnitude > GRID_CLAMP_Q16) {
        magnitude = GRID_CLAMP_Q16;
    }
    return (bits & 0x80000000u) ? -magnitude : magnitude;
}

// Q16.16 weight in [0, 1] -> bf16 bits, truncating like the stock reader's fp32_to_bf16_truncate.
inline uint16_t q16_weight_to_bf16(uint32_t w) {
    if (w == 0) {
        return 0;
    }
    const int32_t msb = 31 - __builtin_clz(w);
    const uint32_t mantissa = (msb >= 7 ? (w >> (msb - 7)) : (w << (7 - msb))) & 0x7Fu;
    const uint32_t exponent = static_cast<uint32_t>(127 + msb - Q16_SHIFT);
    return static_cast<uint16_t>((exponent << 7) | mantissa);
}

// (a * b) >> 16 for a, b in [0, Q16_ONE]; the product can reach 2^32, so widen.
inline uint32_t q16_mul(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> Q16_SHIFT);
}

}  // namespace

void kernel_main() {
    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t grid_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_pages = get_arg_val<uint32_t>(2);
    const uint32_t start_page_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t grid_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t scalar_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t input_stick_nbytes = get_compile_time_arg_val(3);
    constexpr uint32_t grid_stick_nbytes = get_compile_time_arg_val(4);
    constexpr uint32_t input_height = get_compile_time_arg_val(5);
    constexpr uint32_t input_width = get_compile_time_arg_val(6);
    constexpr uint32_t output_hw_size = get_compile_time_arg_val(7);
    constexpr uint32_t grid_block = get_compile_time_arg_val(8);

    constexpr auto src_args = TensorAccessorArgs<9>();
    constexpr auto grid_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto input_accessor = TensorAccessor(src_args, input_addr);
    const auto grid_accessor = TensorAccessor(grid_args, grid_addr);

    DataflowBuffer grid_dfb(grid_cb_index);
    DataflowBuffer input_dfb(input_cb_index);
    DataflowBuffer scalar_dfb(scalar_cb_index);
    Noc noc;

    // Corners outside the image are not read, so the CB slot keeps whatever it held; the weight
    // is zero, but stale garbage could be NaN/Inf. Start from zeros (same as the stock reader).
    zero_out_tiles<input_cb_index>(noc, input_dfb);

    constexpr int32_t in_h = static_cast<int32_t>(input_height);
    constexpr int32_t in_w = static_cast<int32_t>(input_width);
    const uint32_t grid_l1 = grid_dfb.get_write_ptr();

    uint32_t batch = start_page_id / output_hw_size;
    uint32_t in_batch_pos = start_page_id % output_hw_size;
    const uint32_t end_id = start_page_id + num_pages;

    for (uint32_t block_start = start_page_id; block_start < end_id; block_start += grid_block) {
        const uint32_t block_len = (end_id - block_start) < grid_block ? (end_id - block_start) : grid_block;
        for (uint32_t i = 0; i < block_len; ++i) {
            noc.async_read(
                grid_accessor,
                grid_dfb,
                grid_stick_nbytes,
                {.page_id = block_start + i},
                {.offset_bytes = i * grid_stick_nbytes});
        }
        noc.async_read_barrier();

        for (uint32_t i = 0; i < block_len; ++i) {
            volatile tt_l1_ptr uint32_t* g =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(grid_l1 + i * grid_stick_nbytes);
            // align_corners=False: pixel = (g + 1) * size / 2 - 0.5
            const int32_t px = ((fp32_to_q16(g[0]) + Q16_ONE) * in_w >> 1) - Q16_HALF;
            const int32_t py = ((fp32_to_q16(g[1]) + Q16_ONE) * in_h >> 1) - Q16_HALF;
            const int32_t w0 = px >> Q16_SHIFT;  // arithmetic shift == floor
            const int32_t h0 = py >> Q16_SHIFT;
            const int32_t w1 = w0 + 1;
            const int32_t h1 = h0 + 1;
            const uint32_t fw = static_cast<uint32_t>(px) & (Q16_ONE - 1);
            const uint32_t fh = static_cast<uint32_t>(py) & (Q16_ONE - 1);
            const uint32_t gw = Q16_ONE - fw;
            const uint32_t gh = Q16_ONE - fh;

            const bool h0_ok = h0 >= 0 && h0 < in_h;
            const bool h1_ok = h1 >= 0 && h1 < in_h;
            const bool w0_ok = w0 >= 0 && w0 < in_w;
            const bool w1_ok = w1 >= 0 && w1 < in_w;
            const uint16_t w_nw = (h0_ok && w0_ok) ? q16_weight_to_bf16(q16_mul(gh, gw)) : 0;
            const uint16_t w_ne = (h0_ok && w1_ok) ? q16_weight_to_bf16(q16_mul(gh, fw)) : 0;
            const uint16_t w_sw = (h1_ok && w0_ok) ? q16_weight_to_bf16(q16_mul(fh, gw)) : 0;
            const uint16_t w_se = (h1_ok && w1_ok) ? q16_weight_to_bf16(q16_mul(fh, fw)) : 0;

            scalar_dfb.reserve_back(1);
            fill_four_val<scalar_cb_index>(scalar_dfb.get_write_ptr(), w_nw, w_ne, w_sw, w_se);
            scalar_dfb.push_back(1);

            input_dfb.reserve_back(1);
            read_four_corner_inputs(
                noc,
                input_accessor,
                batch * input_height * input_width,
                input_width,
                input_stick_nbytes,
                input_stick_nbytes,
                0,
                h0,
                h1,
                w0,
                w1,
                input_height,
                input_dfb);
            noc.async_read_barrier();
            input_dfb.push_back(1);

            if (++in_batch_pos == output_hw_size) {
                in_batch_pos = 0;
                ++batch;
            }
        }
    }
}
