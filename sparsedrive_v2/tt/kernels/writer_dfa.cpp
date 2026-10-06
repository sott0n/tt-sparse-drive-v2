// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused deformable-aggregation writer: for each anchor, row t of ACC tile t (fp32) holds
// channels 32t..32t+31 of the output row (see compute_dfa.cpp).

#include <stdint.h>
#include "api/compile_time_args.h"
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

constexpr uint32_t cb_acc = get_compile_time_arg_val(0);
constexpr uint32_t c_tiles = get_compile_time_arg_val(1);
constexpr auto out_args = TensorAccessorArgs<2>();

constexpr uint32_t TILE_NBYTES = 4096;    // fp32 32x32
constexpr uint32_t FACE_NBYTES = 1024;    // fp32 16x16
constexpr uint32_t HALF_ROW_NBYTES = 64;  // 16 fp32

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t anchor_start = get_arg_val<uint32_t>(1);
    const uint32_t anchor_count = get_arg_val<uint32_t>(2);

    const auto out = TensorAccessor(out_args, out_addr);
    DataflowBuffer acc_dfb(cb_acc);
    Noc noc;

    for (uint32_t a = anchor_start; a < anchor_start + anchor_count; ++a) {
        acc_dfb.wait_front(c_tiles);
        for (uint32_t t = 0; t < c_tiles; ++t) {
            const uint32_t row = t * TILE_NBYTES + t * HALF_ROW_NBYTES;
            const uint32_t col = t * 2 * HALF_ROW_NBYTES;
            noc.async_write(acc_dfb, out, HALF_ROW_NBYTES, {.offset_bytes = row}, {.page_id = a, .offset_bytes = col});
            noc.async_write(
                acc_dfb,
                out,
                HALF_ROW_NBYTES,
                {.offset_bytes = row + FACE_NBYTES},
                {.page_id = a, .offset_bytes = col + HALF_ROW_NBYTES});
        }
        noc.async_write_barrier();
        acc_dfb.pop_front(c_tiles);
    }
}
