// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused deformable-aggregation reader.
//
// For each anchor it walks (camera, key point, level, bilinear corner) and keeps only corners
// that land inside the feature map. Every 32 kept samples form one "sample tile":
//   V_RM : 32 row-major feature sticks (C channels each)      -> tilized by compute
//   ET   : tile, row g / column s = softmax weight of group g for sample s
//   W    : tile, row 0 / column s = bilinear weight of sample s
// Compute reduces sum_s ET[g, s] * W[s] * V[s, :] per group. Columns past the last kept
// sample are zero in ET and W, so stale V rows contribute nothing.
//
// The anchor's last sample tile carries LAST_FLAG = 1 in W's face 2 (outside row 0, which is
// all the row broadcast reads); compute reads it to know where the anchor ends.

#include <stdint.h>
#include "api/compile_time_args.h"
#include <api/dataflow/dataflow_api.h>
#include "api/dataflow/dataflow_buffer.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "ttnn/operations/pool/device/kernels/pool_kernels_common.hpp"
#include "fixed_point_geometry.hpp"

using namespace sdv2;

constexpr uint32_t cb_ex = get_compile_time_arg_val(0);
constexpr uint32_t cb_grid = get_compile_time_arg_val(1);
constexpr uint32_t cb_v_rm = get_compile_time_arg_val(2);
constexpr uint32_t cb_et = get_compile_time_arg_val(3);
constexpr uint32_t cb_w = get_compile_time_arg_val(4);
constexpr uint32_t num_cams = get_compile_time_arg_val(5);
constexpr uint32_t num_anchors_total = get_compile_time_arg_val(6);
constexpr uint32_t num_pts = get_compile_time_arg_val(7);         // real key points per anchor
constexpr uint32_t num_pts_padded = get_compile_time_arg_val(8);  // row stride of ex / grid
constexpr uint32_t num_groups = get_compile_time_arg_val(9);
constexpr uint32_t stick_nbytes = get_compile_time_arg_val(10);
constexpr uint32_t c_tiles = get_compile_time_arg_val(11);
constexpr uint32_t ex_row_nbytes = get_compile_time_arg_val(12);
constexpr uint32_t grid_row_nbytes = get_compile_time_arg_val(13);
constexpr uint32_t num_levels = 4;
constexpr uint32_t level_h[num_levels] = {
    get_compile_time_arg_val(14),
    get_compile_time_arg_val(15),
    get_compile_time_arg_val(16),
    get_compile_time_arg_val(17)};
constexpr uint32_t level_w[num_levels] = {
    get_compile_time_arg_val(18),
    get_compile_time_arg_val(19),
    get_compile_time_arg_val(20),
    get_compile_time_arg_val(21)};

constexpr auto ex_args = TensorAccessorArgs<22>();
constexpr auto grid_args = TensorAccessorArgs<ex_args.next_compile_time_args_offset()>();
constexpr auto v0_args = TensorAccessorArgs<grid_args.next_compile_time_args_offset()>();
constexpr auto v1_args = TensorAccessorArgs<v0_args.next_compile_time_args_offset()>();
constexpr auto v2_args = TensorAccessorArgs<v1_args.next_compile_time_args_offset()>();
constexpr auto v3_args = TensorAccessorArgs<v2_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t FACE_NBYTES = 512;        // 16 x 16 bf16
constexpr uint32_t LAST_FLAG_OFFSET = 1024;  // face 2, element 0

// Byte offset of bf16 element (row < 16, col) in a 32x32 tile.
inline uint32_t tile_offset(uint32_t row, uint32_t col) {
    return (col >= 16 ? FACE_NBYTES : 0) + ((row * 16 + (col & 15)) << 1);
}

inline void clear_columns(uint32_t et_l1, uint32_t w_l1, uint32_t first_col) {
    for (uint32_t s = first_col; s < TILE_ROWS; ++s) {
        for (uint32_t g = 0; g < num_groups; ++g) {
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(et_l1 + tile_offset(g, s)) = 0;
        }
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1 + tile_offset(0, s)) = 0;
    }
}

void kernel_main() {
    const uint32_t ex_addr = get_arg_val<uint32_t>(0);
    const uint32_t grid_addr = get_arg_val<uint32_t>(1);
    const uint32_t value_addr[num_levels] = {
        get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5)};
    const uint32_t anchor_start = get_arg_val<uint32_t>(6);
    const uint32_t anchor_count = get_arg_val<uint32_t>(7);

    const auto ex_acc = TensorAccessor(ex_args, ex_addr);
    const auto grid_acc = TensorAccessor(grid_args, grid_addr);
    const auto v0 = TensorAccessor(v0_args, value_addr[0]);
    const auto v1 = TensorAccessor(v1_args, value_addr[1]);
    const auto v2 = TensorAccessor(v2_args, value_addr[2]);
    const auto v3 = TensorAccessor(v3_args, value_addr[3]);

    DataflowBuffer ex_dfb(cb_ex);
    DataflowBuffer grid_dfb(cb_grid);
    DataflowBuffer v_dfb(cb_v_rm);
    DataflowBuffer et_dfb(cb_et);
    DataflowBuffer w_dfb(cb_w);
    Noc noc;

    // Scratch arenas: reserved once, never pushed.
    ex_dfb.reserve_back(1);
    grid_dfb.reserve_back(1);
    const uint32_t ex_l1 = ex_dfb.get_write_ptr();
    const uint32_t grid_l1 = grid_dfb.get_write_ptr();

    // Stale V rows are multiplied by zero weights, so they must at least be finite; ET rows past
    // num_groups and W rows past 0 are never written.
    zero_out_tiles<cb_v_rm>(noc, v_dfb);
    zero_out_tiles<cb_et>(noc, et_dfb);
    zero_out_tiles<cb_w>(noc, w_dfb);

    for (uint32_t a = anchor_start; a < anchor_start + anchor_count; ++a) {
        for (uint32_t c = 0; c < num_cams; ++c) {
            const uint32_t page = c * num_anchors_total + a;
            noc.async_read(ex_acc, ex_dfb, ex_row_nbytes, {.page_id = page}, {.offset_bytes = c * ex_row_nbytes});
            noc.async_read(
                grid_acc, grid_dfb, grid_row_nbytes, {.page_id = page}, {.offset_bytes = c * grid_row_nbytes});
        }
        noc.async_read_barrier();

        v_dfb.reserve_back(c_tiles);
        et_dfb.reserve_back(1);
        w_dfb.reserve_back(1);
        uint32_t et_l1 = et_dfb.get_write_ptr();
        uint32_t w_l1 = w_dfb.get_write_ptr();
        uint32_t s = 0;

        for (uint32_t c = 0; c < num_cams; ++c) {
            volatile tt_l1_ptr uint32_t* grid =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(grid_l1 + c * grid_row_nbytes);
            const uint32_t ex_cam_l1 = ex_l1 + c * ex_row_nbytes;
            for (uint32_t p = 0; p < num_pts; ++p) {
                const int32_t gx = fp32_to_q16(grid[2 * p]);
                const int32_t gy = fp32_to_q16(grid[2 * p + 1]);
                // Points outside the image (or masked) were moved to -3 by the host graph.
                if (gx <= -Q16_ONE || gx >= Q16_ONE || gy <= -Q16_ONE || gy >= Q16_ONE) {
                    continue;
                }
                for (uint32_t l = 0; l < num_levels; ++l) {
                    const int32_t h = static_cast<int32_t>(level_h[l]);
                    const int32_t w = static_cast<int32_t>(level_w[l]);
                    const int32_t px = grid_to_pixel(gx, w);
                    const int32_t py = grid_to_pixel(gy, h);
                    const int32_t x0 = px >> Q16_SHIFT;
                    const int32_t y0 = py >> Q16_SHIFT;
                    const uint32_t fx = static_cast<uint32_t>(px) & (Q16_ONE - 1);
                    const uint32_t fy = static_cast<uint32_t>(py) & (Q16_ONE - 1);
                    const uint32_t corner_w[4] = {
                        q16_mul(Q16_ONE - fy, Q16_ONE - fx),
                        q16_mul(Q16_ONE - fy, fx),
                        q16_mul(fy, Q16_ONE - fx),
                        q16_mul(fy, fx)};
                    // ex row layout: (group, level, point)
                    const uint32_t ex_lp = ex_cam_l1 + ((l * num_pts_padded + p) << 1);
                    const uint32_t cam_base = c * level_h[l] * level_w[l];
                    for (uint32_t k = 0; k < 4; ++k) {
                        const int32_t x = x0 + static_cast<int32_t>(k & 1);
                        const int32_t y = y0 + static_cast<int32_t>(k >> 1);
                        if (x < 0 || x >= w || y < 0 || y >= h || corner_w[k] == 0) {
                            continue;
                        }
                        const uint32_t stick =
                            cam_base + static_cast<uint32_t>(y) * level_w[l] + static_cast<uint32_t>(x);
                        const uint32_t dst = s * stick_nbytes;
                        switch (l) {
                            case 0:
                                noc.async_read(v0, v_dfb, stick_nbytes, {.page_id = stick}, {.offset_bytes = dst});
                                break;
                            case 1:
                                noc.async_read(v1, v_dfb, stick_nbytes, {.page_id = stick}, {.offset_bytes = dst});
                                break;
                            case 2:
                                noc.async_read(v2, v_dfb, stick_nbytes, {.page_id = stick}, {.offset_bytes = dst});
                                break;
                            default:
                                noc.async_read(v3, v_dfb, stick_nbytes, {.page_id = stick}, {.offset_bytes = dst});
                                break;
                        }
                        for (uint32_t g = 0; g < num_groups; ++g) {
                            const uint16_t e = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(
                                ex_lp + g * (num_levels * num_pts_padded << 1));
                            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(et_l1 + tile_offset(g, s)) = e;
                        }
                        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w_l1 + tile_offset(0, s)) =
                            q16_weight_to_bf16(corner_w[k]);

                        if (++s == TILE_ROWS) {
                            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1 + LAST_FLAG_OFFSET) = 0;
                            noc.async_read_barrier();
                            v_dfb.push_back(c_tiles);
                            et_dfb.push_back(1);
                            w_dfb.push_back(1);
                            v_dfb.reserve_back(c_tiles);
                            et_dfb.reserve_back(1);
                            w_dfb.reserve_back(1);
                            et_l1 = et_dfb.get_write_ptr();
                            w_l1 = w_dfb.get_write_ptr();
                            s = 0;
                        }
                    }
                }
            }
        }

        // Close the anchor with a (possibly empty) last tile.
        clear_columns(et_l1, w_l1, s);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_l1 + LAST_FLAG_OFFSET) = 1;
        noc.async_read_barrier();
        v_dfb.push_back(c_tiles);
        et_dfb.push_back(1);
        w_dfb.push_back(1);
    }
}
