// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused deformable-aggregation compute.
//
// Per sample tile (32 samples of one anchor, see reader_dfa.cpp):
//   V   = tilize(V_RM)                        32 samples x C, as c_tiles tiles
//   A   = transpose(E * W (column broadcast))  row g = group-g weight x bilinear weight
//   ACC[t] += A @ V[t]   for each channel tile t
// Row g of ACC[t] is the group-g sum over the anchor's samples for channels 32t..32t+31;
// only row t (group t == channel tile t, 32 channels per group) is meaningful. ACC lives in
// L1 (packer L1 accumulation, fp32) across the anchor's sample tiles.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tilize.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/transpose.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

constexpr uint32_t cb_v_rm = get_compile_time_arg_val(0);
constexpr uint32_t cb_v = get_compile_time_arg_val(1);
constexpr uint32_t cb_et = get_compile_time_arg_val(2);
constexpr uint32_t cb_w = get_compile_time_arg_val(3);
constexpr uint32_t cb_a = get_compile_time_arg_val(4);
constexpr uint32_t cb_acc = get_compile_time_arg_val(5);
constexpr uint32_t c_tiles = get_compile_time_arg_val(6);
constexpr uint32_t cb_a_t = get_compile_time_arg_val(7);  // A before the transpose

constexpr uint32_t LAST_FLAG_WORD = 128;  // W tile, face 1, element 0 (see reader_dfa.cpp)

void kernel_main() {
    const uint32_t num_anchors = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_et, cb_w, cb_a);

    for (uint32_t a = 0; a < num_anchors; ++a) {
        cb_reserve_back(cb_acc, c_tiles);
        bool first = true;
        uint32_t last = 0;
        while (last == 0) {
            cb_wait_front(cb_v_rm, c_tiles);
            cb_reserve_back(cb_v, c_tiles);
            // The packer was last set up for the fp32 ACC; tilize_init does not undo that.
            pack_reconfig_data_format(cb_v);
            tilize_init(cb_v_rm, c_tiles, cb_v);
            tilize_block(cb_v_rm, c_tiles, cb_v);
            tilize_uninit(cb_v_rm, cb_v);
            cb_push_back(cb_v, c_tiles);
            cb_pop_front(cb_v_rm, c_tiles);

            cb_wait_front(cb_et, 1);
            cb_wait_front(cb_w, 1);
            last = read_tile_value(cb_w, 0, LAST_FLAG_WORD);
            cb_reserve_back(cb_a_t, 1);
            mul_bcast_cols_init(cb_et, cb_w);
            pack_reconfig_data_format(cb_a_t);
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_et, cb_w, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_a_t);
            tile_regs_release();
            cb_push_back(cb_a_t, 1);
            cb_pop_front(cb_et, 1);
            cb_pop_front(cb_w, 1);

            cb_wait_front(cb_a_t, 1);
            cb_reserve_back(cb_a, 1);
            transpose_init(cb_a_t);
            pack_reconfig_data_format(cb_a);
            tile_regs_acquire();
            transpose_tile(cb_a_t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_a);
            tile_regs_release();
            cb_push_back(cb_a, 1);
            cb_pop_front(cb_a_t, 1);

            cb_wait_front(cb_a, 1);
            cb_wait_front(cb_v, c_tiles);
            matmul_init(cb_a, cb_v);
            pack_reconfig_data_format(cb_acc);
            pack_reconfig_l1_acc(first ? 0 : 1);
            tile_regs_acquire();
            for (uint32_t t = 0; t < c_tiles; ++t) {
                matmul_tiles(cb_a, cb_v, 0, t, t);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < c_tiles; ++t) {
                // Explicit slots: L1 accumulation must land on the same ACC tile every time.
                pack_tile<true>(t, cb_acc, t);
            }
            tile_regs_release();
            pack_reconfig_l1_acc(0);
            cb_pop_front(cb_a, 1);
            cb_pop_front(cb_v, c_tiles);
            first = false;
        }
        cb_push_back(cb_acc, c_tiles);
    }
}
