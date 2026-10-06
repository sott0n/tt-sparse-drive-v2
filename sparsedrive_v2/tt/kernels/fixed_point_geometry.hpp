// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Bilinear sampling geometry in Q16.16 integers for data-movement kernels.
// These cores have no FPU: float math compiles to software routines that dominate a
// per-point loop, so coordinates and weights stay in integers end to end.

#pragma once

#include <stdint.h>

namespace sdv2 {

constexpr int32_t Q16_SHIFT = 16;
constexpr int32_t Q16_ONE = 1 << Q16_SHIFT;
constexpr int32_t Q16_HALF = 1 << (Q16_SHIFT - 1);
// Grid coordinates saturate at +-2: a clamped out-of-range point still maps outside the
// image, and (g + 1) * size stays inside int32 for sizes up to 8192.
constexpr int32_t GRID_CLAMP_Q16 = 2 * Q16_ONE;

// fp32 bit pattern -> Q16.16, saturating at +-GRID_CLAMP_Q16.
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

// Grid coordinate in [-1, 1] (Q16.16) -> pixel coordinate (Q16.16), align_corners=False.
inline int32_t grid_to_pixel(int32_t g_q16, int32_t size) { return ((g_q16 + Q16_ONE) * size >> 1) - Q16_HALF; }

// Q16.16 weight in [0, 1] -> bf16 bits, truncating.
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

}  // namespace sdv2
