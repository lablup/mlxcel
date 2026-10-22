// Ported from mlx-vlm abd3b2d7, models/qwen4_exp/qsa_kernel.py.
// MIT License
//
// Copyright © 2025 Prince Canuma
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

static constexpr const char* QWEN4_QSA_SOURCE = R"QSA(
    uint row_idx = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = D_SIZE / BD;
    constexpr int v_per_thread = D_SIZE / BD;

    typedef float U;
    thread U q[qk_per_thread];
    thread U o[v_per_thread];
    threadgroup U outputs[BN * BD];
    threadgroup U max_scores[BN];
    threadgroup U sum_exp_scores[BN];

    int key_length = int(k_size[0]);
    int query_idx = int(row_idx % Q_LEN);
    int batch_head_idx = int(row_idx / Q_LEN);
    int batch_idx = batch_head_idx / NUM_Q_HEADS;
    int q_head_idx = batch_head_idx - batch_idx * NUM_Q_HEADS;
    int kv_head_idx = q_head_idx / GQA_FACTOR;
    int query_end = int(query_ends[batch_idx * Q_LEN + query_idx]);

    const device T* qptr =
        queries + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_lid) * qk_per_thread;
    device T* optr =
        out + ((batch_head_idx * Q_LEN + query_idx) * D_SIZE) +
        int(simd_gid) * v_per_thread;

    U s = U(scale[0]);
    for (int i = 0; i < qk_per_thread; ++i) {
        q[i] = s * static_cast<U>(qptr[i]);
    }
    for (int i = 0; i < v_per_thread; ++i) {
        o[i] = 0;
    }

    int blocks_offset = (batch_idx * Q_LEN + query_idx) * TOPK_BLOCKS;
    U max_score = -3.4028234663852886e38f;
    U sum_exp_score = 0;

    // The selected blocks are sorted by token position before launch. Each
    // SIMD group owns an interleaved subset and the final reduction combines
    // their independent online-softmax states.
    for (int selected_idx = int(simd_gid); selected_idx < SELECTED_LENGTH;
         selected_idx += BN) {
        int block_slot = selected_idx / BLOCK_SIZE;
        int block_offset = selected_idx - block_slot * BLOCK_SIZE;
        int block_idx = int(block_indices[blocks_offset + block_slot]);
        int key_pos = block_idx * BLOCK_SIZE + block_offset;
        bool valid = block_idx >= 0 && key_pos < key_length && key_pos < query_end;

        U score = -3.4028234663852886e38f;
        if (valid) {
            const device T* kptr =
                keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                         key_pos) * D_SIZE) +
                int(simd_lid) * qk_per_thread;
            score = 0;
            for (int j = 0; j < qk_per_thread; ++j) {
                score += q[j] * static_cast<U>(kptr[j]);
            }
            score = simd_sum(score);
        }

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = valid ? fast::exp(score - new_max) : U(0);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        if (valid) {
            const device T* vptr =
                values + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                           key_pos) * D_SIZE) +
                int(simd_lid) * v_per_thread;
            for (int j = 0; j < v_per_thread; ++j) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
            }
        } else {
            for (int j = 0; j < v_per_thread; ++j) {
                o[j] *= factor;
            }
        }
    }

    // QSA also attends the incomplete block immediately after the selected
    // complete blocks. It contains at most BLOCK_SIZE - 1 tokens.
    int tail_start = (query_end / BLOCK_SIZE) * BLOCK_SIZE;
    int tail_pos = tail_start + int(simd_gid);
    bool valid_tail = tail_pos < query_end && tail_pos < key_length;
    if (valid_tail) {
        const device T* kptr =
            keys + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                     tail_pos) * D_SIZE) +
            int(simd_lid) * qk_per_thread;
        U score = 0;
        for (int j = 0; j < qk_per_thread; ++j) {
            score += q[j] * static_cast<U>(kptr[j]);
        }
        score = simd_sum(score);

        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        const device T* vptr =
            values + (((batch_idx * NUM_KV_HEADS + kv_head_idx) * key_length +
                       tail_pos) * D_SIZE) +
            int(simd_lid) * v_per_thread;
        for (int j = 0; j < v_per_thread; ++j) {
            o[j] = o[j] * factor + exp_score * static_cast<U>(vptr[j]);
        }
    }

    if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    U new_max = simd_max(max_score);
    U factor = fast::exp(max_score - new_max);
    U total_sum = simd_sum(sum_exp_scores[simd_lid] * factor);

    for (int i = 0; i < v_per_thread; ++i) {
        outputs[simd_lid * BD + simd_gid] = o[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
        o[i] = total_sum == 0 ? U(0) : (o[i] / total_sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; ++i) {
            optr[i] = static_cast<T>(o[i]);
        }
    }
)QSA";

// Half-split MRoPE, from mlx-vlm rope_utils.py::_mrope_apply_kernel.
static constexpr const char* QWEN4_MROPE_SOURCE = R"MROPE(
    uint elem = thread_position_in_grid.x;
    constexpr int half_dim = ROTARY_DIM / 2;
    const int q_bsz = x_shape[0];
    const int q_heads = x_shape[1];
    const int q_len = x_shape[2];
    const int q_dim = x_shape[3];
    const int slots = half_dim + q_dim - ROTARY_DIM;
    const int work_size = q_bsz * q_heads * q_len * slots;
    if (elem >= uint(work_size)) return;
    int local = int(elem);
    int slot = local % slots;
    int tmp = local / slots;
    int t = tmp % q_len;
    tmp = tmp / q_len;
    int h = tmp % q_heads;
    int b = tmp / q_heads;
    int base = ((b * q_heads + h) * q_len + t) * q_dim;
    if (slot >= half_dim) {
        int pass_d = ROTARY_DIM + slot - half_dim;
        int pass_idx = base + pass_d;
        x_out[pass_idx] = x[pass_idx];
        return;
    }
    int freq_idx = slot;
    int d = freq_idx;
    int pair_d = d + half_dim;
    int axis = int(position_selector[freq_idx]);
    float pos = static_cast<float>(position_ids[(axis * q_bsz + b) * q_len + t]);
    float angle = pos * static_cast<float>(inv_freq[freq_idx]);
    float c = metal::cos(angle);
    float s = metal::sin(angle);
    int idx = base + d;
    float xv = static_cast<float>(x[idx]);
    float xp = static_cast<float>(x[base + pair_d]);
    x_out[idx] = static_cast<T>(xv * c - xp * s);
    x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);
)MROPE";
