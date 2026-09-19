#include "freetoken_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int apply_freetoken_compression(
    const float* activations,
    int seq_len,
    int hidden_dim,
    float threshold,
    float* out_compressed_act,
    FreeTokenMeta* out_meta
) {
    if (seq_len <= 1) {
        memcpy(out_compressed_act, activations, seq_len * hidden_dim * sizeof(float));
        out_meta[0].orig_idx = 0;
        out_meta[0].importance = 1.0f;
        out_meta[0].is_active = true;
        return seq_len;
    }

    float max_norm = 1e-6f;
    for (int t = 0; t < seq_len; t++) {
        const float* vec = &activations[t * hidden_dim];
        float sum_sq = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            sum_sq += vec[d] * vec[d];
        }
        float norm = sqrtf(sum_sq);
        out_meta[t].orig_idx = t;
        out_meta[t].importance = norm;
        out_meta[t].is_active = false;
        if (norm > max_norm) {
            max_norm = norm;
        }
    }

    int new_seq_len = 0;
    for (int t = 0; t < seq_len; t++) {
        float rel = out_meta[t].importance / max_norm;
        // Anchor Token (t==0) e Recency Token (t==seq_len-1) sono sempre preservati
        bool keep = (t == 0) || (t == seq_len - 1) || (rel >= threshold);

        if (keep) {
            out_meta[t].is_active = true;
            memcpy(&out_compressed_act[new_seq_len * hidden_dim],
                   &activations[t * hidden_dim],
                   hidden_dim * sizeof(float));
            out_meta[new_seq_len] = out_meta[t];
            new_seq_len++;
        }
    }

    return new_seq_len;
}

int load_moe_expert_via_mmap_freetoken(
    FreeTokenModelDisk* disk_ctx,
    const float* compressed_act,
    int comp_seq_len,
    int hidden_dim,
    const float* router_weights
) {
    for (int e = 0; e < disk_ctx->num_experts; e++) {
        disk_ctx->experts[e].ref_count = 0;
    }

    // Top-K routing calcolato esclusivamente sulla sequenza potata
    for (int t = 0; t < comp_seq_len; t++) {
        const float* x = &compressed_act[t * hidden_dim];
        float logits[FREETOKEN_MAX_EXPERTS];

        for (int e = 0; e < disk_ctx->num_experts; e++) {
            float dot = 0.0f;
            for (int d = 0; d < hidden_dim; d++) {
                dot += x[d] * router_weights[d * disk_ctx->num_experts + e];
            }
            logits[e] = dot;
        }

        int top1 = 0, top2 = 1;
        if (logits[top2] > logits[top1]) { top1 = 1; top2 = 0; }
        for (int e = 2; e < disk_ctx->num_experts; e++) {
            if (logits[e] > logits[top1]) {
                top2 = top1;
                top1 = e;
            } else if (logits[e] > logits[top2]) {
                top2 = e;
            }
        }

        disk_ctx->experts[top1].ref_count++;
        disk_ctx->experts[top2].ref_count++;
    }

    int active_experts = 0;
    for (int e = 0; e < disk_ctx->num_experts; e++) {
        if (disk_ctx->experts[e].ref_count > 0) {
            uint8_t* base = (uint8_t*)disk_ctx->mmap_base;
            disk_ctx->experts[e].weights_ptr = (float*)(base + disk_ctx->experts[e].byte_offset);
            
            #ifdef MADV_WILLNEED
            if (disk_ctx->experts[e].weights_ptr) {
                madvise(disk_ctx->experts[e].weights_ptr, disk_ctx->experts[e].byte_size, MADV_WILLNEED);
            }
            #endif
            active_experts++;
        } else {
            #ifdef MADV_DONTNEED
            if (disk_ctx->experts[e].weights_ptr != NULL) {
                madvise(disk_ctx->experts[e].weights_ptr, disk_ctx->experts[e].byte_size, MADV_DONTNEED);
                disk_ctx->experts[e].weights_ptr = NULL;
            }
            #endif
        }
    }

    return active_experts;
}
