#ifndef FREETOKEN_ENGINE_H
#define FREETOKEN_ENGINE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FREETOKEN_MAX_EXPERTS 256
#define FREETOKEN_TOP_K 2

typedef struct {
    int orig_idx;       // Indice originale per correlazione KV-Cache
    float importance;   // Punteggio normalizzato [0.0, 1.0]
    bool is_active;     // Flag di attivazione per il routing MoE
} FreeTokenMeta;

typedef struct {
    int expert_id;
    int ref_count;
    float* weights_ptr;
    size_t byte_offset;
    size_t byte_size;
} FreeTokenExpert;

typedef struct {
    int num_experts;
    size_t expert_size_bytes;
    int fd;
    void* mmap_base;
    FreeTokenExpert experts[FREETOKEN_MAX_EXPERTS];
} FreeTokenModelDisk;

/**
 * @brief Pruning & Merging dei token basato sulla varianza/L2-norm dell'attivazione.
 */
int apply_freetoken_compression(
    const float* activations,
    int seq_len,
    int hidden_dim,
    float threshold,
    float* out_compressed_act,
    FreeTokenMeta* out_meta
);

/**
 * @brief Selezione MoE Top-K ed esecuzione del page-in on-demand tramite mmap.
 */
int load_moe_expert_via_mmap_freetoken(
    FreeTokenModelDisk* disk_ctx,
    const float* compressed_act,
    int comp_seq_len,
    int hidden_dim,
    const float* router_weights
);

#ifdef __cplusplus
}
#endif

#endif // FREETOKEN_ENGINE_H
