#include "yue2.h"
#include "h3_gpu.h"
#include "h3_safetensors.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define W 2048
#define KV 1024
#define FF 6144
#define LAYERS 28
#define VOCAB 184704
#define ROWS 1024
#define CACHE 24576
#define MAX_FRAMES 9000
#define EOD 151643
#define ABC_START 151847
#define ABC_END 151848
#define MUSIC_START 151851
#define MUSIC_END 151852
#define CODEC 151853

typedef struct {
  h3_gpu_tensor *norm, *post, *q, *k, *v, *o, *qn, *kn, *gate, *up, *down;
} block;
struct yue2_model {
  h3_gpu *gpu;
  h3_st_header header, vae;
  h3_gpu_tensor *owned[1024];
  int owned_count;
  block ar[LAYERS], nar[LAYERS];
  h3_gpu_tensor *keys[LAYERS], *values[LAYERS];
  h3_gpu_tensor *embed, *head, *norm, *pos, *to_latent, *to_latent_bias,
      *from_latent, *from_latent_bias;
  h3_gpu_tensor *time1, *time1b, *time2, *time2b;
  h3_gpu_tensor *x, *n, *branch, *q, *k, *v, *attn, *gate, *up, *cos, *sin,
      *last, *logits;
  h3_gpu_tensor *latent, *time_input, *time_hidden, *time_output, *positions;
  float *rope_host;
  int ar_length;
  // Identity and extent of the valid AR prefix, independent of NAR scratch.
  uint32_t cached_ids[CACHE];
  int cached_count;
  int rows, cache;
  char package[4096];
};
static int fail(char *error, size_t size, const char *message) {
  if (error && size)
    snprintf(error, size, "%s", message);
  return 0;
}
static uint16_t bf(float x) {
  uint32_t n;
  memcpy(&n, &x, 4);
  n += 0x7fff + ((n >> 16) & 1);
  return n >> 16;
}
static float fp(uint16_t x) {
  uint32_t n = (uint32_t)x << 16;
  float f;
  memcpy(&f, &n, 4);
  return f;
}
static float roundbf(float x) { return fp(bf(x)); }
static h3_gpu_tensor *own(yue2_model *m, h3_gpu_tensor *t) {
  if (t && m->owned_count < 1024)
    m->owned[m->owned_count++] = t;
  else if (t) {
    h3_gpu_tensor_free(t);
    return NULL;
  }
  return t;
}
static h3_gpu_tensor *load(yue2_model *m, const char *name, int rank,
                           const uint64_t *shape, char *error, size_t size) {
  const h3_st_tensor *t = h3_st_find(&m->header, name);
  if (!t || t->dtype != H3_DTYPE_BF16 || t->ndim != rank ||
      memcmp(t->shape, shape, rank * 8)) {
    snprintf(error, size, "Unexpected YuE2 tensor: %s", name);
    return NULL;
  }
  return own(m, h3_gpu_tensor_load_bf16(m->gpu, m->header.path, t->file_offset,
                                        h3_st_tensor_elements(t)));
}
#define LOAD(dst, name, rank, ...)                                             \
  do {                                                                         \
    const uint64_t dims[] = {__VA_ARGS__};                                     \
    m->dst = load(m, name, rank, dims, error, size);                           \
    if (!m->dst)                                                               \
      goto bad;                                                                \
  } while (0)
static int load_block(yue2_model *m, block *b, int layer, int nar, char *error,
                      size_t size) {
  char name[160];
  const char *att = nar ? "nar_self_attn" : "self_attn",
             *mlp = nar ? "nar_mlp" : "mlp";
#define B(dst, suffix, rank, ...)                                              \
  do {                                                                         \
    const uint64_t dims[] = {__VA_ARGS__};                                     \
    snprintf(name, sizeof(name), "model.layers.%d.%s", layer, suffix);         \
    b->dst = load(m, name, rank, dims, error, size);                           \
    if (!b->dst)                                                               \
      return 0;                                                                \
  } while (0)
  B(norm, nar ? "nar_input_layernorm.weight" : "input_layernorm.weight", 1, W);
  B(post,
    nar ? "nar_pre_mlp_layernorm.weight" : "post_attention_layernorm.weight", 1,
    W);
  char suffix[100];
#define A(dst, suffixname, ...)                                                \
  snprintf(suffix, sizeof(suffix), "%s.%s.weight", att, suffixname);           \
  B(dst, suffix, 2, __VA_ARGS__)
  A(q, "q_proj", W, W);
  A(k, "k_proj", KV, W);
  A(v, "v_proj", KV, W);
  A(o, "o_proj", W, W);
  snprintf(suffix, sizeof(suffix), "%s.q_norm.weight", att);
  B(qn, suffix, 1, 128);
  snprintf(suffix, sizeof(suffix), "%s.k_norm.weight", att);
  B(kn, suffix, 1, 128);
  snprintf(suffix, sizeof(suffix), "%s.gate_proj.weight", mlp);
  B(gate, suffix, 2, FF, W);
  snprintf(suffix, sizeof(suffix), "%s.up_proj.weight", mlp);
  B(up, suffix, 2, FF, W);
  snprintf(suffix, sizeof(suffix), "%s.down_proj.weight", mlp);
  B(down, suffix, 2, W, FF);
#undef A
#undef B
  return 1;
}
yue2_model *yue2_load(const char *package, const char *shaders, char *error,
                      size_t size) {
  if (error && size)
    error[0] = 0;
  if (!package || !shaders || !shaders[0]) {
    fail(error, size, "YuE2 requires its Metal shader library");
    return NULL;
  }
  yue2_model *m = calloc(1, sizeof(*m));
  if (!m)
    return NULL;
  m->rows = ROWS;
  m->cache = 7168;
  snprintf(m->package, sizeof(m->package), "%s", package);
  char path[4096];
  snprintf(path, sizeof(path), "%s/model/model.safetensors", package);
  if (!h3_st_read_header(path, &m->header, error, size))
    goto bad;
  snprintf(path, sizeof(path), "%s/vae/model.safetensors", package);
  if (!h3_st_read_header(path, &m->vae, error, size))
    goto bad;
  if (!h3_gpu_prepare(shaders, error, size))
    goto bad;
  m->gpu = h3_gpu_create(shaders, error, size);
  if (!m->gpu)
    goto bad;
  h3_gpu_set_linear_2d(m->gpu, getenv("H3_YUE2_DISABLE_LINEAR_2D") == NULL);
  LOAD(embed, "model.embed_tokens.weight", 2, VOCAB, W);
  LOAD(head, "lm_head.weight", 2, VOCAB, W);
  LOAD(norm, "model.norm.weight", 1, W);
  LOAD(pos, "latent_pos_embed.pe", 2, 24576, W);
  LOAD(to_latent, "llm2vae.weight", 2, 64, W);
  LOAD(to_latent_bias, "llm2vae.bias", 1, 64);
  LOAD(from_latent, "vae2llm.weight", 2, W, 64);
  LOAD(from_latent_bias, "vae2llm.bias", 1, W);
  LOAD(time1, "time_embedder.mlp.0.weight", 2, W, 256);
  LOAD(time1b, "time_embedder.mlp.0.bias", 1, W);
  LOAD(time2, "time_embedder.mlp.2.weight", 2, W, W);
  LOAD(time2b, "time_embedder.mlp.2.bias", 1, W);
  for (int i = 0; i < LAYERS; i++) {
    if (!load_block(m, &m->ar[i], i, 0, error, size) ||
        !load_block(m, &m->nar[i], i, 1, error, size))
      goto bad;
    m->keys[i] = own(m, h3_gpu_tensor_new_bf16(m->gpu, (size_t)m->cache * KV));
    m->values[i] =
        own(m, h3_gpu_tensor_new_bf16(m->gpu, (size_t)m->cache * KV));
    if (!m->keys[i] || !m->values[i])
      goto bad;
  }
#define BUF(name, count)                                                       \
  m->name = own(m, h3_gpu_tensor_new_bf16(m->gpu, count));                     \
  if (!m->name)                                                                \
  goto bad
  BUF(x, ROWS * W);
  BUF(n, ROWS * W);
  BUF(branch, ROWS * W);
  BUF(q, ROWS * W);
  BUF(k, ROWS * KV);
  BUF(v, ROWS * KV);
  BUF(attn, ROWS * W);
  BUF(gate, ROWS * FF);
  BUF(up, ROWS * FF);
  BUF(last, W);
  BUF(logits, VOCAB);
  BUF(latent, ROWS * 64);
  BUF(time_input, ROWS * 256);
  BUF(time_hidden, ROWS * W);
  BUF(time_output, ROWS * W);
  BUF(positions, ROWS * W);
#undef BUF
  m->cos = own(m, h3_gpu_tensor_new_f32(m->gpu, ROWS * 64));
  m->sin = own(m, h3_gpu_tensor_new_f32(m->gpu, ROWS * 64));
  m->rope_host = malloc(ROWS * 128 * sizeof(float));
  if (!m->cos || !m->sin || !m->rope_host)
    goto bad;
  return m;
bad:
  if (error && size && !error[0])
    snprintf(error, size, "YuE2 allocation or Metal initialization failed");
  yue2_free(m);
  return NULL;
}
#undef LOAD
void yue2_free(yue2_model *m) {
  if (!m)
    return;
  for (int i = 0; i < m->owned_count; i++)
    h3_gpu_tensor_free(m->owned[i]);
  h3_gpu_free(m->gpu);
  h3_st_free_header(&m->header);
  h3_st_free_header(&m->vae);
  free(m->rope_host);
  free(m);
}
/* Grow scratch and KV storage only for requests which need it. Replace each
 * owned entry in place so partial allocation failures remain safely freeable.
 */
static int replace_buffer(yue2_model *m, h3_gpu_tensor **slot, size_t count,
                          int f32) {
  h3_gpu_tensor *next = f32 ? h3_gpu_tensor_new_f32(m->gpu, count)
                            : h3_gpu_tensor_new_bf16(m->gpu, count);
  if (!next)
    return 0;
  for (int i = 0; i < m->owned_count; i++) {
    if (m->owned[i] == *slot) {
      h3_gpu_tensor_free(*slot);
      m->owned[i] = *slot = next;
      return 1;
    }
  }
  h3_gpu_tensor_free(next);
  return 0;
}
static int reserve(yue2_model *m, int rows, int cache, char *error,
                   size_t size) {
  if (rows > MAX_FRAMES + 2 || cache > CACHE)
    return fail(error, size, "YuE2 context exceeded");
#define GROW(name, width, f32)                                                 \
  if (!replace_buffer(m, &m->name, (size_t)rows * (width), f32))               \
  goto bad
  if (rows > m->rows) {
    GROW(x, W, 0);
    GROW(n, W, 0);
    GROW(branch, W, 0);
    GROW(q, W, 0);
    GROW(k, KV, 0);
    GROW(v, KV, 0);
    GROW(attn, W, 0);
    GROW(gate, FF, 0);
    GROW(up, FF, 0);
    GROW(latent, 64, 0);
    GROW(time_input, 256, 0);
    GROW(time_hidden, W, 0);
    GROW(time_output, W, 0);
    GROW(positions, W, 0);
    GROW(cos, 64, 1);
    GROW(sin, 64, 1);
    float *host = realloc(m->rope_host, (size_t)rows * 128 * sizeof(float));
    if (!host)
      goto bad;
    m->rope_host = host;
    m->rows = rows;
  }
#undef GROW
  if (cache > m->cache) {
    m->cached_count = 0; // Replacing KV storage invalidates its prefix.
    for (int i = 0; i < LAYERS; i++) {
      if (!replace_buffer(m, &m->keys[i], (size_t)cache * KV, 0) ||
          !replace_buffer(m, &m->values[i], (size_t)cache * KV, 0))
        goto bad;
    }
    m->cache = cache;
  }
  return 1;
bad:
  return fail(error, size, "Unable to allocate YuE2 context buffers");
}
#define GPU(call)                                                              \
  do {                                                                         \
    if (!(call)) {                                                             \
      snprintf(error, size, "YuE2 Metal: %s", h3_gpu_error(m->gpu));           \
      return 0;                                                                \
    }                                                                          \
  } while (0)
static int rope(yue2_model *m, int rows, int base) {
  float *c = m->rope_host, *s = c + m->rows * 64;
  for (int r = 0; r < rows; r++)
    for (int j = 0; j < 64; j++) {
      float angle =
          (float)(base + r) * (1.0f / powf(1000000.0f, (float)(2 * j) / 128));
      /* Reference RoPE casts the table to BF16 before multiplication. */
      c[r * 64 + j] = roundbf(cosf(angle));
      s[r * 64 + j] = roundbf(sinf(angle));
    }
  return h3_gpu_tensor_write_f32(m->cos, c, rows * 64) &&
         h3_gpu_tensor_write_f32(m->sin, s, rows * 64);
}
static int forward_block(yue2_model *m, int i, int rows, int base, int nar,
                         char *error, size_t size) {
  h3_gpu *g = m->gpu;
  block *b = nar ? &m->nar[i] : &m->ar[i];
  GPU(h3_gpu_rms_norm_bf16(g, m->n, m->x, b->norm, rows, W, 1e-6));
  GPU(h3_gpu_linear_bf16(g, m->q, m->n, b->q, NULL, rows, W, W));
  GPU(h3_gpu_linear_bf16(g, m->k, m->n, b->k, NULL, rows, W, KV));
  GPU(h3_gpu_linear_bf16(g, m->v, m->n, b->v, NULL, rows, W, KV));
  GPU(h3_gpu_head_rms_norm_coop_bf16(g, m->q, b->qn, rows, 16, 128, 1e-6));
  GPU(h3_gpu_head_rms_norm_coop_bf16(g, m->k, b->kn, rows, 8, 128, 1e-6));
  GPU(h3_gpu_rope_text_bf16(g, m->q, m->k, m->cos, m->sin, rows, 16, 8, 128,
                            0));
  GPU(h3_gpu_copy_bf16(g, m->keys[i], (size_t)base * KV, m->k, 0,
                       (size_t)rows * KV));
  GPU(h3_gpu_copy_bf16(g, m->values[i], (size_t)base * KV, m->v, 0,
                       (size_t)rows * KV));
  if (nar)
    GPU(h3_gpu_gqa_cross_bf16(g, m->attn, m->q, m->keys[i], m->values[i], rows,
                              base, 16, 8, 128, 1.0f / sqrtf(128)));
  else
    GPU(h3_gpu_gqa_causal_cache_bf16(g, m->attn, m->q, m->keys[i], m->values[i],
                                     rows, base, 16, 8, 128,
                                     1.0f / sqrtf(128)));
  GPU(h3_gpu_linear_bf16(g, m->branch, m->attn, b->o, NULL, rows, W, W));
  GPU(h3_gpu_add_bf16(g, m->x, m->x, m->branch, rows * W));
  GPU(h3_gpu_rms_norm_bf16(g, m->n, m->x, b->post, rows, W, 1e-6));
  GPU(h3_gpu_linear_bf16(g, m->gate, m->n, b->gate, NULL, rows, W, FF));
  GPU(h3_gpu_linear_bf16(g, m->up, m->n, b->up, NULL, rows, W, FF));
  GPU(h3_gpu_silu_mul_bf16(g, m->gate, m->gate, m->up, rows * FF));
  GPU(h3_gpu_linear_bf16(g, m->branch, m->gate, b->down, NULL, rows, FF, W));
  GPU(h3_gpu_add_bf16(g, m->x, m->x, m->branch, rows * W));
  return 1;
}
/* layer_limit supports block-level parity. A full pass returns final-position
 * logits; a partial one returns every hidden row before final normalization. */
int yue2_forward(yue2_model *m, const uint32_t *ids, int count, int base,
                 int layer_limit, float *out, char *error, size_t size) {
  if (count < 1 || count > m->rows || base < 0 || base + count > m->cache ||
      layer_limit < 1 || layer_limit > LAYERS)
    return fail(error, size, "YuE2 context capacity exceeded");
  for (int i = 0; i < count; i++)
    if (ids[i] >= VOCAB)
      return fail(error, size, "Invalid YuE2 token");
  GPU(rope(m, count, base));
  h3_gpu_tensor *tokens = h3_gpu_tensor_from_u32(m->gpu, ids, count);
  if (!tokens)
    return fail(error, size, "Token allocation failed");
  int ok =
      h3_gpu_begin(m->gpu) &&
      h3_gpu_embedding_bf16(m->gpu, m->x, m->embed, tokens, count, VOCAB, W);
  for (int i = 0; ok && i < layer_limit; i++)
    ok = forward_block(m, i, count, base, 0, error, size);
  if (ok && out && layer_limit == LAYERS) {
    ok =
        h3_gpu_copy_bf16(m->gpu, m->last, 0, m->x, (size_t)(count - 1) * W,
                         W) &&
        h3_gpu_rms_norm_bf16(m->gpu, m->n, m->last, m->norm, 1, W, 1e-6) &&
        h3_gpu_linear_bf16(m->gpu, m->logits, m->n, m->head, NULL, 1, W, VOCAB);
  }
  int submitted = h3_gpu_submit(m->gpu);
  h3_gpu_tensor_free(tokens);
  if (!ok || !submitted) {
    m->cached_count = 0; // A failed pass may have overwritten only some layers.
    return fail(error, size, h3_gpu_error(m->gpu));
  }
  if (layer_limit == LAYERS && base <= m->cached_count) {
    memcpy(m->cached_ids + base, ids, count * sizeof(uint32_t));
    m->cached_count = base + count;
  } else {
    m->cached_count = 0;
  }
  if (out) {
    int n = layer_limit == LAYERS ? VOCAB : count * W;
    uint16_t *v = malloc(n * 2);
    if (!v)
      return 0;
    ok =
        h3_gpu_tensor_read_bf16(layer_limit == LAYERS ? m->logits : m->x, v, n);
    for (int i = 0; i < n; i++)
      out[i] = fp(v[i]);
    free(v);
  }
  return ok;
}
static int prefill(yue2_model *m, const uint32_t *ids, int count, float *logits,
                   yue2_progress cb, void *ctx, const char *phase, char *error,
                   size_t size) {
  int shared = 0;
  while (shared < count && shared < m->cached_count &&
         ids[shared] == m->cached_ids[shared]) shared++;
  // A logits request needs the last position evaluated. A conditioning-only
  // prefill can reuse the entire matching prefix. No sampled token is skipped.
  int start = logits && shared == count ? count - 1 : shared;
  m->cached_count = start;
  if (cb && !cb(phase, start, count, ctx)) return fail(error, size, "Cancelled");
  for (int base = start; base < count; base += 128) {
    if (cb && !cb(phase, base, count, ctx))
      return fail(error, size, "Cancelled");
    int n = count - base < 128 ? count - base : 128;
    if (!yue2_forward(m, ids + base, n, base, LAYERS,
                      base + n == count ? logits : NULL, error, size))
      return 0;
  }
  return 1;
}
static int velocity(yue2_model *m, const float *state, int frames, float time,
                    float *out, yue2_progress cb, void *ctx, int completed,
                    int total, char *error, size_t size) {
  int rows = frames + 2;
  if (frames < 1 || rows > m->rows || m->ar_length + rows > m->cache)
    return fail(error, size, "YuE2 acoustic context capacity exceeded");
  uint16_t *latent = calloc(rows * 64, 2), *t = malloc(rows * 256 * 2);
  if (!latent || !t) {
    free(latent);
    free(t);
    return 0;
  }
  for (int i = 0; i < frames * 64; i++)
    latent[64 + i] = bf(state[i]);
  /* t is a probability in (0,1); emulate the reference's BF16 logit/sigmoid. */
  float raw = roundbf(fmaxf(-20, fminf(20, logf(time / (1 - time)))));
  float shifted = roundbf(1 / (1 + expf(-raw)));
  for (int r = 0; r < rows; r++)
    for (int j = 0; j < 128; j++) {
      float a = shifted * expf(-logf(10000) * (float)j / 128);
      t[r * 256 + j] = bf(cosf(a));
      t[r * 256 + 128 + j] = bf(sinf(a));
    }
  int ok = h3_gpu_tensor_write_bf16(m->latent, latent, rows * 64) &&
           h3_gpu_tensor_write_bf16(m->time_input, t, rows * 256) &&
           rope(m, rows, m->ar_length);
  free(latent);
  free(t);
  if (!ok)
    return 0;
  h3_gpu *g = m->gpu;
  GPU(h3_gpu_begin(g));
  GPU(h3_gpu_linear_bf16(g, m->time_hidden, m->time_input, m->time1, m->time1b,
                         rows, 256, W));
  GPU(h3_gpu_silu_bf16(g, m->time_hidden, m->time_hidden, rows * W));
  GPU(h3_gpu_linear_bf16(g, m->time_output, m->time_hidden, m->time2, m->time2b,
                         rows, W, W));
  GPU(h3_gpu_linear_bf16(g, m->x, m->latent, m->from_latent,
                         m->from_latent_bias, rows, 64, W));
  GPU(h3_gpu_add_bf16(g, m->x, m->x, m->time_output, rows * W));
  GPU(h3_gpu_copy_bf16(g, m->positions, 0, m->pos, 0, rows * W));
  GPU(h3_gpu_add_bf16(g, m->x, m->x, m->positions, rows * W));
  for (int i = 0; i < LAYERS; i++) {
    if (!forward_block(m, i, rows, m->ar_length, 1, error, size)) {
      h3_gpu_submit(g);
      return 0;
    }
    if (cb) {
      // Report completed GPU work, not enqueued work. A single full-song
      // velocity can take minutes; bound cancellation latency to one layer.
      GPU(h3_gpu_submit(g));
      if (!cb("Synthesizing audio", completed + i + 1, total, ctx))
        return fail(error, size, "Cancelled");
      GPU(h3_gpu_begin(g));
    }
  }
  GPU(h3_gpu_rms_norm_bf16(g, m->n, m->x, m->norm, rows, W, 1e-6));
  GPU(h3_gpu_linear_bf16(g, m->latent, m->n, m->to_latent, m->to_latent_bias,
                         rows, W, 64));
  GPU(h3_gpu_submit(g));
  uint16_t *values = malloc(rows * 64 * 2);
  if (!values)
    return 0;
  ok = h3_gpu_tensor_read_bf16(m->latent, values, rows * 64);
  for (int i = 0; i < frames * 64; i++)
    out[i] = fp(values[i + 64]);
  free(values);
  return ok;
}
int yue2_velocity(yue2_model *m, const float *state, int frames, float time,
                  float *out, char *error, size_t size) {
  return velocity(m, state, frames, time, out, NULL, NULL, 0, 0, error, size);
}
int yue2_synthesize(yue2_model *m, const uint32_t *prefix, int prefix_count,
                    const uint32_t *codec, int frames, const float *noise,
                    int steps, float *out, yue2_progress cb, void *ctx,
                    char *error, size_t size) {
  if (frames < 1 || frames > MAX_FRAMES || prefix_count < 1 ||
      prefix_count + frames * 2 + 3 > CACHE ||
      !(steps == 8 || steps == 16 || steps == 32))
    return fail(error, size,
                "YuE2 synthesis requires 1–9000 frames, 8/16/32 steps and a "
                "context at most 24576 tokens");
  if (!reserve(m, frames + 2, prefix_count + 2 * frames + 3, error, size))
    return 0;
  int count = prefix_count + frames + 1;
  uint32_t *ids = malloc(count * 4);
  if (!ids)
    return 0;
  memcpy(ids, prefix, prefix_count * 4);
  for (int i = 0; i < frames; i++) {
    if (codec[i] >= 32768) {
      free(ids);
      return fail(error, size, "Invalid music token");
    }
    ids[prefix_count + i] = CODEC + codec[i];
  }
  ids[count - 1] = MUSIC_END;
  int ok = prefill(m, ids, count, NULL, cb, ctx, "Preparing audio context",
                   error, size);
  free(ids);
  if (!ok)
    return 0;
  m->ar_length = count;
  float *v = malloc(frames * 64 * 4), *mid = malloc(frames * 64 * 4);
  if (!v || !mid) {
    free(v);
    free(mid);
    return 0;
  }
  for (int i = 0; i < frames * 64; i++)
    out[i] = roundbf(noise[i]);
  float dt = 1.0f / steps;
  for (int step = 0; step < steps; step++) {
    if (cb &&
        !cb("Synthesizing audio", step * 2 * LAYERS, steps * 2 * LAYERS, ctx)) {
      ok = fail(error, size, "Cancelled");
      break;
    }
    float time = 1.0f - step * dt;
    if (!(ok = velocity(m, out, frames, time, v, cb, ctx, step * 2 * LAYERS,
                        steps * 2 * LAYERS, error, size)))
      break;
    for (int i = 0; i < frames * 64; i++)
      mid[i] = roundbf(out[i] - roundbf(v[i] * (dt / 2)));
    if (cb && !cb("Synthesizing audio", (step * 2 + 1) * LAYERS,
                  steps * 2 * LAYERS, ctx)) {
      ok = fail(error, size, "Cancelled");
      break;
    }
    if (!(ok = velocity(m, mid, frames, time - dt / 2, v, cb, ctx,
                        (step * 2 + 1) * LAYERS, steps * 2 * LAYERS, error,
                        size)))
      break;
    for (int i = 0; i < frames * 64; i++)
      out[i] = roundbf(out[i] - roundbf(v[i] * dt));
  }
  free(v);
  free(mid);
  return ok;
}

typedef struct {
  h3_gpu_tensor *w, *b;
  int in, out, kernel, stride, pad, dilation, transpose;
} convolution;
typedef struct {
  h3_gpu_tensor *a, *b;
} snake;
typedef struct {
  snake s1, s2;
  convolution c1, c2;
} residual;
typedef struct {
  snake s;
  convolution up;
  residual r[3];
} stage;
typedef struct {
  convolution first, last;
  stage stages[6];
  snake final;
} decoder;
static h3_gpu_tensor *vae_tensor(yue2_model *m, const char *name, size_t n,
                                 char *error, size_t size) {
  const h3_st_tensor *t = h3_st_find(&m->vae, name);
  if (!t || t->dtype != H3_DTYPE_F32 || h3_st_tensor_elements(t) != n) {
    snprintf(error, size, "Unexpected YuE2 VAE tensor %s", name);
    return NULL;
  }
  return own(m, h3_gpu_tensor_load_f32(m->gpu, m->vae.path, t->file_offset, n));
}
static int load_snake(yue2_model *m, snake *s, const char *prefix, int channels,
                      char *error, size_t size) {
  char name[200];
  snprintf(name, sizeof(name), "%s.alpha", prefix);
  s->a = vae_tensor(m, name, channels, error, size);
  snprintf(name, sizeof(name), "%s.beta", prefix);
  s->b = vae_tensor(m, name, channels, error, size);
  return s->a && s->b;
}
static int load_conv(yue2_model *m, convolution *c, const char *prefix, int in,
                     int out, int kernel, int stride, int pad, int dilation,
                     int transpose, int bias, char *error, size_t size) {
  *c = (convolution){.in = in,
                     .out = out,
                     .kernel = kernel,
                     .stride = stride,
                     .pad = pad,
                     .dilation = dilation,
                     .transpose = transpose};
  char name[200];
  snprintf(name, sizeof(name), "%s.weight_v", prefix);
  size_t n = (size_t)in * out * kernel;
  h3_gpu_tensor *v = vae_tensor(m, name, n, error, size);
  snprintf(name, sizeof(name), "%s.weight_g", prefix);
  int outer = transpose ? in : out;
  h3_gpu_tensor *g = vae_tensor(m, name, outer, error, size);
  if (!v || !g)
    return 0;
  c->w = own(m, h3_gpu_tensor_new_f32(m->gpu, n));
  if (!c->w)
    return 0;
  if (!h3_gpu_begin(m->gpu) ||
      !h3_gpu_weight_norm_f32(m->gpu, c->w, v, g, outer, n / outer) ||
      !h3_gpu_submit(m->gpu))
    return 0;
  if (bias) {
    snprintf(name, sizeof(name), "%s.bias", prefix);
    c->b = vae_tensor(m, name, out, error, size);
    if (!c->b)
      return 0;
  }
  return 1;
}
static int decoder_load(yue2_model *m, decoder *d, char *error, size_t size) {
  if (!load_conv(m, &d->first, "decoder.layers.0", 64, 2048, 7, 1, 3, 1, 0, 1,
                 error, size))
    return 0;
  int strides[] = {6, 5, 4, 4, 2, 2},
      channels[] = {2048, 1024, 512, 256, 128, 64, 64};
  char name[200];
  for (int i = 0; i < 6; i++) {
    stage *s = &d->stages[i];
    int in = channels[i], out = channels[i + 1], stride = strides[i];
    snprintf(name, sizeof(name), "decoder.layers.%d.layers.0", i + 1);
    if (!load_snake(m, &s->s, name, in, error, size))
      return 0;
    snprintf(name, sizeof(name), "decoder.layers.%d.layers.1", i + 1);
    if (!load_conv(m, &s->up, name, in, out, 2 * stride, stride,
                   (stride + 1) / 2, 1, 1, 1, error, size))
      return 0;
    for (int j = 0; j < 3; j++) {
      residual *r = &s->r[j];
      int dilation = j == 0 ? 1 : j == 1 ? 3 : 9;
      snprintf(name, sizeof(name), "decoder.layers.%d.layers.%d.layers.0",
               i + 1, j + 2);
      if (!load_snake(m, &r->s1, name, out, error, size))
        return 0;
      snprintf(name, sizeof(name), "decoder.layers.%d.layers.%d.layers.1",
               i + 1, j + 2);
      if (!load_conv(m, &r->c1, name, out, out, 7, 1, 3 * dilation, dilation, 0,
                     1, error, size))
        return 0;
      snprintf(name, sizeof(name), "decoder.layers.%d.layers.%d.layers.2",
               i + 1, j + 2);
      if (!load_snake(m, &r->s2, name, out, error, size))
        return 0;
      snprintf(name, sizeof(name), "decoder.layers.%d.layers.%d.layers.3",
               i + 1, j + 2);
      if (!load_conv(m, &r->c2, name, out, out, 1, 1, 0, 1, 0, 1, error, size))
        return 0;
    }
  }
  return load_snake(m, &d->final, "decoder.layers.7", 64, error, size) &&
         load_conv(m, &d->last, "decoder.layers.8", 64, 2, 7, 1, 3, 1, 0, 0,
                   error, size);
}
static int conv_run(yue2_model *m, convolution *c, h3_gpu_tensor *out,
                    h3_gpu_tensor *in, int length) {
  if (c->transpose)
    return h3_gpu_conv_transpose1d_f32(m->gpu, out, in, c->w, c->b, 1, length,
                                       c->in, c->out, c->kernel, c->stride,
                                       c->pad);
  return h3_gpu_conv1d_f32(m->gpu, out, in, c->w, c->b, 1, length, c->in,
                           c->out, c->kernel, c->pad, c->dilation);
}
static int snake_run(yue2_model *m, snake *s, h3_gpu_tensor *out,
                     h3_gpu_tensor *in, int length, int channels) {
  return h3_gpu_snake_beta_f32(m->gpu, out, in, s->a, s->b, 1, length,
                               channels);
}
int yue2_decode(yue2_model *m, const float *latent, int frames, float **audio,
                int *samples, yue2_progress cb, void *ctx, char *error,
                size_t size) {
  *audio = NULL;
  *samples = 0;
  if (frames < 1 || frames > MAX_FRAMES)
    return fail(error, size, "Invalid YuE2 decoder frame count");
  int owned_start = m->owned_count, ok = 0;
  decoder d = {0};
  h3_gpu_tensor *a = NULL, *b = NULL, *c = NULL, *skip = NULL;
  if (!decoder_load(m, &d, error, size))
    goto done;
  /* 128-frame cores plus the reference's 16-frame halo. Largest activation
   * occurs at the last 64-channel stage; one tile bounds GPU memory. */
  size_t capacity = (size_t)160 * 1920 * 64;
  a = h3_gpu_tensor_new_f32(m->gpu, capacity);
  b = h3_gpu_tensor_new_f32(m->gpu, capacity);
  c = h3_gpu_tensor_new_f32(m->gpu, capacity);
  skip = h3_gpu_tensor_new_f32(m->gpu, capacity);
  if (!a || !b || !c || !skip)
    goto done;
  *samples = frames * 1920 - 64;
  *audio = malloc((size_t)*samples * 2 * 4);
  if (!*audio)
    goto done;
  for (int start = 0; start < frames; start += 128) {
    if (cb && !cb("Decoding audio", start, frames, ctx)) {
      fail(error, size, "Cancelled");
      goto done;
    }
    int end = start + 128 < frames ? start + 128 : frames,
        left = start > 16 ? start - 16 : 0,
        right = end + 16 < frames ? end + 16 : frames;
    int len = right - left;
    if (!h3_gpu_tensor_write_f32(a, latent + left * 64, len * 64) ||
        !h3_gpu_begin(m->gpu))
      goto done;
    if (!conv_run(m, &d.first, b, a, len))
      goto flush;
    h3_gpu_tensor *tmp = a;
    a = b;
    b = tmp;
    for (int i = 0; i < 6; i++) {
      stage *s = &d.stages[i];
      int in = s->up.in, out = s->up.out;
      if (!snake_run(m, &s->s, b, a, len, in) ||
          !conv_run(m, &s->up, c, b, len))
        goto flush;
      len = (len - 1) * s->up.stride + s->up.kernel - 2 * s->up.pad;
      tmp = a;
      a = c;
      c = tmp;
      for (int j = 0; j < 3; j++) {
        residual *r = &s->r[j];
        if (!h3_gpu_copy_f32(m->gpu, skip, 0, a, 0, (size_t)len * out) ||
            !snake_run(m, &r->s1, b, a, len, out) ||
            !conv_run(m, &r->c1, c, b, len) ||
            !snake_run(m, &r->s2, b, c, len, out) ||
            !conv_run(m, &r->c2, a, b, len) ||
            !h3_gpu_add_scaled_f32(m->gpu, a, a, skip, 1, 1, len * out))
          goto flush;
      }
      /* Bound command-buffer working sets and make each decoder stage
       * cancellable. */
      if (!h3_gpu_submit(m->gpu))
        goto done;
      if (cb && !cb("Decoding audio", start, frames, ctx)) {
        fail(error, size, "Cancelled");
        goto done;
      }
      if (!h3_gpu_begin(m->gpu))
        goto done;
    }
    if (!snake_run(m, &d.final, b, a, len, 64) ||
        !conv_run(m, &d.last, c, b, len))
      goto flush;
    if (!h3_gpu_submit(m->gpu))
      goto done;
    int first = start * 1920,
        last = end * 1920 < *samples ? end * 1920 : *samples;
    if (!h3_gpu_tensor_read_f32_range(c, (size_t)(start - left) * 1920 * 2,
                                      *audio + first * 2, (last - first) * 2))
      goto done;
    continue;
  flush:
    h3_gpu_submit(m->gpu);
    goto done;
  }
  ok = 1;
  for (int i = 0; i < *samples * 2; i++)
    if (!isfinite((*audio)[i])) {
      ok = fail(error, size, "Non-finite YuE2 audio");
      break;
    }
done:
  if (!ok) {
    free(*audio);
    *audio = NULL;
    *samples = 0;
    if (error && size && !error[0])
      snprintf(error, size, "YuE2 VAE: %s", h3_gpu_error(m->gpu));
  }
  h3_gpu_tensor_free(a);
  h3_gpu_tensor_free(b);
  h3_gpu_tensor_free(c);
  h3_gpu_tensor_free(skip);
  while (m->owned_count > owned_start)
    h3_gpu_tensor_free(m->owned[--m->owned_count]);
  return ok;
}
/* Request-local RNG: reproducible within the native engine. Reference parity
 * injects fixed noise/tokens instead of assuming PyTorch shares this RNG. */
static uint64_t random64(uint64_t *s) {
  *s += UINT64_C(0x9e3779b97f4a7c15);
  uint64_t z = *s;
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}
static double uniform(uint64_t *s) {
  return ((random64(s) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}
typedef struct {
  int id;
  float score;
} candidate;
static int sample_token(float *logits, const uint32_t *history, int count,
                        int abc, int minimum, uint64_t *seed) {
  candidate top[101];
  int k = abc ? 30 : 100, n = 0;
  float penalty = abc ? 1.005f : 1.2f, temp = abc ? .7f : 1.f;
  int window = abc ? 100 : 50, end = abc ? ABC_END : MUSIC_END;
  /* Frequency penalty counts repeats, not merely membership. */
  for (int i = count > window ? count - window : 0; i < count; i++) {
    int id = history[i];
    logits[id] = logits[id] < 0 ? logits[id] * penalty : logits[id] / penalty;
  }
  int begin = abc ? 0 : CODEC, limit = abc ? EOD : CODEC + 32768;
  for (int j = begin; j <= limit; j++) {
    int id = j == limit ? end : j;
    if (id == end && count < minimum)
      continue;
    float score = logits[id] / temp;
    if (!isfinite(score))
      continue;
    if (n == k && score <= top[n - 1].score)
      continue;
    int pos = n < k ? n++ : k - 1;
    while (pos > 0 && score > top[pos - 1].score) {
      top[pos] = top[pos - 1];
      pos--;
    }
    top[pos] = (candidate){id, score};
  }
  if (!n)
    return -1;
  double sum = 0, p[100];
  for (int i = 0; i < n; i++) {
    p[i] = exp((double)top[i].score - top[0].score);
    sum += p[i];
  }
  double cumulative = 0, kept = 0;
  int used = 0;
  for (int i = 0; i < n; i++) {
    if (i && cumulative > (abc ? .9 : .95))
      break;
    cumulative += p[i] / sum;
    kept += p[i];
    used++;
  }
  double draw = uniform(seed) * kept;
  for (int i = 0; i < used; i++) {
    draw -= p[i];
    if (draw <= 0)
      return top[i].id;
  }
  return top[used - 1].id;
}
static int generate_tokens(yue2_model *m, const uint32_t *prefix, int count,
                           int maximum, int abc, uint64_t seed,
                           uint32_t *tokens, int *produced, yue2_progress cb,
                           void *ctx, char *error, size_t size) {
  *produced = 0;
  if (count + maximum > CACHE)
    return fail(
        error, size,
        "Music text and score exceed native YuE2's 24576-token context; "
        "shorten the lyrics or style");
  float *logits = malloc(VOCAB * 4);
  if (!logits)
    return 0;
  int ok = prefill(m, prefix, count, logits, cb, ctx,
                   abc ? "Preparing score context" : "Preparing token context",
                   error, size);
  if (!ok) {
    free(logits);
    return 0;
  }
  int minimum = abc ? 32 : maximum < 200 ? maximum : 200;
  for (int step = 0; step < maximum; step++) {
    if (cb &&
        !cb(abc ? "Planning melody and chords" : "Generating music tokens",
            step, maximum, ctx)) {
      ok = fail(error, size, "Cancelled");
      break;
    }
    int token = sample_token(logits, tokens, step, abc, minimum, &seed);
    if (token < 0) {
      ok = fail(error, size, "Invalid music sampling distribution");
      break;
    }
    if (token == (abc ? ABC_END : MUSIC_END))
      break;
    tokens[(*produced)++] = token;
    if (step + 1 < maximum &&
        !yue2_forward(m, (uint32_t *)&token, 1, count + step, LAYERS, logits,
                      error, size)) {
      ok = 0;
      break;
    }
  }
  free(logits);
  return ok;
}
typedef struct {
  yue2_progress callback;
  void *opaque;
  int index, count;
} chunk_progress;
static int report_chunk(const char *phase, int done, int total, void *opaque) {
  chunk_progress *p = opaque;
  if (!p->callback)
    return 1;
  if (!strcmp(phase, "Synthesizing audio")) {
    done += p->index * total;
    total *= p->count;
  }
  return p->callback(phase, done, total, p->opaque);
}
int yue2_generate(yue2_model *m, const uint32_t *text, int count,
                  double seconds, int steps, uint64_t seed, float **audio,
                  int *samples, yue2_progress cb, void *ctx, char *error,
                  size_t size) {
  *audio = NULL;
  *samples = 0;
  if (count < 1 || !isfinite(seconds) || seconds < 1 || seconds > 360 ||
      !(steps == 8 || steps == 16 || steps == 32))
    return fail(error, size, "Invalid YuE2 music request");
  int max_frames = (int)ceil(seconds * 25);
  if (count + 4096 + max_frames + 5 > CACHE)
    return fail(error, size,
                "Shorten the lyrics or style to leave room for YuE2's score "
                "and audio context");
  int context = count + 4096 + 2 * max_frames + 5;
  if (context > CACHE) context = CACHE;
  if (!reserve(m, ROWS, context, error, size))
    return 0;
  uint32_t *prefix = malloc(CACHE * 4), *abc = malloc(4096 * 4),
           *codec = malloc(max_frames * 4);
  float *noise = NULL, *latents = NULL;
  int ok = 0, n = 0, frames = 0;
  if (!prefix || !abc || !codec)
    goto done;
  prefix[0] = EOD;
  memcpy(prefix + 1, text, count * 4);
  prefix[count + 1] = ABC_START;
  int prefix_count = count + 2;
  if (!generate_tokens(m, prefix, prefix_count, 4096, 1, seed, abc, &n, cb, ctx,
                       error, size))
    goto done;
  memcpy(prefix + prefix_count, abc, n * 4);
  prefix_count += n;
  prefix[prefix_count++] = ABC_END;
  prefix[prefix_count++] = MUSIC_START;
  if (!generate_tokens(m, prefix, prefix_count, max_frames, 0, seed, codec,
                       &frames, cb, ctx, error, size) ||
      frames < 1)
    goto done;
  for (int i = 0; i < frames; i++)
    codec[i] -= CODEC;
  noise = malloc(frames * 64 * 4);
  latents = malloc(frames * 64 * 4);
  if (!noise || !latents)
    goto done;
  uint64_t rng = seed;
  for (int i = 0; i < frames * 64; i += 2) {
    double radius = sqrt(-2 * log(uniform(&rng))),
           angle = 6.283185307179586 * uniform(&rng);
    noise[i] = radius * cos(angle);
    noise[i + 1] = radius * sin(angle);
  }
  // Preserve the reference's acoustic chunk boundaries and full-song noise.
  int chunk_size = (CACHE - prefix_count - 3) / 2;
  if (chunk_size < 1) {
    fail(error, size, "No acoustic context remains");
    goto done;
  }
  chunk_progress progress = {cb, ctx, 0,
                             (frames + chunk_size - 1) / chunk_size};
  for (int offset = 0; offset < frames;
       offset += chunk_size, progress.index++) {
    int chunk_frames =
        frames - offset < chunk_size ? frames - offset : chunk_size;
    if (!yue2_synthesize(m, prefix, prefix_count, codec + offset, chunk_frames,
                         noise + offset * 64, steps, latents + offset * 64,
                         report_chunk, &progress, error, size))
      goto done;
  }
  if (!yue2_decode(m, latents, frames, audio, samples, cb, ctx, error, size))
    goto done;
  if (*samples > (int)(seconds * 48000))
    *samples = (int)(seconds * 48000);
  ok = 1;
done:
  free(prefix);
  free(abc);
  free(codec);
  free(noise);
  free(latents);
  return ok;
}

/* Development benchmark counters; never emitted during app generation. */
void yue2_print_timing(yue2_model *m) {
  h3_gpu_stats stats;
  if (h3_gpu_get_stats(m->gpu, &stats))
    fprintf(stderr, "timing encode=%.6f wait=%.6f gpu=%.6f submissions=%llu\n",
            stats.command_encode_seconds, stats.command_wait_seconds,
            stats.gpu_seconds, (unsigned long long)stats.submissions);
}

void yue2_reset_cache(yue2_model *m) { m->cached_count = 0; }
void yue2_set_linear_2d(yue2_model *m, int enabled) {
  h3_gpu_set_linear_2d(m->gpu, enabled);
}
