#include "yue2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static void *read_file(const char *dir, const char *name, int *count) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    exit(2);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  rewind(f);
  void *p = calloc(1, n + 1);
  if (!p || fread(p, 1, n, f) != n)
    exit(2);
  fclose(f);
  *count = (int)n;
  return p;
}
static void write_file(const char *dir, const char *name, const void *data,
                       size_t count) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/native-%s.bin", dir, name);
  FILE *f = fopen(path, "wb");
  if (!f || fwrite(data, 1, count, f) != count)
    exit(2);
  fclose(f);
}
static int progress(const char *phase, int done, int total, void *ctx) {
  (void)ctx;
  static char prior[100];
  if (strcmp(prior, phase) || done == 0) {
    snprintf(prior, sizeof(prior), "%s", phase);
    fprintf(stderr, "%s %d/%d\n", phase, done, total);
  }
  return 1;
}
#include "attention_check.h"
#include "benchmark.h"
#include "cache_check.h"
/* Exercise scratch growth beyond 1024 rows with real weights, then evaluate
 * one acoustic velocity. This is a capacity check, not a song-quality fixture.
 */
static int stop_at_synthesis(const char *phase, int done, int total,
                             void *ctx) {
  (void)done;
  (void)total;
  (void)ctx;
  return strcmp(phase, "Synthesizing audio") != 0;
}
static int cancel_capacity_prefill(const char *phase, int done, int total,
                                   void *ctx) {
  (void)phase;
  (void)done;
  (void)total;
  (void)ctx;
  return 0;
}
static int cancel_after_layer(const char *phase, int done, int total,
                              void *opaque) {
  int *observed = opaque;
  if (!strcmp(phase, "Synthesizing audio") && done > 0) {
    *observed = done;
    return 0;
  }
  (void)total;
  return 1;
}
static int capacity_check(const char *package, const char *shaders, int compare) {
  char error[512] = {0};
  yue2_model *m = yue2_load(package, shaders, error, sizeof(error));
  if (!m) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }
  const int frames = 1100;
  uint32_t *prefix = calloc(6573, sizeof(uint32_t));
  uint32_t *codec = calloc(9000, sizeof(uint32_t));
  float *noise = calloc(9000 * 64, sizeof(float));
  float *out = calloc(9000 * 64, sizeof(float));
  if (!prefix || !codec || !noise || !out)
    return 1;
  for (int i = 0; i < 6573; i++)
    prefix[i] = 151643;
  prefix[1] = 151851;
  int result =
      yue2_synthesize(m, prefix, 6573, codec, 9000, noise, 8, out,
                      cancel_capacity_prefill, NULL, error, sizeof(error));
  if (result || strcmp(error, "Cancelled")) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }
  printf("capacity allocated frames=9000 rows=9002 context=24576\n");
  fflush(stdout);
  result = yue2_synthesize(m, prefix, 2, codec, frames, noise, 8, out,
                           stop_at_synthesis, NULL, error, sizeof(error));
  if (result || strcmp(error, "Cancelled")) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }
  struct timespec velocity_start, velocity_end;
  clock_gettime(CLOCK_MONOTONIC, &velocity_start);
  if (!yue2_velocity(m, noise, frames, .5f, out, error, sizeof(error))) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }
  clock_gettime(CLOCK_MONOTONIC, &velocity_end);
  printf("velocity seconds=%.6f\n",
         velocity_end.tv_sec - velocity_start.tv_sec +
             (velocity_end.tv_nsec - velocity_start.tv_nsec) / 1e9);
  float peak = 0;
  for (int i = 0; i < frames * 64; i++) {
    if (!isfinite(out[i]))
      return 1;
    peak = fmaxf(peak, fabsf(out[i]));
  }
  printf("capacity frames=%d rows=%d finite_velocity_peak=%g\n", frames,
         frames + 2, peak);
  if (compare) {
    float *fast = malloc(frames * 64 * sizeof(float));
    if (!fast) return 1;
    memcpy(fast, out, frames * 64 * sizeof(float));
    setenv("H3_YUE2_REFERENCE_ATTENTION", "1", 1);
    clock_gettime(CLOCK_MONOTONIC, &velocity_start);
    if (!yue2_velocity(m, noise, frames, .5f, out, error, sizeof(error))) return 1;
    clock_gettime(CLOCK_MONOTONIC, &velocity_end);
    unsetenv("H3_YUE2_REFERENCE_ATTENTION");
    double energy = 0, difference = 0;
    for (int i = 0; i < frames * 64; i++) {
      energy += out[i] * out[i];
      difference += (fast[i] - out[i]) * (fast[i] - out[i]);
    }
    double relative = sqrt(difference / energy);
    printf("reference velocity seconds=%.6f relative_l2=%g\n",
           velocity_end.tv_sec - velocity_start.tv_sec +
           (velocity_end.tv_nsec - velocity_start.tv_nsec) / 1e9, relative);
    free(fast);
    if (!isfinite(relative) || relative > .03) return 1;
  }
  int observed = 0;
  result = yue2_synthesize(m, prefix, 2, codec, frames, noise, 8, out,
                           cancel_after_layer, &observed, error, sizeof(error));
  if (result || observed != 1 || strcmp(error, "Cancelled")) {
    fprintf(stderr, "layer cancellation failed: %s\n", error);
    return 1;
  }
  printf("cancellation stopped after acoustic layer %d\n", observed);
  yue2_free(m);
  free(prefix);
  free(codec);
  free(noise);
  free(out);
  return peak > 0 ? 0 : 1;
}
static int linear_schedule(const char *package, const char *shaders, const char *dir) {
  char error[512] = {0}; int n;
  uint32_t *prefix = read_file(dir, "prefix.bin", &n); int count=n/4;
  uint32_t *codec = read_file(dir, "codec.bin", &n); int frames=n/4;
  float *noise = read_file(dir, "noise.bin", &n), *out=malloc(n);
  yue2_model *m=yue2_load(package,shaders,error,sizeof(error));
  if (!m || !out) return 1;
  if (yue2_synthesize(m,prefix,count,codec,frames,noise,8,out,stop_at_synthesis,NULL,error,sizeof(error)) || strcmp(error,"Cancelled")) return 1;
  for(int mode=0;mode<2;mode++) {
    yue2_set_linear_2d(m,mode);
    if(!yue2_velocity(m,noise,frames,.5f,out,error,sizeof(error))) {puts(error);return 1;}
  }
  int order[]={0,1,1,0,1,0,0,1};
  for(int i=0;i<8;i++) {
    yue2_set_linear_2d(m,order[i]);
    double start=benchmark_clock();
    if(!yue2_velocity(m,noise,frames,.5f,out,error,sizeof(error))) {puts(error);return 1;}
    printf("{\"linear_2d\":%d,\"seconds\":%.6f}\n",order[i],benchmark_clock()-start);fflush(stdout);
  }
  free(prefix);free(codec);free(noise);free(out);yue2_free(m);return 0;
}
int main(int argc, char **argv) {
  if (argc == 5 && !strcmp(argv[1], "--cache")) return cache_check(argv[2],argv[3],argv[4]);
  if (argc == 5 && !strcmp(argv[1], "--linear-schedule")) return linear_schedule(argv[2],argv[3],argv[4]);
  if (argc == 5 && !strcmp(argv[1], "--benchmark")) return benchmark(argv[2], argv[3], argv[4], 0);
  if (argc == 5 && !strcmp(argv[1], "--replay-benchmark")) return benchmark(argv[2], argv[3], argv[4], 1);
  if (argc == 4 && (!strcmp(argv[1], "--capacity") || !strcmp(argv[1], "--compare-attention")))
    return capacity_check(argv[2], argv[3], !strcmp(argv[1], "--compare-attention"));
  if (argc == 3 && !strcmp(argv[1], "--attention"))
    return attention_check(argv[2]);
  if (argc != 4) {
    fprintf(stderr, "usage: yue2_check PACKAGE SHADERS FIXTURES\n");
    return 2;
  }
  char error[512] = {0}, path[4096];
  int n;
  snprintf(path, sizeof(path), "%s/model/qwen.tiktoken", argv[1]);
  yue2_tokenizer *tok = yue2_tokenizer_load(path, error, sizeof(error));
  if (!tok) {
    puts(error);
    return 1;
  }
  for (int i = -1; i < 3; i++) {
    char name[50];
    snprintf(name, sizeof(name), i < 0 ? "text.txt" : "text-%d.txt", i);
    char *text = read_file(argv[3], name, &n);
    uint32_t *ids = NULL;
    int count = yue2_tokenize(tok, text, &ids, error, sizeof(error));
    if (!count) {
      puts(error);
      return 1;
    }
    snprintf(name, sizeof(name), i < 0 ? "text_ids" : "text-%d", i);
    write_file(argv[3], name, ids, count * 4);
    free(text);
    free(ids);
  }
  yue2_tokenizer_free(tok);
  fprintf(stderr, "Loading native YuE2\n");
  yue2_model *m = yue2_load(argv[1], argv[2], error, sizeof(error));
  if (!m) {
    puts(error);
    return 1;
  }
  uint32_t *ids = read_file(argv[3], "ids.bin", &n);
  int count = n / 4;
  float *out = malloc(184704 * 4 + count * 2048 * 4);
#define CHECK(call)                                                            \
  if (!(call)) {                                                               \
    fprintf(stderr, "%s\n", error);                                            \
    return 1;                                                                  \
  }
  CHECK(yue2_forward(m, ids, count, 0, 1, out, error, sizeof(error)));
  write_file(argv[3], "block", out, count * 2048 * 4);
  CHECK(yue2_forward(m, ids, count, 0, 28, out, error, sizeof(error)));
  write_file(argv[3], "logits", out, 184704 * 4);
  /* Cached decode must agree with one-shot prefill for the final position. */
  CHECK(yue2_forward(m, ids, count - 1, 0, 28, NULL, error, sizeof(error)));
  CHECK(yue2_forward(m, ids + count - 1, 1, count - 1, 28, out, error,
                     sizeof(error)));
  write_file(argv[3], "cached-logits", out, 184704 * 4);
  uint32_t *prefix = read_file(argv[3], "prefix.bin", &n);
  int pn = n / 4;
  uint32_t *codec = read_file(argv[3], "codec.bin", &n);
  int frames = n / 4;
  float *noise = read_file(argv[3], "noise.bin", &n);
  float *latent = malloc(n);
  for (int steps = 8; steps <= 32; steps *= 2) {
    fprintf(stderr, "Native synthesis %d\n", steps);
    CHECK(yue2_synthesize(m, prefix, pn, codec, frames, noise, steps, latent,
                          progress, NULL, error, sizeof(error)));
    char name[50];
    snprintf(name, sizeof(name), "latent-%d", steps);
    write_file(argv[3], name, latent, n);
  }
  CHECK(yue2_velocity(m, noise, frames, .5f, latent, error, sizeof(error)));
  write_file(argv[3], "velocity", latent, n);
  float *decode = read_file(argv[3], "decode_input.bin", &n), *audio = NULL;
  int samples;
  CHECK(yue2_decode(m, decode, n / (64 * 4), &audio, &samples, progress, NULL,
                    error, sizeof(error)));
  write_file(argv[3], "audio", audio, (size_t)samples * 2 * 4);
  yue2_free(m);
  free(ids);
  free(out);
  free(prefix);
  free(codec);
  free(noise);
  free(latent);
  free(decode);
  free(audio);
  fprintf(stderr, "Native parity outputs complete\n");
  return 0;
}
