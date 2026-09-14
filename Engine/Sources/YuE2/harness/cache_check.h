/* Exercise prefix identity, reuse after NAR writes, and a diverging suffix. */
typedef struct { int first, total; } cache_observation;
static int observe_cache(const char *phase, int done, int total, void *opaque) {
  cache_observation *observation = opaque;
  if (!strcmp(phase, "Preparing audio context") && observation->first < 0) {
    observation->first = done;
    observation->total = total;
  }
  return strcmp(phase, "Synthesizing audio") != 0;
}
static int cache_condition(yue2_model *model, const uint32_t *prefix, int count,
                           const uint32_t *codec, int frames, const float *noise,
                           float *out, int expected, char *error, size_t size) {
  cache_observation observation = {-1, 0};
  int result = yue2_synthesize(model, prefix, count, codec, frames, noise, 8,
      out, observe_cache, &observation, error, size);
  if (result || strcmp(error, "Cancelled") || observation.first != expected ||
      observation.total != count + frames + 1) {
    fprintf(stderr, "cache prefill: observed %d, expected %d: %s\n",
            observation.first, expected, error);
    return 0;
  }
  return yue2_velocity(model, noise, frames, .5f, out, error, size);
}
static int cache_check(const char *package, const char *shaders, const char *dir) {
  char error[512] = {0}; int bytes;
  uint32_t *prefix = read_file(dir, "prefix.bin", &bytes); int count = bytes / 4;
  uint32_t *codec = read_file(dir, "codec.bin", &bytes); int frames = bytes / 4;
  float *noise = read_file(dir, "noise.bin", &bytes);
  float *cached = malloc(bytes), *cold = malloc(bytes);
  yue2_model *model = yue2_load(package, shaders, error, sizeof(error));
  int ok = 0;
  if (!model || !cached || !cold || frames < 1) goto done;
  if (!cache_condition(model,prefix,count,codec,frames,noise,cold,0,error,sizeof(error))) goto done;
  if (!cache_condition(model,prefix,count,codec,frames,noise,cached,count+frames+1,error,sizeof(error))) goto done;
  if (memcmp(cold,cached,bytes)) {
    fprintf(stderr,"identical cached conditioning changed velocity\n"); goto done;
  }
  codec[frames-1] ^= 1;
  if (!cache_condition(model,prefix,count,codec,frames,noise,cached,count+frames-1,error,sizeof(error))) goto done;
  yue2_reset_cache(model);
  if (!cache_condition(model,prefix,count,codec,frames,noise,cold,0,error,sizeof(error))) goto done;
  double energy=0, difference=0;
  for (int i=0;i<frames*64;i++) {
    energy += cold[i]*cold[i];
    difference += (cached[i]-cold[i])*(cached[i]-cold[i]);
  }
  double relative=sqrt(difference/energy);
  if (!isfinite(relative) || relative >= .03) goto done;
  printf("cache reuse exact; divergent suffix relative_l2=%g\n",relative);
  ok=1;
done:
  if (!ok) fprintf(stderr,"cache check failed: %s\n",error);
  yue2_free(model);free(prefix);free(codec);free(noise);free(cached);free(cold);
  return ok ? 0 : 1;
}
