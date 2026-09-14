#include "H3Native.h"
#include "yue2.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The Swift service serializes generation and evicts other model caches before
 * entering. Keep all YuE2 ownership within this call, including cancellation.
 */
int h3ddle_yue2_generate(const char *package, const char *shaders,
                         const char *style, const char *lyrics, double seconds,
                         int steps, unsigned long long seed, const char *output,
                         h3ddle_zimage_step callback, void *opaque,
                         double *duration, char *error, size_t size) {
  if (error && size)
    error[0] = 0;
  if (duration)
    *duration = 0;
  if (!package || !shaders || !style || !lyrics || !style[0] || !output ||
      !isfinite(seconds) || seconds < 1 || seconds > 360 ||
      !(steps == 8 || steps == 16 || steps == 32)) {
    snprintf(
        error, size,
        "YuE2 needs style, optional lyrics, 1–360 seconds and 8/16/32 steps");
    return 0;
  }
  if (strlen(style) > 16000 || strlen(lyrics) > 20000) {
    snprintf(error, size, "YuE2 style or lyrics are too long");
    return 0;
  }
  if (access(output, F_OK) == 0) {
    snprintf(error, size, "YuE2 output already exists");
    return 0;
  }
  if (callback && !callback("Loading YuE2", 0, 1, opaque))
    return 0;
  char path[4096], temporary[4096];
  int ok = 0, count = 0, samples = 0;
  uint32_t *ids = NULL;
  float *audio = NULL;
  yue2_model *m = NULL;
  yue2_tokenizer *tokenizer = NULL;
  FILE *file = NULL;
  size_t text_size = strlen(style) + strlen(lyrics) + 256;
  char *text = malloc(text_size);
  if (!text)
    goto done;
  snprintf(
      text, text_size,
      "Generate a chord-annotated ABC transcription, then generate music with "
      "codec tokens from the given conditions.\n[Tags]\n%s%s\n[Lyrics]\n%s\n",
      style, lyrics[0] ? "" : ", instrumental, no vocals", lyrics);
  snprintf(path, sizeof(path), "%s/model/qwen.tiktoken", package);
  tokenizer = yue2_tokenizer_load(path, error, size);
  if (!tokenizer)
    goto done;
  count = yue2_tokenize(tokenizer, text, &ids, error, size);
  if (!count)
    goto done;
  m = yue2_load(package, shaders, error, size);
  if (!m)
    goto done;
  if (!yue2_generate(m, ids, count, seconds, steps, seed, &audio, &samples,
                     callback, opaque, error, size))
    goto done;
  if (callback && !callback("Saving audio", 0, 1, opaque))
    goto done;
  if (snprintf(temporary, sizeof(temporary), "%s.partial", output) >=
      (int)sizeof(temporary)) {
    snprintf(error, size, "Output path too long");
    goto done;
  }
  file = fopen(temporary, "wb");
  if (!file) {
    snprintf(error, size, "Cannot create YuE2 output");
    goto done;
  }
  uint32_t data_bytes = samples * 2u * 3u, chunk = 36u + data_bytes, fmt = 16,
           rate = 48000, byte_rate = 48000 * 6;
  uint16_t pcm = 1, channels = 2, align = 6, bits = 24;
#define WRITE(value) (fwrite(&(value), sizeof(value), 1, file) == 1)
  ok = fwrite("RIFF", 1, 4, file) == 4 && WRITE(chunk) &&
       fwrite("WAVEfmt ", 1, 8, file) == 8 && WRITE(fmt) && WRITE(pcm) &&
       WRITE(channels) && WRITE(rate) && WRITE(byte_rate) && WRITE(align) &&
       WRITE(bits) && fwrite("data", 1, 4, file) == 4 && WRITE(data_bytes);
  for (int i = 0; ok && i < samples * 2; i++) {
    int32_t sample = (int32_t)lrintf(fmaxf(-1, fminf(1, audio[i])) * 8388607);
    unsigned char bytes[] = {sample & 255, (sample >> 8) & 255,
                             (sample >> 16) & 255};
    ok = fwrite(bytes, 1, 3, file) == 3;
  }
#undef WRITE
  if (fclose(file) != 0)
    ok = 0;
  file = NULL;
  if (callback && !callback("Saving audio", 1, 1, opaque))
    ok = 0;
  if (ok && rename(temporary, output) != 0)
    ok = 0;
  if (!ok)
    unlink(temporary);
  else if (duration)
    *duration = (double)samples / 48000;
done:
  if (file)
    fclose(file);
  yue2_tokenizer_free(tokenizer);
  yue2_free(m);
  free(text);
  free(ids);
  free(audio);
  if (!ok && error && size && !error[0])
    snprintf(error, size,
             "YuE2 generation cancelled or output could not be written");
  return ok;
}
