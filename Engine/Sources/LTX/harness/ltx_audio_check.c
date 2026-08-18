/* Does the lifted module still make the same sound as the driver it came from?
 *
 * `ltx_audio.c` is `Vendor/h3.c/tests/ltx_audio_decode.c` with its exits turned
 * into error returns. That is a mechanical change, which is exactly the kind
 * that looks fine and is not -- an error latch that short-circuits one branch
 * too early produces a plausible waveform from a half-finished decode.
 *
 * So: decode a real latent through the module and write it beside the driver's
 * WAV for the same file. Identical bytes is the only acceptable result; the
 * arithmetic did not change, so anything else is the lift.
 *
 * build:
 *   clang -O2 -fobjc-arc -o ltx_audio_check \
 *       harness/ltx_audio_check.c ltx_audio.c \
 *       ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
 *       ../../Vendor/h3.c/h3_safetensors.c \
 *       -I../../Vendor/h3.c -I. \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate -lm
 *
 * usage: ltx_audio_check AUDIO_VAE.safetensors LATENT.bin OUT.wav */
#include "ltx_audio.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put32(FILE *out, uint32_t value) {
    uint8_t bytes[4] = {(uint8_t)value, (uint8_t)(value >> 8),
                        (uint8_t)(value >> 16), (uint8_t)(value >> 24)};
    fwrite(bytes, 1, 4, out);
}

static void put16(FILE *out, uint16_t value) {
    uint8_t bytes[2] = {(uint8_t)value, (uint8_t)(value >> 8)};
    fwrite(bytes, 1, 2, out);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s AUDIO_VAE.safetensors LATENT.bin OUT.wav\n",
                argv[0]);
        return 2;
    }
    char error[512];
    h3_weight_store *store = h3_weight_store_open(argv[1], error, sizeof(error));
    if (!store) { fprintf(stderr, "cannot open the audio VAE: %s\n", error); return 1; }
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) { fprintf(stderr, "cannot create the Metal context: %s\n", error); return 1; }

    /* The generator's latent file: header, then the video latent, then ours. */
    uint32_t rows = 0;
    float *tokens = NULL;
    {
        FILE *file = fopen(argv[2], "rb");
        if (!file) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        uint32_t header[8];
        if (fread(header, sizeof(header), 1, file) != 1 ||
            header[0] != UINT32_C(0x4C545847)) {
            fprintf(stderr, "%s is not a latent\n", argv[2]);
            return 1;
        }
        const uint32_t latent = header[4];
        rows = header[5];
        if (latent != LTX_AUDIO_PATCHED) {
            fprintf(stderr, "the latent is %u wide, expected %d\n", latent,
                    LTX_AUDIO_PATCHED);
            return 1;
        }
        const size_t skip = (size_t)header[1] * header[2] * header[3] * latent;
        fseek(file, (long)(skip * sizeof(float)), SEEK_CUR);
        tokens = malloc((size_t)rows * LTX_AUDIO_PATCHED * sizeof(*tokens));
        if (!tokens ||
            fread(tokens, sizeof(*tokens), (size_t)rows * LTX_AUDIO_PATCHED,
                  file) != (size_t)rows * LTX_AUDIO_PATCHED) {
            fprintf(stderr, "cannot read the audio latent\n");
            return 1;
        }
        fclose(file);
    }

    const uint32_t frames = ltx_audio_frames_for(rows);
    printf("audio latent [%u, %d] -> %u samples, %.3f s\n", rows,
           LTX_AUDIO_PATCHED, frames,
           (double)frames / LTX_AUDIO_SAMPLE_RATE);
    /* The derived-length rule, checked against itself: 24 fps and this many
     * pixel frames should ask for exactly the rows the file carries. */
    printf("  ltx_audio_rows_for(65, 24) = %u\n", ltx_audio_rows_for(65, 24));

    float *samples = malloc((size_t)frames * 2 * sizeof(*samples));
    if (!samples) { fprintf(stderr, "cannot allocate the waveform\n"); return 1; }
    if (!ltx_audio_decode(gpu, store, tokens, rows, samples, error,
                          sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    free(tokens);

    double peak = 0.0, energy = 0.0;
    for (size_t index = 0; index < (size_t)frames * 2; index++) {
        const double value = samples[index];
        if (value > peak || -value > peak) peak = value < 0 ? -value : value;
        energy += value * value;
    }
    printf("  peak %.4f, rms %.4f\n", peak,
           sqrt(energy / ((double)frames * 2)));

    FILE *out = fopen(argv[3], "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const uint32_t data_bytes = frames * 2 * 2;
    fwrite("RIFF", 1, 4, out);
    put32(out, 36 + data_bytes);
    fwrite("WAVEfmt ", 1, 8, out);
    put32(out, 16); put16(out, 1); put16(out, 2);
    put32(out, LTX_AUDIO_SAMPLE_RATE);
    put32(out, LTX_AUDIO_SAMPLE_RATE * 2 * 2);
    put16(out, 2 * 2); put16(out, 16);
    fwrite("data", 1, 4, out);
    put32(out, data_bytes);
    for (size_t index = 0; index < (size_t)frames * 2; index++) {
        double value = samples[index];
        if (value > 1.0) value = 1.0;
        if (value < -1.0) value = -1.0;
        put16(out, (uint16_t)(int16_t)lrint(value * 32767.0));
    }
    fclose(out);
    printf("wrote %s -- compare it byte for byte with the driver's\n", argv[3]);

    free(samples);
    h3_gpu_free(gpu);
    h3_weight_store_free(store);
    return 0;
}
