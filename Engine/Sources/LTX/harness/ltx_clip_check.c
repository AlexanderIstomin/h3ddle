/* A prompt to a clip, in one process — does `ltx_generate` actually work?
 *
 * The five stage checks each hold one piece against the driver. This holds the
 * *join*, which is the part no component check reaches and the part that has
 * produced every bug worth having in this port. `ltx_clip.sh` does the same
 * thing as six processes; the whole point of `ltx_generate` is that it does not
 * have to be six, so this is the check that the sequencing survives being in
 * one address space.
 *
 * It writes the same two files the standalone decoders write, so the existing
 * mux script takes them unchanged.
 *
 * build:
 *   clang -O2 -fobjc-arc -o ltx_clip_check \
 *       harness/ltx_clip_check.c ltx_generate.c ltx_dit.c ltx_text.c \
 *       ltx_connector.c ltx_video.c ltx_audio.c \
 *       ../../Vendor/h3.c/h3_gpu.m ../../Vendor/h3.c/h3_weights.c \
 *       ../../Vendor/h3.c/h3_safetensors.c ../../Vendor/h3.c/h3_tokenizer.m \
 *       -I../../Vendor/h3.c -I. \
 *       -framework Foundation -framework Metal \
 *       -framework MetalPerformanceShaders \
 *       -framework MetalPerformanceShadersGraph -framework Accelerate \
 *       -licucore -lm
 *
 * usage: ltx_clip_check PACKAGE "a prompt" OUT [PIXELS] [FRAMES] [STEPS] [SEED]
 *
 * writes OUT.frames.bin and OUT.wav. */
#include "ltx_generate.h"
#include "ltx_audio.h"
#include "ltx_video.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FRAMES_MAGIC 0x4C545846
#define STEREO 2

static double now(void) {
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec * 1e-9;
}

/* One line per phase change, plus the step counter inside the two phases that
 * have one. The phases are the reason this prints at all: the tower is a
 * minute of silence before the first denoise step. */
static int report(const char *phase, int step, int steps, void *context) {
    static char seen[32] = "";
    static int last_step = -1;
    double *began = context;
    if (strcmp(phase, seen) != 0) {
        printf("== %s ==\n", phase);
        snprintf(seen, sizeof(seen), "%s", phase);
        last_step = -1;
    }
    if (step != last_step) {
        last_step = step;
        if (step > 0)
            printf("   %s %d/%d, %.1f s in\n", phase, step, steps,
                   now() - *began);
    }
    fflush(stdout);
    return 1;
}

static void put32(FILE *out, uint32_t value) {
    const unsigned char bytes[4] = {(unsigned char)(value & 0xff),
                                    (unsigned char)((value >> 8) & 0xff),
                                    (unsigned char)((value >> 16) & 0xff),
                                    (unsigned char)((value >> 24) & 0xff)};
    fwrite(bytes, 1, 4, out);
}

static void put16(FILE *out, uint16_t value) {
    const unsigned char bytes[2] = {(unsigned char)(value & 0xff),
                                    (unsigned char)((value >> 8) & 0xff)};
    fwrite(bytes, 1, 2, out);
}

/* `ltx_audio_decode`'s own WAV writer, so the two outputs are interchangeable
 * and the same mux script takes either. */
static int write_wav(const char *path, const float *samples, uint32_t frames) {
    FILE *out = fopen(path, "wb");
    if (!out) return 0;
    const uint32_t data_bytes = frames * STEREO * 2;
    fwrite("RIFF", 1, 4, out);
    put32(out, 36 + data_bytes);
    fwrite("WAVEfmt ", 1, 8, out);
    put32(out, 16);
    put16(out, 1);
    put16(out, STEREO);
    put32(out, LTX_AUDIO_SAMPLE_RATE);
    put32(out, LTX_AUDIO_SAMPLE_RATE * STEREO * 2);
    put16(out, STEREO * 2);
    put16(out, 16);
    fwrite("data", 1, 4, out);
    put32(out, data_bytes);
    for (size_t index = 0; index < (size_t)frames * STEREO; index++) {
        double value = samples[index];
        if (value > 1.0) value = 1.0;
        if (value < -1.0) value = -1.0;
        put16(out, (uint16_t)(int16_t)lrint(value * 32767.0));
    }
    return fclose(out) == 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 8) {
        fprintf(stderr, "usage: %s PACKAGE \"a prompt\" OUT "
                        "[PIXELS] [FRAMES] [STEPS] [SEED]\n", argv[0]);
        return 2;
    }
    ltx_request request = {0};
    request.package = argv[1];
    request.shaders = "h3_shaders.metal";
    request.prompt = argv[2];
    request.pixels = argc > 4 ? atoi(argv[4]) : 512;
    request.frames = argc > 5 ? atoi(argv[5]) : 65;
    request.steps = argc > 6 ? atoi(argv[6]) : 0;
    request.seed = argc > 7 ? strtoull(argv[7], NULL, 10) : 16;

    char error[512];
    ltx_shape shape;
    if (!ltx_plan(&request, &shape, error, sizeof(error))) {
        fprintf(stderr, "FAIL: %s\n", error);
        return 1;
    }
    printf("%s\n", request.prompt);
    printf("%d frames of %dx%d at %d fps = %.3f s; %u audio frames = %.3f s\n",
           shape.frames, shape.pixels, shape.pixels,
           request.fps > 0 ? request.fps : LTX_DEFAULT_FPS,
           (double)shape.frames /
               (request.fps > 0 ? request.fps : LTX_DEFAULT_FPS),
           shape.audio_frames,
           (double)shape.audio_frames / LTX_AUDIO_SAMPLE_RATE);
    printf("%.1f MB of video, %.1f MB of audio\n",
           (double)shape.video_floats * sizeof(float) / 1e6,
           (double)shape.audio_floats * sizeof(float) / 1e6);
    fflush(stdout);

    float *video = malloc(shape.video_floats * sizeof(float));
    float *audio = malloc(shape.audio_floats * sizeof(float));
    if (!video || !audio) {
        fprintf(stderr, "cannot allocate the clip\n");
        return 1;
    }
    double began = now();
    if (!ltx_generate(&request, video, audio, report, &began, error,
                      sizeof(error))) {
        fprintf(stderr, error[0] ? "FAIL: %s\n" : "cancelled\n", error);
        return 1;
    }
    printf("== done ==\nprompt to clip in %.1f s\n", now() - began);

    /* Nothing downstream will say so if a stage quietly produced zeros, and a
     * black clip with a silent track is exactly what a short-circuited error
     * latch looks like. */
    double video_peak = 0.0, audio_peak = 0.0, audio_energy = 0.0;
    for (size_t index = 0; index < shape.video_floats; index++) {
        const double magnitude = fabs((double)video[index]);
        if (magnitude > video_peak) video_peak = magnitude;
    }
    for (size_t index = 0; index < shape.audio_floats; index++) {
        const double magnitude = fabs((double)audio[index]);
        if (magnitude > audio_peak) audio_peak = magnitude;
        audio_energy += (double)audio[index] * (double)audio[index];
    }
    const double rms = sqrt(audio_energy / (double)shape.audio_floats);
    printf("video peak %.4f, audio peak %.4f rms %.4f\n", video_peak,
           audio_peak, rms);
    if (video_peak < 1e-3) { fprintf(stderr, "FAIL: the picture is blank\n"); return 1; }
    if (rms < 1e-4) { fprintf(stderr, "FAIL: the track is silent\n"); return 1; }

    char path[1200];
    snprintf(path, sizeof(path), "%s.frames.bin", argv[3]);
    FILE *out = fopen(path, "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", path); return 1; }
    /* `ltx_decode`'s header, field for field. */
    const uint32_t header[4] = {(uint32_t)FRAMES_MAGIC, LTX_VIDEO_CHANNELS,
                                (uint32_t)shape.frames, (uint32_t)shape.pixels};
    fwrite(header, sizeof(header), 1, out);
    const uint32_t width = (uint32_t)shape.pixels;
    fwrite(&width, sizeof(width), 1, out);
    fwrite(video, sizeof(float), shape.video_floats, out);
    if (fclose(out) != 0) { fprintf(stderr, "cannot close %s\n", path); return 1; }
    printf("wrote %s: %d frames of %dx%d, %d channels\n", path, shape.frames,
           shape.pixels, shape.pixels, LTX_VIDEO_CHANNELS);

    snprintf(path, sizeof(path), "%s.wav", argv[3]);
    if (!write_wav(path, audio, shape.audio_frames)) {
        fprintf(stderr, "cannot write %s\n", path);
        return 1;
    }
    printf("wrote %s: %u samples, %.3f s at %d Hz\n", path, shape.audio_frames,
           (double)shape.audio_frames / LTX_AUDIO_SAMPLE_RATE,
           LTX_AUDIO_SAMPLE_RATE);
    free(video); free(audio);
    return 0;
}
