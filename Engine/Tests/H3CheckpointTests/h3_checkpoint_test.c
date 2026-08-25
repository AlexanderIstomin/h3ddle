#include "h3_checkpoint.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define REQUIRE(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "checkpoint test failed: %s\n", message); \
        exit(1); \
    } \
} while (0)

int main(void) {
    char path[] = "/tmp/h3ddle-checkpoint-XXXXXX";
    int placeholder = mkstemp(path);
    REQUIRE(placeholder >= 0, "make path");
    REQUIRE(close(placeholder) == 0 && unlink(path) == 0, "clear path");

    const char fingerprint[] =
        "0123456789abcdef0123456789abcdef"
        "0123456789abcdef0123456789abcdef";
    float video[] = {1, 2, 3};
    float audio[] = {4, 5};
    float last_video[] = {6, 7, 8};
    float last_audio[] = {9, 10};
    h3_checkpoint_state source = {
        .total_steps = 8, .next_step = 3, .reuse_interval = 2,
        .last_evaluated = 2, .previous_evaluated = -1,
        .video_count = 3, .audio_count = 2,
        .video = video, .audio = audio,
        .last_video_velocity = last_video,
        .last_audio_velocity = last_audio
    };
    char detail[256];
    REQUIRE(h3_checkpoint_save(path, fingerprint, &source,
                               detail, sizeof(detail)), detail);

    h3_checkpoint_state loaded = {0};
    REQUIRE(h3_checkpoint_load(path, fingerprint, 8, 2, 3, 2,
                               &loaded, detail, sizeof(detail)), detail);
    REQUIRE(loaded.next_step == 3, "next step");
    REQUIRE(!memcmp(loaded.video, video, sizeof(video)), "video payload");
    REQUIRE(!memcmp(loaded.last_audio_velocity, last_audio,
                    sizeof(last_audio)), "reuse payload");
    h3_checkpoint_state_free(&loaded);

    int file = open(path, O_WRONLY);
    REQUIRE(file >= 0 && lseek(file, -1, SEEK_END) >= 0, "open payload");
    unsigned char corrupt = 0xff;
    REQUIRE(write(file, &corrupt, 1) == 1 && close(file) == 0,
            "corrupt payload");
    REQUIRE(!h3_checkpoint_load(path, fingerprint, 8, 2, 3, 2,
                                &loaded, detail, sizeof(detail)),
            "corruption rejected");
    h3_checkpoint_remove(path);
    puts("checkpoint test passed");
    return 0;
}
