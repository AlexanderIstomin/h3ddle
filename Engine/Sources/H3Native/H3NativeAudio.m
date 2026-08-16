/* Reading a reference clip for voice cloning.
 *
 * h3.c already reads audio through AVFoundation, but at 32 kHz stereo, which
 * is what H3's own audio branch wants. Qwen3-TTS's speaker encoder wants 24
 * kHz mono, and its mel filterbank is sensitive to the rate — resampling by
 * hand between the two would put a homemade filter in front of the one thing
 * that carries the voice. AVAudioConverter does it properly, so this asks for
 * the format the encoder actually wants and lets the system produce it.
 *
 * Anything AVFoundation reads works: WAV, AIFF, MP3, M4A, MP4, MOV. */
#import <AVFoundation/AVFoundation.h>

#include "H3Native.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int h3ddle_read_mono_f32(const char *path, int sample_rate, int max_samples,
                         float **pcm, int *samples, char *error,
                         size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !pcm || !samples || sample_rate <= 0 || max_samples <= 0) {
        if (error && error_size)
            snprintf(error, error_size, "a path, rate and destination are "
                     "required");
        return 0;
    }
    *pcm = NULL;
    *samples = 0;

    @autoreleasepool {
        NSError *failure = nil;
        AVAudioFile *file = [[AVAudioFile alloc]
            initForReading:[NSURL fileURLWithPath:@(path)] error:&failure];
        if (!file) {
            if (error && error_size)
                snprintf(error, error_size, "cannot read %s: %s", path,
                         failure.localizedDescription.UTF8String ?: "unknown");
            return 0;
        }
        AVAudioFormat *source = file.processingFormat;
        AVAudioFormat *target = [[AVAudioFormat alloc]
            initWithCommonFormat:AVAudioPCMFormatFloat32
                      sampleRate:(double)sample_rate
                        channels:1
                     interleaved:NO];
        AVAudioConverter *converter = [[AVAudioConverter alloc]
            initFromFormat:source toFormat:target];
        if (!target || !converter) {
            if (error && error_size)
                snprintf(error, error_size, "cannot convert %s to %d Hz mono",
                         path, sample_rate);
            return 0;
        }

        double ratio = (double)sample_rate / source.sampleRate;
        double expected = (double)file.length * ratio + 4096.0;
        if (expected > (double)max_samples) expected = (double)max_samples;
        AVAudioPCMBuffer *output = [[AVAudioPCMBuffer alloc]
            initWithPCMFormat:target
                frameCapacity:(AVAudioFrameCount)expected];
        if (!output) {
            if (error && error_size)
                snprintf(error, error_size, "out of memory");
            return 0;
        }

        /* The converter pulls; the file is drained once and then reports the
         * end of the stream, which is what stops the pull. */
        __block BOOL drained = NO;
        __block NSError *readFailure = nil;
        AVAudioConverterOutputStatus status = [converter
            convertToBuffer:output
                      error:&failure
         withInputFromBlock:^AVAudioBuffer *(
                AVAudioPacketCount count,
                AVAudioConverterInputStatus *inputStatus) {
            if (drained) {
                *inputStatus = AVAudioConverterInputStatus_EndOfStream;
                return nil;
            }
            AVAudioPCMBuffer *chunk = [[AVAudioPCMBuffer alloc]
                initWithPCMFormat:source frameCapacity:count];
            if (!chunk || ![file readIntoBuffer:chunk error:&readFailure] ||
                chunk.frameLength == 0) {
                drained = YES;
                *inputStatus = AVAudioConverterInputStatus_EndOfStream;
                return nil;
            }
            *inputStatus = AVAudioConverterInputStatus_HaveData;
            return chunk;
        }];
        if (status == AVAudioConverterOutputStatus_Error || readFailure) {
            NSError *reported = failure ?: readFailure;
            if (error && error_size)
                snprintf(error, error_size, "cannot decode %s: %s", path,
                         reported.localizedDescription.UTF8String ?: "unknown");
            return 0;
        }

        const int count = (int)output.frameLength;
        if (count <= 0) {
            if (error && error_size)
                snprintf(error, error_size, "%s holds no audio", path);
            return 0;
        }
        float *values = malloc((size_t)count * sizeof(float));
        if (!values) {
            if (error && error_size)
                snprintf(error, error_size, "out of memory");
            return 0;
        }
        memcpy(values, output.floatChannelData[0],
               (size_t)count * sizeof(float));
        *pcm = values;
        *samples = count;
    }
    return 1;
}
