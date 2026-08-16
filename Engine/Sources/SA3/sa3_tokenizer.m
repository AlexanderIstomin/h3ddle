#import <Foundation/Foundation.h>

#include "sa3_tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Gemma's BPE, driven from the same tokenizer.json the reference uses.
 *
 * The pipeline the file describes is: rewrite every space as U+2581, then
 * merge adjacent pairs cheapest-rank-first. The declared pre-tokenizer splits
 * on spaces, but the normaliser has already removed them, so it never fires —
 * the whole prompt is merged as one run, which is what SentencePiece does.
 *
 * Anything with no vocabulary entry falls back to one token per UTF-8 byte,
 * spelled <0xNN>, so an unusual character costs several tokens instead of
 * derailing the prompt. */

#define SPACE_MARKER @"▁"

@interface SA3Tokenizer : NSObject
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *vocab;
/* "first\x01second" -> rank; lower merges first. */
@property(nonatomic, strong) NSDictionary<NSString *, NSNumber *> *ranks;
@end
@implementation SA3Tokenizer
@end

struct sa3_tokenizer {
    void *object;
};

static void fail(char *error, size_t error_size, NSString *message) {
    if (!error || !error_size) return;
    const char *text = message.UTF8String;
    snprintf(error, error_size, "%s", text ? text : "tokenizer failure");
}

sa3_tokenizer *sa3_tokenizer_load(const char *tokenizer_json, char *error,
                                  size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!tokenizer_json) {
        fail(error, error_size, @"no tokenizer path");
        return NULL;
    }
    @autoreleasepool {
        NSString *path = [NSString stringWithUTF8String:tokenizer_json];
        NSData *data = [NSData dataWithContentsOfFile:path];
        if (!data) {
            fail(error, error_size, @"cannot read the tokenizer file");
            return NULL;
        }
        NSError *parseError = nil;
        NSDictionary *config = [NSJSONSerialization JSONObjectWithData:data
                                                              options:0
                                                                error:&parseError];
        if (![config isKindOfClass:NSDictionary.class]) {
            fail(error, error_size, parseError.localizedDescription ?:
                 @"the tokenizer file is not JSON");
            return NULL;
        }
        NSDictionary *model = config[@"model"];
        if (![model[@"type"] isEqual:@"BPE"] ||
            ![model[@"vocab"] isKindOfClass:NSDictionary.class] ||
            ![model[@"merges"] isKindOfClass:NSArray.class]) {
            fail(error, error_size, @"unexpected tokenizer specification");
            return NULL;
        }

        SA3Tokenizer *tokenizer = [[SA3Tokenizer alloc] init];
        tokenizer.vocab = model[@"vocab"];

        NSArray *merges = model[@"merges"];
        NSMutableDictionary *ranks =
            [NSMutableDictionary dictionaryWithCapacity:merges.count];
        NSUInteger rank = 0;
        for (id entry in merges) {
            NSString *left = nil, *right = nil;
            if ([entry isKindOfClass:NSArray.class] &&
                [(NSArray *)entry count] == 2) {
                left = ((NSArray *)entry)[0];
                right = ((NSArray *)entry)[1];
            } else if ([entry isKindOfClass:NSString.class]) {
                /* Older exports join the pair with a space. */
                NSRange split = [(NSString *)entry rangeOfString:@" "];
                if (split.location != NSNotFound) {
                    left = [(NSString *)entry substringToIndex:split.location];
                    right = [(NSString *)entry
                             substringFromIndex:split.location + 1];
                }
            }
            if (![left isKindOfClass:NSString.class] ||
                ![right isKindOfClass:NSString.class]) {
                fail(error, error_size, @"malformed merge table");
                return NULL;
            }
            NSString *key = [NSString stringWithFormat:@"%@\x01%@", left, right];
            if (!ranks[key]) ranks[key] = @(rank);
            rank++;
        }
        tokenizer.ranks = ranks;

        sa3_tokenizer *handle = calloc(1, sizeof(*handle));
        if (!handle) {
            fail(error, error_size, @"out of memory");
            return NULL;
        }
        handle->object = (__bridge_retained void *)tokenizer;
        return handle;
    }
}

void sa3_tokenizer_free(sa3_tokenizer *tokenizer) {
    if (!tokenizer) return;
    if (tokenizer->object) {
        SA3Tokenizer *object = (__bridge_transfer SA3Tokenizer *)tokenizer->object;
        object = nil;
    }
    free(tokenizer);
}

/* Splits a character with no vocabulary entry into its UTF-8 bytes. */
static void append_bytes(NSMutableArray<NSString *> *symbols,
                         NSString *character) {
    const char *utf8 = character.UTF8String;
    for (size_t index = 0; utf8 && utf8[index]; index++)
        [symbols addObject:[NSString stringWithFormat:@"<0x%02X>",
                            (unsigned char)utf8[index]]];
}

int sa3_tokenizer_encode(const sa3_tokenizer *tokenizer, const char *utf8,
                         uint32_t *ids, int capacity, int *count,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (count) *count = 0;
    if (!tokenizer || !tokenizer->object || !utf8 || !ids || capacity < 1) {
        fail(error, error_size, @"invalid arguments for the tokenizer");
        return 0;
    }
    @autoreleasepool {
        SA3Tokenizer *object = (__bridge SA3Tokenizer *)tokenizer->object;
        NSString *text = [NSString stringWithUTF8String:utf8];
        if (!text) {
            fail(error, error_size, @"the prompt is not valid UTF-8");
            return 0;
        }
        text = [text stringByReplacingOccurrencesOfString:@" "
                                               withString:SPACE_MARKER];

        /* One symbol per user-perceived character, so combining marks and
         * emoji survive the fallback intact. */
        NSMutableArray<NSString *> *symbols = [NSMutableArray array];
        [text enumerateSubstringsInRange:NSMakeRange(0, text.length)
                                 options:NSStringEnumerationByComposedCharacterSequences
                              usingBlock:^(NSString *piece, NSRange range,
                                           NSRange enclosing, BOOL *stop) {
            (void)range; (void)enclosing; (void)stop;
            if (object.vocab[piece]) {
                [symbols addObject:piece];
            } else {
                append_bytes(symbols, piece);
            }
        }];

        /* Merge the cheapest adjacent pair until none is left. Prompts are a
         * few dozen symbols, so the straightforward scan is ample. */
        while (symbols.count > 1) {
            NSUInteger best = NSNotFound;
            NSUInteger bestRank = NSUIntegerMax;
            for (NSUInteger index = 0; index + 1 < symbols.count; index++) {
                NSString *key = [NSString stringWithFormat:@"%@\x01%@",
                                 symbols[index], symbols[index + 1]];
                NSNumber *rank = object.ranks[key];
                if (rank && rank.unsignedIntegerValue < bestRank) {
                    bestRank = rank.unsignedIntegerValue;
                    best = index;
                }
            }
            if (best == NSNotFound) break;
            NSString *merged = [symbols[best]
                                stringByAppendingString:symbols[best + 1]];
            symbols[best] = merged;
            [symbols removeObjectAtIndex:best + 1];
        }

        int written = 0;
        for (NSString *symbol in symbols) {
            if (written >= capacity) break;
            NSNumber *identifier = object.vocab[symbol];
            if (!identifier) {
                /* A merge can only ever produce entries that exist, so this
                 * means the vocabulary and merges disagree. */
                fail(error, error_size,
                     [NSString stringWithFormat:@"no id for %@", symbol]);
                return 0;
            }
            ids[written++] = (uint32_t)identifier.unsignedIntegerValue;
        }
        if (count) *count = written;
        return 1;
    }
}
