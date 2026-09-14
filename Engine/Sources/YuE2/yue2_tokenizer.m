#include "yue2.h"
#import <Foundation/Foundation.h>
/* Checkpoint-native byte BPE. The rank file and Unicode splitting expression
 * are specified by the pinned public YuE2 release; see NOTICE.md. */
@interface YuE2Tokenizer : NSObject
@property(nonatomic, strong) NSDictionary<NSData *, NSNumber *> *ranks;
@property(nonatomic, strong) NSRegularExpression *split;
@end
@implementation YuE2Tokenizer
@end
struct yue2_tokenizer {
  void *object;
};
yue2_tokenizer *yue2_tokenizer_load(const char *path, char *error,
                                    size_t size) {
  @autoreleasepool {
    NSString *text = [NSString stringWithContentsOfFile:@(path)
                                               encoding:NSUTF8StringEncoding
                                                  error:nil];
    NSMutableDictionary *ranks = [NSMutableDictionary dictionary];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
      NSArray *parts = [line componentsSeparatedByString:@" "];
      if (parts.count != 2)
        continue;
      NSData *bytes = [[NSData alloc] initWithBase64EncodedString:parts[0]
                                                          options:0];
      NSInteger rank = [parts[1] integerValue];
      if (!bytes.length || rank < 0 || rank >= 151643 || ranks[bytes]) {
        snprintf(error, size, "Invalid YuE2 vocabulary");
        return NULL;
      }
      ranks[bytes] = @(rank);
    }
    if (ranks.count != 151643) {
      snprintf(error, size, "Expected 151643 YuE2 text tokens");
      return NULL;
    }
    YuE2Tokenizer *obj = [YuE2Tokenizer new];
    obj.ranks = ranks;
    obj.split = [NSRegularExpression
        regularExpressionWithPattern:
            @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}"
            @"| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                             options:0
                               error:nil];
    yue2_tokenizer *t = calloc(1, sizeof(*t));
    if (!t)
      return NULL;
    t->object = (__bridge_retained void *)obj;
    return t;
  }
}
void yue2_tokenizer_free(yue2_tokenizer *t) {
  if (t) {
    if (t->object) {
      id obj = (__bridge_transfer id)t->object;
      (void)obj;
    }
    free(t);
  }
}
int yue2_tokenize(yue2_tokenizer *t, const char *input, uint32_t **output,
                  char *error, size_t size) {
  *output = NULL;
  @autoreleasepool {
    YuE2Tokenizer *obj = (__bridge YuE2Tokenizer *)t->object;
    NSString *text = [[NSString stringWithUTF8String:input]
        precomposedStringWithCanonicalMapping];
    if (!text) {
      snprintf(error, size, "Invalid UTF-8 music prompt");
      return 0;
    }
    NSMutableData *ids = [NSMutableData data];
    NSArray *matches = [obj.split matchesInString:text
                                          options:0
                                            range:NSMakeRange(0, text.length)];
    for (NSTextCheckingResult *match in matches) {
      NSData *word = [[text substringWithRange:match.range]
          dataUsingEncoding:NSUTF8StringEncoding];
      NSNumber *whole = obj.ranks[word];
      if (whole) {
        uint32_t token = whole.unsignedIntValue;
        [ids appendBytes:&token length:4];
        continue;
      }
      NSMutableArray<NSData *> *pieces = [NSMutableArray array];
      for (NSUInteger i = 0; i < word.length; i++)
        [pieces addObject:[word subdataWithRange:NSMakeRange(i, 1)]];
      while (pieces.count > 1) {
        NSUInteger best = NSNotFound;
        uint32_t rank = UINT32_MAX;
        for (NSUInteger i = 0; i + 1 < pieces.count; i++) {
          NSMutableData *pair = [pieces[i] mutableCopy];
          [pair appendData:pieces[i + 1]];
          NSNumber *r = obj.ranks[pair];
          if (r && r.unsignedIntValue < rank) {
            rank = r.unsignedIntValue;
            best = i;
          }
        }
        if (best == NSNotFound)
          break;
        NSMutableData *pair = [pieces[best] mutableCopy];
        [pair appendData:pieces[best + 1]];
        pieces[best] = pair;
        [pieces removeObjectAtIndex:best + 1];
      }
      for (NSData *piece in pieces) {
        NSNumber *r = obj.ranks[piece];
        if (!r) {
          snprintf(error, size, "Unencodable YuE2 byte sequence");
          return 0;
        }
        uint32_t token = r.unsignedIntValue;
        [ids appendBytes:&token length:4];
      }
    }
    if (!ids.length) {
      snprintf(error, size, "Empty music prompt");
      return 0;
    }
    *output = malloc(ids.length);
    if (!*output)
      return 0;
    memcpy(*output, ids.bytes, ids.length);
    return (int)(ids.length / 4);
  }
}
