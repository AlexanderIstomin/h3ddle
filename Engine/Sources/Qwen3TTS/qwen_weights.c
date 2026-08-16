#include "qwen_weights.h"

#include "h3_safetensors.h"

#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct qwen_weights {
    h3_st_header header;
    void *mapping;
    size_t mapped_size;
    int resident;          /* mapping is malloc'd, not file-backed */
    char *config;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

qwen_weights *qwen_weights_open(const char *path, char *error,
                                size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path) {
        fail(error, error_size, "no weights path");
        return NULL;
    }

    qwen_weights *weights = calloc(1, sizeof(*weights));
    if (!weights) {
        fail(error, error_size, "out of memory");
        return NULL;
    }
    if (!h3_st_read_header(path, &weights->header, error, error_size)) {
        free(weights);
        return NULL;
    }

    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        fail(error, error_size, "cannot open %s", path);
        h3_st_free_header(&weights->header);
        free(weights);
        return NULL;
    }
    /* MAP_PRIVATE so pages are evictable under pressure and never written
     * back, and no MAP_POPULATE: an utterance touches a fraction of the text
     * embedding, and paging the rest in would cost more than it saves. */
    weights->mapped_size = (size_t)weights->header.file_size;
    weights->mapping = mmap(NULL, weights->mapped_size, PROT_READ, MAP_PRIVATE,
                            descriptor, 0);
    close(descriptor);
    if (weights->mapping == MAP_FAILED) {
        fail(error, error_size, "cannot map %s", path);
        h3_st_free_header(&weights->header);
        free(weights);
        return NULL;
    }
    /* Generation re-reads every weight once per frame — gigabytes of it — so
     * how those pages are backed is a throughput question, not just a
     * footprint one. QWEN_RESIDENT_WEIGHTS=1 copies the mapping into
     * anonymous memory to measure the difference against file-backed pages. */
    const char *resident = getenv("QWEN_RESIDENT_WEIGHTS");
    if (resident && resident[0] == '1') {
        void *copy = malloc(weights->mapped_size);
        if (copy) {
            memcpy(copy, weights->mapping, weights->mapped_size);
            munmap(weights->mapping, weights->mapped_size);
            weights->mapping = copy;
            weights->resident = 1;
        }
    }
    return weights;
}

void qwen_weights_close(qwen_weights *weights) {
    if (!weights) return;
    if (weights->resident) {
        free(weights->mapping);
    } else if (weights->mapping && weights->mapping != MAP_FAILED) {
        munmap(weights->mapping, weights->mapped_size);
    }
    h3_st_free_header(&weights->header);
    free(weights->config);
    free(weights);
}

/* Shared lookup: finds the tensor, checks its dtype and shape, and returns a
 * pointer into the mapping. */
static const void *locate(const qwen_weights *weights, const char *name,
                          h3_dtype expected, int ndim, const int64_t *shape,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !name) {
        fail(error, error_size, "no weights");
        return NULL;
    }
    const h3_st_tensor *tensor = h3_st_find(&weights->header, name);
    if (!tensor) {
        fail(error, error_size, "%s is missing", name);
        return NULL;
    }
    if (tensor->dtype != expected) {
        fail(error, error_size, "%s is %s, expected %s", name,
             h3_dtype_name(tensor->dtype), h3_dtype_name(expected));
        return NULL;
    }
    if (ndim >= 0 && tensor->ndim != ndim) {
        fail(error, error_size, "%s has %d dimensions, expected %d", name,
             tensor->ndim, ndim);
        return NULL;
    }
    for (int index = 0; shape && index < ndim; index++) {
        if (shape[index] < 0) continue;
        if ((int64_t)tensor->shape[index] != shape[index]) {
            fail(error, error_size, "%s dimension %d is %llu, expected %lld",
                 name, index, (unsigned long long)tensor->shape[index],
                 (long long)shape[index]);
            return NULL;
        }
    }
    /* file_offset is already absolute: the reader folds the data section's
     * start into it, so data_begin/data_end only give the length. */
    const uint64_t bytes = tensor->data_end - tensor->data_begin;
    if (tensor->file_offset + bytes > weights->mapped_size) {
        fail(error, error_size, "%s runs past the end of the file", name);
        return NULL;
    }
    return (const unsigned char *)weights->mapping + tensor->file_offset;
}

const uint16_t *qwen_weights_bf16(const qwen_weights *weights, const char *name,
                                  int ndim, const int64_t *shape,
                                  char *error, size_t error_size) {
    return locate(weights, name, H3_DTYPE_BF16, ndim, shape, error, error_size);
}

const float *qwen_weights_f32(const qwen_weights *weights, const char *name,
                              int ndim, const int64_t *shape,
                              char *error, size_t error_size) {
    return locate(weights, name, H3_DTYPE_F32, ndim, shape, error, error_size);
}

const uint16_t *qwen_weights_bf16_at(const qwen_weights *weights,
                                     const char *pattern, int index,
                                     int ndim, const int64_t *shape,
                                     char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), pattern, index);
    return qwen_weights_bf16(weights, name, ndim, shape, error, error_size);
}

const float *qwen_weights_f32_at(const qwen_weights *weights,
                                 const char *pattern, int index,
                                 int ndim, const int64_t *shape,
                                 char *error, size_t error_size) {
    char name[256];
    snprintf(name, sizeof(name), pattern, index);
    return qwen_weights_f32(weights, name, ndim, shape, error, error_size);
}

const char *qwen_weights_config(const qwen_weights *weights) {
    return weights ? weights->config : NULL;
}
