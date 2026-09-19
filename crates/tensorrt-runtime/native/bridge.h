#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Private ABI: all pointers are borrowed except the session returned by open.
typedef struct TrtSession TrtSession;
typedef struct {
    const char* name;
    int32_t dtype;
    int32_t input;
    int32_t rank;
    int64_t dims[8];
} TrtInfo;
typedef struct {
    const char* name;
    const uint8_t* data;
    size_t bytes;
    int32_t dtype;
    int32_t rank;
    int64_t dims[8];
} TrtInput;
const char* vt_error(void);
int32_t vt_version(void);
TrtSession* vt_open(const uint8_t* plan, size_t bytes, int32_t device);
void vt_close(TrtSession* session);
int32_t vt_count(TrtSession* session);
int32_t vt_info(TrtSession* session, int32_t index, TrtInfo* info);
int32_t vt_run(TrtSession* session, const TrtInput* inputs, size_t count, int32_t profile);
int32_t vt_output(TrtSession* session, int32_t index, TrtInfo* info, const uint8_t** data, size_t* bytes);
#ifdef __cplusplus
}
#endif
