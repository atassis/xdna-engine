/* Minimal C ABI over the ONNX Runtime C API, so Rust can run the GigaAM-v3 preprocessor/decoder/
 * joint ONNX graphs via the system libonnxruntime without the (broken-here) `ort` crate. Parallels
 * rust/npu-xrt's XRT shim. dtype: 0 = f32, 1 = i64. Fallible calls return NULL/-1 + set last error. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ShimOrtEnv ShimOrtEnv;
typedef struct ShimOrtSession ShimOrtSession;
typedef struct ShimOrtRun ShimOrtRun; /* holds output tensors alive until freed */

ShimOrtEnv* sort_env_create(void);
void        sort_env_free(ShimOrtEnv*);

ShimOrtSession* sort_session_create(ShimOrtEnv*, const char* model_path);
void            sort_session_free(ShimOrtSession*);

/* Run the session. Inputs given as parallel arrays of length n_in (dtype 0=f32,1=i64).
 * Returns a run handle owning the n_out output tensors (NULL on error). */
ShimOrtRun* sort_run(ShimOrtSession*,
                     int n_in, const char* const* in_names, const void* const* in_data,
                     const int64_t* const* in_dims, const int* in_ndims, const int* in_dtypes,
                     int n_out, const char* const* out_names);
int         sort_run_ndims(ShimOrtRun*, int out_i);                /* -1 on error */
void        sort_run_dims(ShimOrtRun*, int out_i, int64_t* dims);  /* fills `ndims` values */
const void* sort_run_data(ShimOrtRun*, int out_i);                 /* tensor data pointer */
void        sort_run_free(ShimOrtRun*);

const char* sort_last_error(void);

#ifdef __cplusplus
}
#endif
