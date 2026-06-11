// C ABI shim over the ONNX Runtime C API (see onnx_shim.h). Uses the system libonnxruntime.
#include "onnx_shim.h"
#include "onnxruntime_c_api.h"
#include <string>
#include <vector>

static const OrtApi* ort() {
    static const OrtApi* a = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return a;
}
static thread_local std::string g_err;
const char* sort_last_error(void) { return g_err.c_str(); }

// returns true on success; on failure stashes the message and releases the status
static bool ok(OrtStatus* s) {
    if (s) {
        g_err = ort()->GetErrorMessage(s);
        ort()->ReleaseStatus(s);
        return false;
    }
    return true;
}

struct ShimOrtEnv { OrtEnv* env; };
struct ShimOrtSession { OrtSession* sess; };
struct ShimOrtRun { std::vector<OrtValue*> outs; };

ShimOrtEnv* sort_env_create(void) {
    OrtEnv* e = nullptr;
    if (!ok(ort()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "npu-asr", &e))) return nullptr;
    return new ShimOrtEnv{e};
}
void sort_env_free(ShimOrtEnv* e) {
    if (e) { ort()->ReleaseEnv(e->env); delete e; }
}

ShimOrtSession* sort_session_create(ShimOrtEnv* env, const char* model_path) {
    OrtSessionOptions* opt = nullptr;
    if (!ok(ort()->CreateSessionOptions(&opt))) return nullptr;
    // tiny models — a 1-thread intra-op pool avoids contending with the encoder's rayon glue
    ort()->SetIntraOpNumThreads(opt, 1);
    OrtSession* s = nullptr;
    bool good = ok(ort()->CreateSession(env->env, model_path, opt, &s));
    ort()->ReleaseSessionOptions(opt);
    if (!good) return nullptr;
    return new ShimOrtSession{s};
}
void sort_session_free(ShimOrtSession* s) {
    if (s) { ort()->ReleaseSession(s->sess); delete s; }
}

ShimOrtRun* sort_run(ShimOrtSession* sess,
                     int n_in, const char* const* in_names, const void* const* in_data,
                     const int64_t* const* in_dims, const int* in_ndims, const int* in_dtypes,
                     int n_out, const char* const* out_names) {
    OrtMemoryInfo* mem = nullptr;
    if (!ok(ort()->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem))) return nullptr;

    std::vector<OrtValue*> ins(n_in, nullptr);
    bool good = true;
    for (int i = 0; i < n_in && good; i++) {
        size_t nel = 1;
        for (int d = 0; d < in_ndims[i]; d++) nel *= (size_t)in_dims[i][d];
        bool i64 = in_dtypes[i] == 1;
        ONNXTensorElementDataType dt = i64 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
                                           : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        size_t bytes = nel * (i64 ? 8 : 4);
        good = ok(ort()->CreateTensorWithDataAsOrtValue(
            mem, const_cast<void*>(in_data[i]), bytes, in_dims[i], (size_t)in_ndims[i], dt, &ins[i]));
    }

    ShimOrtRun* run = nullptr;
    if (good) {
        std::vector<OrtValue*> outs(n_out, nullptr);
        good = ok(ort()->Run(sess->sess, nullptr, in_names, ins.data(), (size_t)n_in,
                             out_names, (size_t)n_out, outs.data()));
        if (good) {
            run = new ShimOrtRun();
            run->outs = outs;
        } else {
            for (auto* o : outs) if (o) ort()->ReleaseValue(o);
        }
    }
    for (auto* v : ins) if (v) ort()->ReleaseValue(v);
    ort()->ReleaseMemoryInfo(mem);
    return run;
}

int sort_run_ndims(ShimOrtRun* r, int i) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (!ok(ort()->GetTensorTypeAndShape(r->outs[i], &info))) return -1;
    size_t n = 0;
    ort()->GetDimensionsCount(info, &n);
    ort()->ReleaseTensorTypeAndShapeInfo(info);
    return (int)n;
}
void sort_run_dims(ShimOrtRun* r, int i, int64_t* dims) {
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (!ok(ort()->GetTensorTypeAndShape(r->outs[i], &info))) return;
    size_t n = 0;
    ort()->GetDimensionsCount(info, &n);
    ort()->GetDimensions(info, dims, n);
    ort()->ReleaseTensorTypeAndShapeInfo(info);
}
const void* sort_run_data(ShimOrtRun* r, int i) {
    void* p = nullptr;
    ok(ort()->GetTensorMutableData(r->outs[i], &p));
    return p;
}
void sort_run_free(ShimOrtRun* r) {
    if (r) {
        for (auto* o : r->outs) if (o) ort()->ReleaseValue(o);
        delete r;
    }
}
