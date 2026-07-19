/*
 * NPU super-resolution video filter (vf_xdna_sr).
 *
 * A thin libavfilter adapter over the npu-sr engine C ABI (libxdna_sr.so). Per frame it hands the
 * RGB24 pixels to the engine, which does Y-only SR on the AMD XDNA2 NPU + bicubic chroma, and forwards
 * the enlarged frame. Decode/encode stay ffmpeg's job; this filter only upscales.
 *
 * Written against FFmpeg 8.0 (FFFilter API). Delivered as an out-of-tree patch (ffmpeg has no stable
 * filter-plugin ABI) -- see ffmpeg/apply.sh. Upstreaming is a separate, owner-gated act.
 *
 * This file is part of the xdna-engine project. AGPL-3.0 (matches the engine); when contributed to
 * FFmpeg it would carry FFmpeg's LGPL header instead.
 */
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "avfilter.h"
#include "filters.h"
#include "video.h"
#include "xdna_sr.h"

typedef struct XdnaSrContext {
    const AVClass *class;
    char *schedule;   /* path to <net>.json */
    int   npu;        /* use the NPU frontier (1) or force CPU (0) */
    XdnaSr *eng;
    int   scale;
} XdnaSrContext;

static av_cold int init(AVFilterContext *ctx)
{
    XdnaSrContext *s = ctx->priv;
    s->eng = xdna_sr_create(s->schedule, s->npu);
    if (!s->eng) {
        av_log(ctx, AV_LOG_ERROR, "xdna_sr_create failed: %s\n", xdna_sr_last_error());
        return AVERROR(EINVAL);
    }
    s->scale = xdna_sr_scale(s->eng);
    if (s->scale < 1) {
        av_log(ctx, AV_LOG_ERROR, "xdna_sr_scale failed: %s\n", xdna_sr_last_error());
        return AVERROR(EINVAL);
    }
    av_log(ctx, AV_LOG_INFO, "xdna_sr: schedule=%s npu=%d scale=%d\n", s->schedule, s->npu, s->scale);
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    XdnaSrContext *s = ctx->priv;
    if (s->eng) {
        xdna_sr_free(s->eng);
        s->eng = NULL;
    }
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    XdnaSrContext *s = ctx->priv;
    AVFilterLink *inlink = ctx->inputs[0];
    outlink->w = inlink->w * s->scale;
    outlink->h = inlink->h * s->scale;
    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    XdnaSrContext *s = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int w = in->width, h = in->height;
    int ow = outlink->w, oh = outlink->h;
    int ret = 0;

    AVFrame *out = ff_get_video_buffer(outlink, ow, oh);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    /* Pack input to tightly-packed RGB24 (the C ABI expects linesize == w*3). */
    uint8_t *packed_in = av_malloc((size_t)w * h * 3);
    uint8_t *packed_out = av_malloc((size_t)ow * oh * 3);
    if (!packed_in || !packed_out) {
        ret = AVERROR(ENOMEM);
        goto done;
    }
    for (int y = 0; y < h; y++)
        memcpy(packed_in + (size_t)y * w * 3, in->data[0] + (size_t)y * in->linesize[0], (size_t)w * 3);

    size_t got_w = 0, got_h = 0;
    if (xdna_sr_process_rgb8(s->eng, packed_in, w, h, packed_out, (size_t)ow * oh * 3, &got_w, &got_h) != 0) {
        av_log(ctx, AV_LOG_ERROR, "xdna_sr_process_rgb8 failed: %s\n", xdna_sr_last_error());
        ret = AVERROR(EINVAL);
        goto done;
    }

    /* Unpack into the output frame, respecting its linesize. */
    for (int y = 0; y < oh; y++)
        memcpy(out->data[0] + (size_t)y * out->linesize[0], packed_out + (size_t)y * ow * 3, (size_t)ow * 3);

done:
    av_free(packed_in);
    av_free(packed_out);
    av_frame_free(&in);
    if (ret < 0) {
        av_frame_free(&out);
        return ret;
    }
    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(XdnaSrContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
static const AVOption xdna_sr_options[] = {
    { "schedule", "path to the net schedule <net>.json", OFFSET(schedule),
      AV_OPT_TYPE_STRING, { .str = "artifacts/espcn/espcn.json" }, 0, 0, FLAGS },
    { "npu", "use the NPU frontier (0 forces CPU)", OFFSET(npu),
      AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(xdna_sr);

static const AVFilterPad xdna_sr_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
};

static const AVFilterPad xdna_sr_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_output,
    },
};

const FFFilter ff_vf_xdna_sr = {
    .p.name        = "xdna_sr",
    .p.description = NULL_IF_CONFIG_SMALL("Super-resolution upscaling on the AMD XDNA2 NPU."),
    .p.priv_class  = &xdna_sr_class,
    .priv_size     = sizeof(XdnaSrContext),
    .init          = init,
    .uninit        = uninit,
    FILTER_INPUTS(xdna_sr_inputs),
    FILTER_OUTPUTS(xdna_sr_outputs),
    FILTER_SINGLE_PIXFMT(AV_PIX_FMT_RGB24),
};
