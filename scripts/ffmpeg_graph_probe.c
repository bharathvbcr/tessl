// Black-box probe through the public libavfilter / libswscale API (no FFmpeg source read):
// does a graph "format=rgb24" (torchcodec's filtergraph for widths that are not multiples of
// 32 and for any transform: a buffer source given only format/size/aspect/time base, a plain
// buffer sink, the frame carrying its own colorspace and range) give the bytes of the ffmpeg
// CLI's rgb24 conversion, and the bytes of sws_scale with flags 0?
//
//   graph_probe FMT W H MATRIX RANGE IN.planar OUT_GRAPH.rgb OUT_SWS.rgb [TRANSFORMS]
//
// TRANSFORMS, if given, is appended to the graph after "format=rgb24," (e.g. transpose=cclock);
// the graph's output size is then the transformed one.
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *what, int code) {
    char buf[128] = "";
    if (code < 0) av_strerror(code, buf, sizeof buf);
    fprintf(stderr, "%s: %s\n", what, buf);
    exit(2);
}

int main(int argc, char **argv) {
    if (argc < 9) die("usage", 0);
    enum AVPixelFormat fmt = av_get_pix_fmt(argv[1]);
    int w = atoi(argv[2]), h = atoi(argv[3]);
    enum AVColorSpace cs = av_color_space_from_name(argv[4]);
    enum AVColorRange range = strcmp(argv[5], "pc") == 0 ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    const char *transforms = argc > 9 ? argv[9] : "";
    if (fmt == AV_PIX_FMT_NONE || cs == AVCOL_SPC_UNSPECIFIED) die("format or matrix name", 0);

    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    int bytes = d->comp[0].depth > 8 ? 2 : 1;
    int cw = AV_CEIL_RSHIFT(w, d->log2_chroma_w), ch = AV_CEIL_RSHIFT(h, d->log2_chroma_h);

    AVFrame *f = av_frame_alloc();
    f->format = fmt; f->width = w; f->height = h;
    f->colorspace = cs; f->color_range = range;
    f->pts = 0;
    if (av_frame_get_buffer(f, 0) < 0) die("frame buffer", -1);
    FILE *in = fopen(argv[6], "rb");
    if (!in) die("open planar input", 0);
    for (int p = 0; p < 3; p++) {
        int pw = (p ? cw : w) * bytes, ph = p ? ch : h;
        for (int r = 0; r < ph; r++)
            if (fread(f->data[p] + (size_t)r * f->linesize[p], 1, pw, in) != (size_t)pw) die("short planar input", 0);
    }
    fclose(in);

    // torchcodec's graph: buffer source with format/size/aspect/time base only, then the parsed
    // filters string, then a plain buffer sink.
    AVFilterGraph *g = avfilter_graph_alloc();
    AVFilterContext *src = avfilter_graph_alloc_filter(g, avfilter_get_by_name("buffer"), "in");
    AVBufferSrcParameters *sp = av_buffersrc_parameters_alloc();
    sp->format = fmt; sp->width = w; sp->height = h;
    sp->sample_aspect_ratio = (AVRational){1, 1};
    sp->time_base = (AVRational){1, 25};
    if (av_buffersrc_parameters_set(src, sp) < 0) die("buffersrc params", -1);
    av_free(sp);
    int r = avfilter_init_str(src, NULL);
    if (r < 0) die("init src", r);
    AVFilterContext *sink = NULL;
    r = avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "out", NULL, NULL, g);
    if (r < 0) die("create sink", r);
    AVFilterInOut *outs = avfilter_inout_alloc(), *ins = avfilter_inout_alloc();
    outs->name = av_strdup("in"); outs->filter_ctx = src; outs->pad_idx = 0; outs->next = NULL;
    ins->name = av_strdup("out"); ins->filter_ctx = sink; ins->pad_idx = 0; ins->next = NULL;
    char desc[256];
    snprintf(desc, sizeof desc, "format=rgb24%s%s", transforms[0] ? "," : "", transforms);
    r = avfilter_graph_parse_ptr(g, desc, &ins, &outs, NULL);
    if (r < 0) die("parse graph", r);
    r = avfilter_graph_config(g, NULL);
    if (r < 0) die("configure graph", r);
    r = av_buffersrc_write_frame(src, f);
    if (r < 0) die("write frame", r);
    AVFrame *o = av_frame_alloc();
    r = av_buffersink_get_frame(sink, o);
    if (r < 0) die("get frame", r);
    if (o->format != AV_PIX_FMT_RGB24) die("graph did not yield rgb24", 0);
    FILE *out = fopen(argv[7], "wb");
    for (int y = 0; y < o->height; y++) fwrite(o->data[0] + (size_t)y * o->linesize[0], 1, (size_t)o->width * 3, out);
    fclose(out);
    printf("graph %dx%d -> %dx%d\n", w, h, o->width, o->height);

    // sws_scale with flags 0, the frame's matrix and range set the way a caller would.
    if (!transforms[0]) {
        struct SwsContext *c = sws_getContext(w, h, fmt, w, h, AV_PIX_FMT_RGB24, 0, NULL, NULL, NULL);
        if (!c) die("sws_getContext", 0);
        int sws_cs = cs == AVCOL_SPC_SMPTE170M || cs == AVCOL_SPC_BT470BG ? 5 : (int)cs;
        sws_setColorspaceDetails(c, sws_getCoefficients(sws_cs), range == AVCOL_RANGE_JPEG, sws_getCoefficients(5), 1, 0, 1 << 16, 1 << 16);
        uint8_t *rgb = malloc((size_t)w * h * 3);
        uint8_t *dst[1] = {rgb};
        int dst_stride[1] = {w * 3};
        sws_scale(c, (const uint8_t *const *)f->data, f->linesize, 0, h, dst, dst_stride);
        out = fopen(argv[8], "wb");
        fwrite(rgb, 1, (size_t)w * h * 3, out);
        fclose(out);
        sws_freeContext(c);
        free(rgb);
    }
    return 0;
}
