//! Frame geometry types shared by the closed FSR1 backends and their cfg-independent callers
//! (`npu-sr-capi`). Kept always-compiled (unlike `fsr1_frame`/`fsr1_rt`) so the public API surface
//! does not change when those backends are absent.

/// The padded-frame geometry a `layout: "frame"` design expects. A zero-copy caller (dma-buf) builds
/// its own input/output buffers to this exact shape -- `xdna_sr_frame_layout` in the C ABI.
#[derive(Debug, Clone, Copy)]
pub struct FrameLayout {
    pub in_w: usize,
    pub in_h: usize,
    pub in_pad_w: usize,
    pub in_pad_h: usize,
    pub in_pad_x: usize,
    pub in_pad_y: usize,
    pub out_pad_w: usize,
    pub out_pad_h: usize,
    pub scale: usize,
}

/// Padded buffers a zero-copy caller builds for a one-design (`fsr1_rt`) engine: the input is the
/// frame at (in_pad_x, in_pad_y) of an in_pad_w x in_pad_h plane, edge-replicated around it; the
/// output's valid out_w x out_h is at the top left of an out_pad_w x out_pad_h plane. 1 byte per
/// pixel (the NV12 Y plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtLayout {
    pub in_w: usize,
    pub in_h: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub in_pad_w: usize,
    pub in_pad_h: usize,
    pub in_pad_x: usize,
    pub in_pad_y: usize,
    pub out_pad_w: usize,
    pub out_pad_h: usize,
    pub scale_num: usize,
    pub scale_den: usize,
}
