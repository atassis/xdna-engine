// Native-compiled smoke test of fsr1_kernel.cc, run on the host BEFORE the AIE toolchain --
// cheap way to catch logic bugs without a device round trip. Reads/writes raw float32 RGB.
#include <stdio.h>
#include <stdlib.h>

extern void fsr1_strip(const float *in_rgb, float *out_rgb);

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s in.f32 out.f32\n", argv[0]); return 1; }
  FILE *fi = fopen(argv[1], "rb");
  FILE *fo = fopen(argv[2], "wb");
  static float in_buf[4096], out_buf[4096 * 9];
  size_t n = fread(in_buf, sizeof(float), 4096, fi);
  fsr1_strip(in_buf, out_buf);
  fwrite(out_buf, sizeof(float), n * 9, fo);
  fclose(fi); fclose(fo);
  return 0;
}
