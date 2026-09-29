#include <stdio.h>
extern void fsr1_easu_only(const float *in_rgb, float *out_rgb);
int main(int argc, char **argv) {
  FILE *fi = fopen(argv[1], "rb");
  FILE *fo = fopen(argv[2], "wb");
  static float in_buf[4096], out_buf[4096*9];
  size_t n = fread(in_buf, sizeof(float), 4096, fi);
  fsr1_easu_only(in_buf, out_buf);
  fwrite(out_buf, sizeof(float), n*9, fo);
  return 0;
}
