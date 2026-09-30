// SPDX-License-Identifier: Apache-2.0
// LD_PRELOAD read recorder for scripts/buildstore/record.py; line format is documented there.
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

static int logfd = -1;

static void emit2(char kind, const char *a, const char *b) {
  if (logfd < 0 || !a) return;
  char buf[2 * PATH_MAX + 8];
  int n = b ? snprintf(buf, sizeof buf, "%c %s\t%s\n", kind, a, b)
            : snprintf(buf, sizeof buf, "%c %s\n", kind, a);
  if (n > 0 && n < (int)sizeof buf) syscall(SYS_write, logfd, buf, (size_t)n);
}

static int fd_path(int fd, char *out, size_t cap) {
  char link[64];
  snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
  ssize_t n = readlink(link, out, cap - 1);
  if (n <= 0) return -1;
  out[n] = 0;
  return out[0] == '/' ? 0 : -1;  // pipe:[..], socket:[..], anon_inode:..
}

static void absolute(int dirfd, const char *p, char *out, size_t cap) {
  if (p[0] == '/') { snprintf(out, cap, "%s", p); return; }
  char base[PATH_MAX];
  if (dirfd == AT_FDCWD ? !getcwd(base, sizeof base) : fd_path(dirfd, base, sizeof base) < 0)
    base[0] = 0;
  snprintf(out, cap, "%s/%s", base, p);
}

static void on_open(int fd, int dirfd, const char *given, int writing) {
  int saved = errno;
  if (fd >= 0) {
    char real[PATH_MAX], abs[PATH_MAX];
    if (fd_path(fd, real, sizeof real) == 0) {
      struct stat st;
      int isdir = fstat(fd, &st) == 0 && S_ISDIR(st.st_mode);
      emit2(isdir ? 'D' : (writing ? 'W' : 'R'), real, NULL);
      absolute(dirfd, given, abs, sizeof abs);
      if (strcmp(abs, real) != 0) emit2('L', abs, real);
    }
  } else if (saved == ENOENT) {
    char abs[PATH_MAX];
    absolute(dirfd, given, abs, sizeof abs);
    emit2('A', abs, NULL);
  }
  errno = saved;
}

static void on_probe(int r, int dirfd, const char *p) {
  int saved = errno;
  if (r < 0 && saved == ENOENT) {
    char abs[PATH_MAX];
    absolute(dirfd, p, abs, sizeof abs);
    emit2('A', abs, NULL);
  }
  errno = saved;
}

static void scan_maps(void) {
  FILE *(*real_fopen)(const char *, const char *) = dlsym(RTLD_NEXT, "fopen");
  FILE *f = real_fopen("/proc/self/maps", "re");
  if (!f) return;
  char line[PATH_MAX + 128];
  while (fgets(line, sizeof line, f)) {
    char *p = strchr(line, '/');
    if (!p) continue;
    p[strcspn(p, "\n")] = 0;
    if (strstr(p, " (deleted)")) continue;
    emit2('M', p, NULL);
  }
  fclose(f);
}

__attribute__((constructor)) static void rec_init(void) {
  const char *dir = getenv("REC_DIR");
  if (!dir) return;
  char path[PATH_MAX];
  snprintf(path, sizeof path, "%s/%d.log", dir, (int)getpid());
  logfd = (int)syscall(SYS_openat, AT_FDCWD, path,
                       O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
  scan_maps();
}

__attribute__((destructor)) static void rec_fini(void) { scan_maps(); }

#define WRITING(fl) (((fl) & (O_WRONLY | O_RDWR | O_CREAT | O_TRUNC)) != 0)
#define NEXT(name) static __typeof__(name) *real; if (!real) real = dlsym(RTLD_NEXT, #name)

#define OPEN_HOOK(name)                                                     \
  int name(const char *p, int fl, ...) {                                    \
    NEXT(name);                                                             \
    mode_t m = 0;                                                           \
    if (fl & (O_CREAT | O_TMPFILE)) { va_list a; va_start(a, fl); m = va_arg(a, mode_t); va_end(a); } \
    int fd = real(p, fl, m);                                                \
    on_open(fd, AT_FDCWD, p, WRITING(fl));                                  \
    return fd;                                                              \
  }
OPEN_HOOK(open)
OPEN_HOOK(open64)

#define OPENAT_HOOK(name)                                                   \
  int name(int d, const char *p, int fl, ...) {                             \
    NEXT(name);                                                             \
    mode_t m = 0;                                                           \
    if (fl & (O_CREAT | O_TMPFILE)) { va_list a; va_start(a, fl); m = va_arg(a, mode_t); va_end(a); } \
    int fd = real(d, p, fl, m);                                             \
    on_open(fd, d, p, WRITING(fl));                                         \
    return fd;                                                              \
  }
OPENAT_HOOK(openat)
OPENAT_HOOK(openat64)

#define FOPEN_HOOK(name)                                                    \
  FILE *name(const char *p, const char *mode) {                             \
    NEXT(name);                                                             \
    FILE *f = real(p, mode);                                                \
    on_open(f ? fileno(f) : -1, AT_FDCWD, p, strpbrk(mode, "wa+") != NULL); \
    return f;                                                              \
  }
FOPEN_HOOK(fopen)
FOPEN_HOOK(fopen64)

DIR *opendir(const char *p) {
  NEXT(opendir);
  DIR *d = real(p);
  on_open(d ? dirfd(d) : -1, AT_FDCWD, p, 0);
  return d;
}

DIR *fdopendir(int fd) {
  NEXT(fdopendir);
  DIR *d = real(fd);
  char path[PATH_MAX];
  if (d && fd_path(fd, path, sizeof path) == 0) emit2('D', path, NULL);
  return d;
}

int stat(const char *p, struct stat *s) { NEXT(stat); int r = real(p, s); on_probe(r, AT_FDCWD, p); return r; }
int lstat(const char *p, struct stat *s) { NEXT(lstat); int r = real(p, s); on_probe(r, AT_FDCWD, p); return r; }
int access(const char *p, int m) { NEXT(access); int r = real(p, m); on_probe(r, AT_FDCWD, p); return r; }
int fstatat(int d, const char *p, struct stat *s, int fl) {
  NEXT(fstatat); int r = real(d, p, s, fl); on_probe(r, d, p); return r;
}
// libpython3.14 imports these, not the names above (nm -D, measured 2026-09-30).
int stat64(const char *p, struct stat64 *s) { NEXT(stat64); int r = real(p, s); on_probe(r, AT_FDCWD, p); return r; }
int lstat64(const char *p, struct stat64 *s) { NEXT(lstat64); int r = real(p, s); on_probe(r, AT_FDCWD, p); return r; }
int fstatat64(int d, const char *p, struct stat64 *s, int fl) {
  NEXT(fstatat64); int r = real(d, p, s, fl); on_probe(r, d, p); return r;
}
