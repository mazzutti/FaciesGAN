#include <execinfo.h>
#include <mlx/c/error.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void mlx_error_trace_handler(const char *msg, void *data) {
  (void)data;
  fprintf(stderr, "MLX error: %s\n", msg);
  void *bt[64];
  int bt_size = backtrace(bt, 64);
  char **syms = backtrace_symbols(bt, bt_size);
  if (syms) {
    for (int i = 0; i < bt_size; ++i) {
      fprintf(stderr, "  [%d] %s\n", i, syms[i]);
    }
    free(syms);
  } else {
    backtrace_symbols_fd(bt, bt_size, STDERR_FILENO);
  }
  fflush(stderr);
  exit(-1);
}

__attribute__((constructor)) static void install_mlx_error_trace_handler(void) {
  mlx_set_error_handler(mlx_error_trace_handler, NULL, NULL);
}
