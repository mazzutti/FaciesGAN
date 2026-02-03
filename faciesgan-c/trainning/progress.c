#include "progress.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>

static size_t fg_total = 0;
static const char *fg_desc = NULL;
static int fg_width = 0;

int fg_progress_init(const char *desc, size_t total, int width) {
    fg_total = total;
    fg_desc = desc;
    fg_width = width;
    return 0;
}

static int terminal_width(void) {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
        return w.ws_col;
    }
    return 80;
}

void fg_progress_update(size_t done) {
    if (fg_total == 0) return;
    int tw = fg_width;
    if (tw <= 0) {
        tw = terminal_width() - 40;
        if (tw < 10) tw = 10;
    }
    double ratio = (double)done / (double)fg_total;
    if (ratio > 1.0) ratio = 1.0;
    int filled = (int)(ratio * tw);
    int percent = (int)(ratio * 100.0);
    if (fg_desc)
        fprintf(stdout, "\r%s ", fg_desc);
    else
        fprintf(stdout, "\r");
    fprintf(stdout, "[%.*s%*s] %3d%% (%zu/%zu)   \r",
            filled, "################################################################", tw-filled, "",
            percent, done, fg_total);
    fflush(stdout);
}

void fg_progress_finish(void) {
    if (fg_total == 0) return;
    fg_progress_update(fg_total);
    fprintf(stdout, "\n");
    fflush(stdout);
}
