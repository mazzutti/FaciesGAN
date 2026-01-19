#ifndef PYBRIDGE_H
#define PYBRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

int pybridge_initialize(void);
int pybridge_finalize(void);

int pybridge_create_visualizer(int num_scales, const char *output_dir, const char *log_dir, int update_interval);
int pybridge_update_visualizer_from_json(int epoch, const char *metrics_json, int samples_processed);
int pybridge_close_visualizer(void);

int pybridge_create_background_worker(int max_workers, int max_pending);
int pybridge_background_pending_count(void);
int pybridge_shutdown_background_worker(int wait);

#ifdef __cplusplus
}
#endif

#endif
