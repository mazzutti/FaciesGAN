/* Small header exposing dataset path/name constants used by C code.
 * This mirrors the high-level dataset names managed by Python's
 * datasets/data_files.py so C code can reference the same names.
#ifndef FACIESGAN_C_DATA_FILES_H
#define FACIESGAN_C_DATA_FILES_H

#define DF_BASE_DATA_DIR "data"

/* Subdirectory names */
#define DF_DIR_FACIES "facies"
#define DF_DIR_WELLS "wells"
#define DF_DIR_SEISMIC "seismic"

/* Common file patterns */
#define DF_IMAGE_FILE_PATTERN "*.png"
#define DF_MODEL_FILE_PATTERN "*.pt"
#define DF_MAPPING_FILE_PATTERN "*.npz"

#endif /* FACIESGAN_C_DATA_FILES_H */
