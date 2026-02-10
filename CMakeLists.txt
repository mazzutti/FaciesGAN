cmake_minimum_required(VERSION 3.16)
project(FaciesGAN_MLXC LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for IDEs (e.g. VSCode/C++ Intellisense)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile_commands.json for editor integration")

# Option to enable AddressSanitizer for debugging memory issues
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Option to enable MLX array memory leak tracking
option(ENABLE_MLX_MEM_DEBUG "Enable MLX array memory leak tracking" OFF)

# Option to enable per-batch memory usage tracking (FG_MEM_DEBUG)
option(ENABLE_FG_MEM_DEBUG "Enable per-batch GPU/RSS memory usage display" OFF)

# Option to enable link-time optimization (LTO/IPO)
option(ENABLE_LTO "Enable link-time optimization (IPO/LTO)" ON)

if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer: enabled (building with -fsanitize=address)")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address)
  else()
    message(WARNING "ENABLE_ASAN set but compiler is not Clang/GNU; skipping sanitizer flags")
  endif()
endif()

if(ENABLE_MLX_MEM_DEBUG)
  message(STATUS "MLX memory debugging: enabled (MLX_MEM_DEBUG defined)")
  add_compile_definitions(MLX_MEM_DEBUG)
endif()

if(ENABLE_FG_MEM_DEBUG)
  message(STATUS "FaciesGAN memory tracking: enabled (FG_MEM_DEBUG defined)")
  add_compile_definitions(FG_MEM_DEBUG)
endif()

if(ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
  if(ipo_supported)
    message(STATUS "LTO/IPO: enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(WARNING "LTO/IPO requested but not supported: ${ipo_error}")
  endif()
endif()

# Option to enable the small C progress bar (fg_progress)
option(ENABLE_FG_PROGRESS "Enable FaciesGAN C progress bar (fg_progress)" ON)

if(ENABLE_FG_PROGRESS)
  message(STATUS "FaciesGAN progress: enabled (FG_PROGRESS defined)")
  add_compile_definitions(FG_PROGRESS)
endif()


# Optionally set MLXC_DIR to point to the installed mlx-c package config
if(NOT DEFINED MLXC_DIR)
  set(MLXC_DIR "" CACHE PATH "Path to MLXC package directory (optional)")
endif()

if(MLXC_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${MLXC_DIR}")
endif()

# Try to find the installed MLXC package (installed by mlx-c)
find_package(MLXC CONFIG QUIET)
if(MLXC_FOUND)
  message(STATUS "Found MLXC package via CMake: enabling MLXC linkage")
else()
  message(STATUS "MLXC package not found via CMake. Building local scaffolding only.")
  # Add the in-tree mlx-c implementation so we can link against its `mlxc` target.
  add_subdirectory(${PROJECT_SOURCE_DIR}/mlx-c)
endif()

# Library target for our custom layers
# NOTE: Must be STATIC (not SHARED) to avoid duplicate MLX singleton instances.
# If this is a SHARED library, the dylib gets its own copy of the MLX Metal
# allocator, causing memory tracking to be split between two allocator instances.
add_library(custom_layer STATIC
  faciesgan-c/models/custom_layer.c
)

target_sources(custom_layer PRIVATE
  faciesgan-c/models/discriminator.c
  faciesgan-c/models/generator.c
  faciesgan-c/models/facies_gan.c
  faciesgan-c/models/base_manager.c
  faciesgan-c/models/base_adapter.c
  faciesgan-c/trainning/optimizer.c
  faciesgan-c/trainning/metrics.c
  faciesgan-c/trainning/train_step.c
  faciesgan-c/trainning/checkpoint.c
  faciesgan-c/trainning/logger.c
  faciesgan-c/options.c
  faciesgan-c/utils.c
  # faciesgan-c/utils_extra.c (merged into utils.c)
  faciesgan-c/trainning/scheduler.c
  faciesgan-c/trainning/array_helpers.c
  faciesgan-c/trainning/scalar_pool.c
  faciesgan-c/trainning/train_manager.c
  faciesgan-c/trainning/mlx_native_grad.c
  faciesgan-c/trainning/python_entry.c
  faciesgan-c/trainning/train_utils.c
  faciesgan-c/trainning/mlx_compat.c
  faciesgan-c/trainning/mem_debug.c
  faciesgan-c/trainning/mlx_error_trace.c
  faciesgan-c/io/npz_create.c
  faciesgan-c/io/npz_unzip.c
  faciesgan-c/third_party/miniz.c
  faciesgan-c/third_party/miniz_tdef.c
  faciesgan-c/third_party/miniz_tinfl.c
  faciesgan-c/third_party/miniz_zip.c
  # faciesgan-c/trainning/c_trainer_api.c
)
# Hot-path files: aggressive optimisation flags (-O3 -march=native -ffast-math)
set_source_files_properties(
  faciesgan-c/utils.c
  faciesgan-c/models/custom_layer.c
  faciesgan-c/models/generator.c
  faciesgan-c/models/facies_gan.c
  faciesgan-c/trainning/optimizer.c
  faciesgan-c/trainning/train_step.c
  faciesgan-c/trainning/mlx_native_grad.c
  PROPERTIES COMPILE_OPTIONS "-O3;-march=native;-ffast-math"
)
set_source_files_properties(faciesgan-c/trainning/mem_debug.c PROPERTIES
  COMPILE_DEFINITIONS MLX_MEM_NO_WRAP=1
)

target_include_directories(custom_layer
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/faciesgan-c/models
    ${PROJECT_SOURCE_DIR}/faciesgan-c
)

target_include_directories(custom_layer PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c/third_party)
target_compile_options(custom_layer PRIVATE -include ${PROJECT_SOURCE_DIR}/faciesgan-c/trainning/mem_debug.h)


# If MLXC was found, link against its target `mlxc` (installed by mlx-c)
if(MLXC_FOUND)
  # MLX linkage is handled at executable level to avoid duplicate static libs
else()
  # Add the in-tree mlx-c top-level directory so headers referenced
  # as <mlx/c/...> resolve correctly (keeps include paths explicit).
  target_include_directories(custom_layer PRIVATE
    ${PROJECT_SOURCE_DIR}/mlx-c
  )
endif()

set_target_properties(custom_layer PROPERTIES
  OUTPUT_NAME custom_layer
  POSITION_INDEPENDENT_CODE ON
)

# Small static library providing the C prefetcher.
# NOTE: Must be STATIC (not SHARED) to avoid duplicate MLX singleton instances.
# If this is a SHARED library, the dylib gets its own copy of the MLX Metal
# allocator, causing memory tracking to be split between two allocator instances.
add_library(facies_prefetcher STATIC
  faciesgan-c/datasets/prefetcher.c
  faciesgan-c/datasets/collate.c
  faciesgan-c/datasets/dataloader.c
  faciesgan-c/datasets/utils.c
)

target_include_directories(facies_prefetcher
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/faciesgan-c
    ${PROJECT_SOURCE_DIR}/faciesgan-c/datasets
)
target_compile_options(facies_prefetcher PRIVATE -include ${PROJECT_SOURCE_DIR}/faciesgan-c/trainning/mem_debug.h)

# Optionally enable STB-based C generator. To use, place `stb_image.h` and
# `stb_image_resize.h` under `faciesgan-c/third_party` and run cmake with
# `-DENABLE_STB=ON`.
# STB support: headers are expected under `faciesgan-c/third_party` and
# STB-based C generator is enabled by including the third_party headers.
target_include_directories(facies_prefetcher PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c/third_party)

# Add NPZ unzip helper and dataset sources
target_sources(facies_prefetcher PRIVATE
  faciesgan-c/io/npz_unzip.c
  faciesgan-c/io/npz_create.c
  faciesgan-c/datasets/func_cache.c
  faciesgan-c/datasets/mlx_dataset.c
  faciesgan-c/datasets/wells.c
)

# Try to find libzip for in-C NPZ handling; optional.
find_path(LIBZIP_INCLUDE_DIR zip.h)
find_library(LIBZIP_LIBRARY NAMES zip libzip)
if(LIBZIP_INCLUDE_DIR AND LIBZIP_LIBRARY)
  message(STATUS "Found libzip; enabling in-C .npz extraction")
  target_include_directories(facies_prefetcher PRIVATE ${LIBZIP_INCLUDE_DIR})
  target_link_libraries(facies_prefetcher PRIVATE ${LIBZIP_LIBRARY})
  target_compile_definitions(facies_prefetcher PRIVATE HAVE_LIBZIP=1)
else()
  message(STATUS "libzip not found; falling back to external unzip for .npz extraction")
endif()

# Vendor miniz (compact wrapper). Compile it into facies_prefetcher so we
# have an in-tree zip extractor even when libzip isn't installed.
# Add full upstream miniz sources for in-process ZIP handling.
target_sources(facies_prefetcher PRIVATE
  faciesgan-c/third_party/miniz.c
  faciesgan-c/third_party/miniz_tdef.c
  faciesgan-c/third_party/miniz_tinfl.c
)

# ZIP read/write support
target_sources(facies_prefetcher PRIVATE faciesgan-c/third_party/miniz_zip.c)
target_include_directories(facies_prefetcher PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c/third_party)

if(MLXC_FOUND)
  target_link_libraries(facies_prefetcher PRIVATE custom_layer)
else()
  target_link_libraries(facies_prefetcher PRIVATE custom_layer)
  target_include_directories(facies_prefetcher PRIVATE ${PROJECT_SOURCE_DIR}/mlx-c)
endif()

set_target_properties(facies_prefetcher PROPERTIES
  OUTPUT_NAME facies_prefetcher
  POSITION_INDEPENDENT_CODE ON
)

# Top-level C launcher (thin port of Python main.py)
add_executable(faciesgan
  faciesgan-c/main.c
)

target_include_directories(faciesgan PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c ${PROJECT_SOURCE_DIR}/mlx-c)
target_link_libraries(faciesgan PRIVATE facies_prefetcher custom_layer mlxc)
target_compile_options(faciesgan PRIVATE -include ${PROJECT_SOURCE_DIR}/faciesgan-c/trainning/mem_debug.h)

# Include the C-native trainer implementation so the executable provides a
# runtime `MLXTrainer_run_full` implementation (used when running the
# dataset-driven trainer from the CLI). The library `custom_layer` contains a
# stub to avoid linking dataset internals into the shared library; the
# executable should supply the real implementation.
target_sources(faciesgan PRIVATE faciesgan-c/trainning/mlx_trainer_api.c)
target_sources(faciesgan PRIVATE faciesgan-c/trainning/mlx_error_trace.c)
target_sources(faciesgan PRIVATE faciesgan-c/trainer.c)
target_sources(faciesgan PRIVATE faciesgan-c/cli/options_builder.c)
target_sources(faciesgan PRIVATE faciesgan-c/cli/cli_args.c)
target_sources(faciesgan PRIVATE faciesgan-c/cli/cli_help.c)
target_sources(faciesgan PRIVATE faciesgan-c/cli/cli_defs.c)

if(ENABLE_FG_PROGRESS)
  target_sources(faciesgan PRIVATE faciesgan-c/trainning/progress.c)
endif()

# Build C-native trainer
# (old `c_trainer.c` removed; functionality provided by `mlx_trainer_api.c`)
target_include_directories(faciesgan PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c)

## Optional Python bridge for TensorBoard/background workers
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python3 found: enabling Python bridge (pybridge.c)")
  target_sources(faciesgan PRIVATE faciesgan-c/trainning/pybridge.c)
  target_include_directories(faciesgan PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c)
  target_link_libraries(faciesgan PRIVATE Python3::Python)
  # Expose Python include dirs to the faciesgan target so IDEs can resolve
  # Python headers for translation units that include <Python.h>.
  if(TARGET Python3::Python)
    if(Python3_INCLUDE_DIRS)
      target_include_directories(faciesgan PRIVATE ${Python3_INCLUDE_DIRS})
    endif()
  endif()

# Install targets

## removed gen_cache executable from top-level CMake (generated code now
## integrated into library). Building the standalone `gen_cache` tool was
## causing build failures when the source file was absent; the lightweight
## generator is available via `gen_cache_generate()` in the C generator.

if(EXISTS "${PROJECT_SOURCE_DIR}/faciesgan-c/tools/c_trainer_smoke.c")
  # Smoke-test executable for the new MLXTrainer API
  add_executable(c_trainer_smoke
    faciesgan-c/tools/c_trainer_smoke.c
  )
  target_include_directories(c_trainer_smoke PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c ${PROJECT_SOURCE_DIR}/mlx-c)
  target_link_libraries(c_trainer_smoke PRIVATE mlxc custom_layer facies_prefetcher)
  target_sources(c_trainer_smoke PRIVATE faciesgan-c/trainning/mlx_trainer_api.c)
  if(TARGET Python3::Python)
    target_sources(c_trainer_smoke PRIVATE faciesgan-c/trainning/pybridge.c)
    target_include_directories(c_trainer_smoke PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c)
    if(Python3_INCLUDE_DIRS)
      target_include_directories(c_trainer_smoke PRIVATE ${Python3_INCLUDE_DIRS})
    endif()
    target_link_libraries(c_trainer_smoke PRIVATE Python3::Python)
  endif()
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/faciesgan-c/tools/prefetcher_test.c")
  # Small test executable for prefetcher iterator
  add_executable(prefetcher_test
    faciesgan-c/tools/prefetcher_test.c
  )
  target_include_directories(prefetcher_test PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c ${PROJECT_SOURCE_DIR}/mlx-c)
  target_link_libraries(prefetcher_test PRIVATE mlxc facies_prefetcher)
  target_link_libraries(prefetcher_test PRIVATE custom_layer)
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/faciesgan-c/tools/compare_direct.c")
  # Direct NPZ -> stacked .npy comparator helper
  add_executable(compare_direct
    faciesgan-c/tools/compare_direct.c
  )
  target_include_directories(compare_direct PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c ${PROJECT_SOURCE_DIR}/mlx-c)
  target_link_libraries(compare_direct PRIVATE mlxc facies_prefetcher)
  target_link_libraries(compare_direct PRIVATE custom_layer)
endif()
 
if(EXISTS "${PROJECT_SOURCE_DIR}/faciesgan-c/tools/dataloader_sampler_test.c")
  # Dataloader sampler test
  add_executable(dataloader_sampler_test
    faciesgan-c/tools/dataloader_sampler_test.c
  )
  target_include_directories(dataloader_sampler_test PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c ${PROJECT_SOURCE_DIR}/mlx-c)
  target_link_libraries(dataloader_sampler_test PRIVATE mlxc facies_prefetcher custom_layer)
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/faciesgan-c/tools/facies_worker.c")
  add_executable(facies_worker
    faciesgan-c/tools/facies_worker.c
  )
  target_include_directories(facies_worker PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c ${PROJECT_SOURCE_DIR}/mlx-c)
  target_link_libraries(facies_worker PRIVATE mlxc facies_prefetcher custom_layer)
endif()
include(GNUInstallDirs)
install(TARGETS custom_layer
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES faciesgan-c/models/custom_layer.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/models/discriminator.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/models/generator.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/models/facies_gan.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/models/base_manager.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/optimizer.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/metrics.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/train_step.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/checkpoint.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/logger.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/scheduler.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/train_manager.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/train_utils.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/trainning/progress.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS facies_prefetcher
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES faciesgan-c/datasets/prefetcher.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/datasets/collate.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/datasets/dataloader.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/datasets/mlx_dataset.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "Configured FaciesGAN MLX-C build. Use: mkdir build && cd build && cmake .. && make")
