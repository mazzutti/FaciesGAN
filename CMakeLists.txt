cmake_minimum_required(VERSION 3.16)
project(FaciesGAN_MLXC LANGUAGES C CXX)


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
add_library(custom_layer SHARED
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
  faciesgan-c/utils_extra.c
  faciesgan-c/trainning/scheduler.c
  faciesgan-c/trainning/train_manager.c
  faciesgan-c/trainning/autodiff.c
  faciesgan-c/trainning/python_entry.c
  faciesgan-c/trainning/train_utils.c
  faciesgan-c/io/npz_create.c
  faciesgan-c/io/npz_unzip.c
  faciesgan-c/third_party/miniz.c
  faciesgan-c/third_party/miniz_tdef.c
  faciesgan-c/third_party/miniz_tinfl.c
  faciesgan-c/third_party/miniz_zip.c
)

target_include_directories(custom_layer
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/faciesgan-c/models
    ${PROJECT_SOURCE_DIR}/faciesgan-c
)

target_include_directories(custom_layer PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c/third_party)


# If MLXC was found, link against its target `mlxc` (installed by mlx-c)
if(MLXC_FOUND)
  target_link_libraries(custom_layer PRIVATE mlxc)
else()
  # Fallback: link against the in-tree `mlxc` target we added above
  target_link_libraries(custom_layer PRIVATE mlxc)
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

# Small shared library providing the C prefetcher used by Python wrappers.
add_library(facies_prefetcher SHARED
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

# Optionally enable STB-based C generator. To use, place `stb_image.h` and
# `stb_image_resize.h` under `faciesgan-c/third_party` and run cmake with
# `-DENABLE_STB=ON`.
# STB support: headers are expected under `faciesgan-c/third_party` and
# STB-based C generator is enabled by including the third_party headers.
target_include_directories(facies_prefetcher PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c/third_party)

# Add C generator and NPZ unzip helper
target_sources(facies_prefetcher PRIVATE
  faciesgan-c/datasets/c_generator.c
  faciesgan-c/io/npz_unzip.c
  faciesgan-c/io/npz_create.c
  faciesgan-c/datasets/func_cache.c
  faciesgan-c/datasets/mlx_dataset.c
  faciesgan-c/datasets/mlx_pyramids_dataset.c
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
  target_link_libraries(facies_prefetcher PRIVATE mlxc custom_layer)
else()
  target_link_libraries(facies_prefetcher PRIVATE mlxc custom_layer)
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
target_link_libraries(faciesgan PRIVATE facies_prefetcher custom_layer)

# Build C-native trainer
target_sources(faciesgan PRIVATE
  faciesgan-c/trainning/c_trainer.c
)
target_include_directories(faciesgan PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c)

## Optional Python bridge for TensorBoard/background workers
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(Python3_FOUND)
  message(STATUS "Python3 found: enabling Python bridge (pybridge.c)")
  target_sources(faciesgan PRIVATE faciesgan-c/trainning/pybridge.c)
  target_include_directories(faciesgan PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c)
  target_link_libraries(faciesgan PRIVATE Python3::Python)
else()
  message(STATUS "Python3 development not found; building without Python bridge")
endif()

# Install targets

add_executable(gen_cache
  faciesgan-c/tools/gen_cache.c
)
target_include_directories(gen_cache PRIVATE ${PROJECT_SOURCE_DIR}/faciesgan-c ${PROJECT_SOURCE_DIR}/mlx-c)
target_link_libraries(gen_cache PRIVATE mlxc)

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

install(TARGETS facies_prefetcher
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES faciesgan-c/datasets/prefetcher.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/datasets/collate.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/datasets/dataloader.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES faciesgan-c/datasets/mlx_pyramids_dataset.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "Configured FaciesGAN MLX-C build. Use: mkdir build && cd build && cmake .. && make")
