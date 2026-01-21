/* Minimal miniz_export.h to satisfy miniz.h expectations in-tree. */
#ifndef MINIZ_EXPORT_H
#define MINIZ_EXPORT_H

#if defined(_WIN32) || defined(_WIN64)
#  ifdef MINIZ_DLL_EXPORT
#    define MINIZ_EXPORT __declspec(dllexport)
#  else
#    define MINIZ_EXPORT __declspec(dllimport)
#  endif
#else
#  define MINIZ_EXPORT
#endif

#endif /* MINIZ_EXPORT_H */
