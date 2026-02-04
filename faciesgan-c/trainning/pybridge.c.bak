/* Consolidated pybridge: stubs and Python-backed implementation in one file.
 * Detection uses __has_include(<Python.h>) so IDEs/CMake can compile the
 * appropriate branch even when Python headers are not available. */

#include "pybridge.h"
#include "trainning/common_helpers.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>

#ifdef __has_include
#if __has_include(<Python.h>)
#include <Python.h>
#define HAVE_PYTHON_API 1
#else
#define HAVE_PYTHON_API 0
#endif
#else
#define HAVE_PYTHON_API 0
#endif

#if !HAVE_PYTHON_API
/* Stubs when Python headers aren't available. */
int pybridge_initialize(void) {
    return 0;
}
int pybridge_finalize(void) {
    return 0;
}
int pybridge_create_visualizer(int num_scales, const char *output_dir,
                               const char *log_dir, int update_interval) {
    /* stub: parameters unused without Python support */
    return 0;
}
int pybridge_update_visualizer_from_json(int epoch, const char *metrics_json,
        int samples_processed) {
    /* stub: parameters unused without Python support */
    return 0;
}
int pybridge_close_visualizer(void) {
    return 0;
}
int pybridge_create_background_worker(int max_workers, int max_pending) {
    /* stub: parameters unused without Python support */
    return 0;
}
int pybridge_background_pending_count(void) {
    return -1;
}
int pybridge_shutdown_background_worker(int wait) {
    /* stub: parameter unused without Python support */
    return 0;
}

int pybridge_submit_plot_generated_facies(const char *fake_path,
        const char *real_path, int stage,
        int index, const char *out_dir,
        const char *masks_path) {
    /* stub: parameters unused without Python support */
    return 0;
}

#else

/* Real implementation using the Python C API. */
static PyObject *g_visualizer = NULL;
static PyObject *g_background_worker = NULL;

static void pybridge_print_py_err(void) {
    if (PyErr_Occurred())
        PyErr_Print();
}

int pybridge_initialize(void) {
    if (Py_IsInitialized())
        return 1;

    /* Set up Python configuration before initialization to use venv if available */
    const char *venv = getenv("VIRTUAL_ENV");
    char venv_abs[PATH_MAX];
    if (!venv || !venv[0]) {
        /* Prefer local .venv if present */
        if (access(".venv/bin/python", X_OK) == 0) {
            venv = ".venv";
        }
    }
    if (venv && venv[0]) {
        const char *venv_res = venv;
        if (realpath(venv, venv_abs) != NULL) {
            venv_res = venv_abs;
        }
        venv = venv_res;
        setenv("VIRTUAL_ENV", venv, 1);
    }
    if (venv && venv[0]) {
        /* Prefer classic APIs to avoid getpath.py failures with venv embeds. */
        char py_path[PATH_MAX];
        snprintf(py_path, sizeof(py_path), "%s/bin/python", venv);
        static wchar_t py_w[PATH_MAX];
        mbstowcs(py_w, py_path, PATH_MAX - 1);
        Py_SetProgramName(py_w);
        Py_Initialize();
    } else {
        Py_Initialize();
    }

    if (!Py_IsInitialized())
        return 0;

    /* Add current directory to Python path */
    PyRun_SimpleString("import sys; sys.path.insert(0, '.')\n");

    /* Ensure venv site-packages is in path and sys.executable is correct */
    if (venv && venv[0]) {
        PyRun_SimpleString(
            "import sys, os, site\n"
            "venv = os.environ.get('VIRTUAL_ENV', '')\n"
            "if venv:\n"
            "    py = os.path.join(venv, 'bin', 'python')\n"
            "    if os.path.exists(py):\n"
            "        sys.executable = py\n"
            "        sys._base_executable = py\n"
            "        os.environ['PYTHONEXECUTABLE'] = py\n"
            "    # Add venv site-packages to path\n"
            "    import glob\n"
            "    sp = glob.glob(os.path.join(venv, 'lib', 'python*', 'site-packages'))\n"
            "    for p in sp:\n"
            "        if p not in sys.path:\n"
            "            sys.path.insert(0, p)\n"
            "    # Refresh site to pick up .pth files\n"
            "    site.main()\n"
        );
    }

    /* Ensure multiprocessing spawn uses a real python executable.
     * Without this, embedded Python may default to the C binary, causing
     * errors like "unrecognized option -c" in child processes. */
    PyRun_SimpleString(
        "import os, sys, shutil\n"
        "py = getattr(sys, 'executable', '') or ''\n"
        "venv = os.environ.get('VIRTUAL_ENV', '')\n"
        "if venv:\n"
        "    vpy = os.path.join(venv, 'bin', 'python')\n"
        "    if os.path.exists(vpy):\n"
        "        py = vpy\n"
        "if not py or os.path.basename(py) in ('faciesgan', 'faciesgan.exe'):\n"
        "    py = shutil.which('python3') or shutil.which('python') or py\n"
        "if py:\n"
        "    sys.executable = py\n"
        "    os.environ['PYTHONEXECUTABLE'] = py\n"
        "    try:\n"
        "        import multiprocessing as _mp\n"
        "        _mp.set_executable(py)\n"
        "        try:\n"
        "            import multiprocessing.spawn as _s\n"
        "            _s._python_exe = py.encode()\n"
        "        except Exception:\n"
        "            pass\n"
        "    except Exception:\n"
        "        pass\n"
    );

    /* Also set multiprocessing start method to 'spawn' for safety */
    PyRun_SimpleString(
        "import multiprocessing\n"
        "try:\n"
        "    multiprocessing.set_start_method('spawn', force=True)\n"
        "except RuntimeError:\n"
        "    pass\n"
    );

    return 1;
}

int pybridge_finalize(void) {
    if (!Py_IsInitialized())
        return 0;
    if (g_visualizer) {
        Py_DECREF(g_visualizer);
        g_visualizer = NULL;
    }
    if (g_background_worker) {
        Py_DECREF(g_background_worker);
        g_background_worker = NULL;
    }
    Py_Finalize();
    return 1;
}

int pybridge_create_visualizer(int num_scales, const char *output_dir,
                               const char *log_dir, int update_interval) {
    if (!Py_IsInitialized() && !pybridge_initialize())
        return 0;
    PyObject *mod = PyImport_ImportModule("tensorboard_visualizer");
    if (!mod) {
        pybridge_print_py_err();
        return 0;
    }
    PyObject *cls = PyObject_GetAttrString(mod, "TensorBoardVisualizer");
    Py_DECREF(mod);
    if (!cls || !PyCallable_Check(cls)) {
        Py_XDECREF(cls);
        pybridge_print_py_err();
        return 0;
    }
    PyObject *pargs = PyTuple_New(4);
    PyTuple_SetItem(pargs, 0, PyLong_FromLong((long)num_scales));
    PyTuple_SetItem(pargs, 1, PyUnicode_FromString(output_dir ? output_dir : ""));
    PyTuple_SetItem(pargs, 2, PyUnicode_FromString(log_dir ? log_dir : ""));
    PyTuple_SetItem(pargs, 3, PyLong_FromLong((long)update_interval));
    PyObject *inst = PyObject_CallObject(cls, pargs);
    Py_DECREF(cls);
    Py_DECREF(pargs);
    if (!inst) {
        pybridge_print_py_err();
        return 0;
    }
    Py_XDECREF(g_visualizer);
    g_visualizer = inst;
    return 1;
}

int pybridge_update_visualizer_from_json(int epoch, const char *metrics_json,
        int samples_processed) {
    if (!Py_IsInitialized() || !g_visualizer || !metrics_json)
        return 0;
    PyObject *json_mod = PyImport_ImportModule("json");
    if (!json_mod) {
        pybridge_print_py_err();
        return 0;
    }
    PyObject *loads = PyObject_GetAttrString(json_mod, "loads");
    Py_DECREF(json_mod);
    if (!loads || !PyCallable_Check(loads)) {
        Py_XDECREF(loads);
        pybridge_print_py_err();
        return 0;
    }
    PyObject *py_json_str = PyUnicode_FromString(metrics_json);
    PyObject *metrics = PyObject_CallFunctionObjArgs(loads, py_json_str, NULL);
    Py_DECREF(py_json_str);
    Py_DECREF(loads);
    if (!metrics) {
        pybridge_print_py_err();
        return 0;
    }
    PyObject *none = Py_None;
    Py_INCREF(none);
    PyObject *res = PyObject_CallMethod(g_visualizer, "update", "iOOi", epoch,
                                        metrics, none, samples_processed);
    Py_DECREF(none);
    Py_DECREF(metrics);
    if (!res) {
        pybridge_print_py_err();
        return 0;
    }
    Py_DECREF(res);
    return 1;
}

int pybridge_close_visualizer(void) {
    if (!Py_IsInitialized() || !g_visualizer)
        return 0;
    PyObject *res = PyObject_CallMethod(g_visualizer, "close", NULL);
    if (!res) {
        pybridge_print_py_err();
        return 0;
    }
    Py_DECREF(res);
    Py_DECREF(g_visualizer);
    g_visualizer = NULL;
    return 1;
}

int pybridge_create_background_worker(int max_workers, int max_pending) {
    if (!Py_IsInitialized() && !pybridge_initialize())
        return 0;
    PyObject *mod = PyImport_ImportModule("background_workers");
    if (!mod) {
        pybridge_print_py_err();
        return 0;
    }
    PyObject *cls = PyObject_GetAttrString(mod, "BackgroundWorker");
    Py_DECREF(mod);
    if (!cls || !PyCallable_Check(cls)) {
        Py_XDECREF(cls);
        pybridge_print_py_err();
        return 0;
    }
    PyObject *pargs = PyTuple_New(2);
    PyTuple_SetItem(pargs, 0, PyLong_FromLong((long)max_workers));
    PyTuple_SetItem(pargs, 1, PyLong_FromLong((long)max_pending));
    PyObject *inst = PyObject_CallObject(cls, pargs);
    Py_DECREF(cls);
    Py_DECREF(pargs);
    if (!inst) {
        pybridge_print_py_err();
        return 0;
    }
    Py_XDECREF(g_background_worker);
    g_background_worker = inst;
    return 1;
}

int pybridge_background_pending_count(void) {
    if (!Py_IsInitialized() || !g_background_worker)
        return -1;
    PyObject *res =
        PyObject_CallMethod(g_background_worker, "pending_count", NULL);
    if (!res) {
        pybridge_print_py_err();
        return -1;
    }
    if (!PyLong_Check(res)) {
        Py_DECREF(res);
        return -1;
    }
    long val = PyLong_AsLong(res);
    Py_DECREF(res);
    return (int)val;
}

int pybridge_shutdown_background_worker(int wait) {
    if (!Py_IsInitialized() || !g_background_worker) {
        return 0;
    }
    PyObject *wait_obj = wait ? Py_True : Py_False;
    PyObject *res =
        PyObject_CallMethod(g_background_worker, "shutdown", "O", wait_obj);
    if (!res) {
        pybridge_print_py_err();
        return 0;
    }
    Py_DECREF(res);
    Py_DECREF(g_background_worker);
    g_background_worker = NULL;
    return 1;
}

int pybridge_submit_plot_generated_facies(const char *fake_path,
        const char *real_path, int stage,
        int index, const char *out_dir,
        const char *masks_path) {
    if (!Py_IsInitialized() || !g_background_worker) {
        return 0;
    }
    /* Call submit_plot_generated_facies_from_npy on the background worker.
     * Signature: (fake_path, real_path, stage, index, out_dir, masks_path) */
    PyObject *res = PyObject_CallMethod(
                        g_background_worker, "submit_plot_generated_facies_from_npy", "ssiiss",
                        fake_path ? fake_path : "",
                        real_path ? real_path : "",
                        stage,
                        index,
                        out_dir ? out_dir : "",
                        masks_path ? masks_path : ""
                    );
    if (!res) {
        pybridge_print_py_err();
        return 0;
    }
    Py_DECREF(res);
    return 1;
}

#endif /* HAVE_PYTHON_API */
