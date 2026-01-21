#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pybridge.h"

#ifdef __APPLE__
// macOS: ensure we include the Python headers from the system or virtualenv
#endif

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
/* Stubs when Python dev headers are not available. */
int pybridge_initialize(void) { return 0; }
int pybridge_finalize(void) { return 0; }
int pybridge_create_visualizer(int num_scales, const char *output_dir, const char *log_dir, int update_interval)
{
    (void)num_scales;
    (void)output_dir;
    (void)log_dir;
    (void)update_interval;
    return 0;
}
int pybridge_update_visualizer_from_json(int epoch, const char *metrics_json, int samples_processed)
{
    (void)epoch;
    (void)metrics_json;
    (void)samples_processed;
    return -1;
}
int pybridge_close_visualizer(void) { return 0; }
int pybridge_create_background_worker(int max_workers, int max_pending)
{
    (void)max_workers;
    (void)max_pending;
    return 0;
}
int pybridge_background_pending_count(void) { return 0; }
int pybridge_shutdown_background_worker(int wait)
{
    (void)wait;
    return 0;
}
#else

static PyObject *g_visualizer = NULL;
static PyObject *g_background_worker = NULL;

int pybridge_initialize(void)
{
    if (Py_IsInitialized())
        return 1;

    Py_Initialize();
    if (!Py_IsInitialized())
        return 0;

    /* Ensure current working directory is on sys.path so project modules import */
    PyRun_SimpleString("import sys; sys.path.insert(0, '.')\n");
    return 1;
}

int pybridge_finalize(void)
{
    if (!Py_IsInitialized())
        return 0;

    /* Clean up any created objects */
    if (g_visualizer)
    {
        Py_DECREF(g_visualizer);
        g_visualizer = NULL;
    }
    if (g_background_worker)
    {
        Py_DECREF(g_background_worker);
        g_background_worker = NULL;
    }

    Py_Finalize();
    return 1;
}

int pybridge_create_visualizer(int num_scales, const char *output_dir, const char *log_dir, int update_interval)
{
    if (!Py_IsInitialized())
        if (!pybridge_initialize())
            return 0;

    PyObject *mod = PyImport_ImportModule("tensorboard_visualizer");
    if (!mod)
    {
        PyErr_Print();
        return 0;
    }

    PyObject *cls = PyObject_GetAttrString(mod, "TensorBoardVisualizer");
    Py_DECREF(mod);
    if (!cls || !PyCallable_Check(cls))
    {
        Py_XDECREF(cls);
        PyErr_Print();
        return 0;
    }

    PyObject *args = Py_BuildValue("iszi", num_scales, output_dir ? output_dir : "", log_dir ? log_dir : "", update_interval);
    /* BuildValue with format 'iszi' isn't valid; construct tuple instead */
    Py_DECREF(args);

    PyObject *pargs = PyTuple_New(4);
    PyTuple_SetItem(pargs, 0, PyLong_FromLong((long)num_scales));
    PyTuple_SetItem(pargs, 1, PyUnicode_FromString(output_dir ? output_dir : ""));
    PyTuple_SetItem(pargs, 2, PyUnicode_FromString(log_dir ? log_dir : ""));
    PyTuple_SetItem(pargs, 3, PyLong_FromLong((long)update_interval));

    PyObject *inst = PyObject_CallObject(cls, pargs);
    Py_DECREF(cls);
    Py_DECREF(pargs);

    if (!inst)
    {
        PyErr_Print();
        return 0;
    }

    Py_XDECREF(g_visualizer);
    g_visualizer = inst; /* steal reference */
    return 1;
}

int pybridge_update_visualizer_from_json(int epoch, const char *metrics_json, int samples_processed)
{
    if (!Py_IsInitialized() || !g_visualizer || !metrics_json)
        return 0;

    PyObject *json_mod = PyImport_ImportModule("json");
    if (!json_mod)
    {
        PyErr_Print();
        return 0;
    }
    PyObject *loads = PyObject_GetAttrString(json_mod, "loads");
    Py_DECREF(json_mod);
    if (!loads || !PyCallable_Check(loads))
    {
        Py_XDECREF(loads);
        PyErr_Print();
        return 0;
    }

    PyObject *py_json_str = PyUnicode_FromString(metrics_json);
    PyObject *metrics = PyObject_CallFunctionObjArgs(loads, py_json_str, NULL);
    Py_DECREF(py_json_str);
    Py_DECREF(loads);
    if (!metrics)
    {
        PyErr_Print();
        return 0;
    }

    /* Call visualizer.update(epoch, metrics, None, samples_processed) */
    PyObject *none = Py_None;
    Py_INCREF(none);
    PyObject *res = PyObject_CallMethod(g_visualizer, "update", "iOOi", epoch, metrics, none, samples_processed);
    Py_DECREF(none);
    Py_DECREF(metrics);
    if (!res)
    {
        PyErr_Print();
        return 0;
    }
    Py_DECREF(res);
    return 1;
}

int pybridge_close_visualizer(void)
{
    if (!Py_IsInitialized() || !g_visualizer)
        return 0;

    PyObject *res = PyObject_CallMethod(g_visualizer, "close", NULL);
    if (!res)
    {
        PyErr_Print();
        return 0;
    }
    Py_DECREF(res);
    Py_DECREF(g_visualizer);
    g_visualizer = NULL;
    return 1;
}

int pybridge_create_background_worker(int max_workers, int max_pending)
{
    if (!Py_IsInitialized())
        if (!pybridge_initialize())
            return 0;

    PyObject *mod = PyImport_ImportModule("background_workers");
    if (!mod)
    {
        PyErr_Print();
        return 0;
    }
    PyObject *cls = PyObject_GetAttrString(mod, "BackgroundWorker");
    Py_DECREF(mod);
    if (!cls || !PyCallable_Check(cls))
    {
        Py_XDECREF(cls);
        PyErr_Print();
        return 0;
    }

    PyObject *pargs = PyTuple_New(2);
    PyTuple_SetItem(pargs, 0, PyLong_FromLong((long)max_workers));
    PyTuple_SetItem(pargs, 1, PyLong_FromLong((long)max_pending));

    PyObject *inst = PyObject_CallObject(cls, pargs);
    Py_DECREF(cls);
    Py_DECREF(pargs);
    if (!inst)
    {
        PyErr_Print();
        return 0;
    }

    Py_XDECREF(g_background_worker);
    g_background_worker = inst;
    return 1;
}

int pybridge_background_pending_count(void)
{
    if (!Py_IsInitialized() || !g_background_worker)
        return -1;

    PyObject *res = PyObject_CallMethod(g_background_worker, "pending_count", NULL);
    if (!res)
    {
        PyErr_Print();
        return -1;
    }
    if (!PyLong_Check(res))
    {
        Py_DECREF(res);
        return -1;
    }
    long val = PyLong_AsLong(res);
    Py_DECREF(res);
    return (int)val;
}

int pybridge_shutdown_background_worker(int wait)
{
    if (!Py_IsInitialized() || !g_background_worker)
        return 0;

    PyObject *res = PyObject_CallMethod(g_background_worker, "shutdown", "p", wait ? 1 : 0);
    if (!res)
    {
        PyErr_Print();
        return 0;
    }
    Py_DECREF(res);
    Py_DECREF(g_background_worker);
    g_background_worker = NULL;
    return 1;
}

#endif
