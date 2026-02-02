#ifndef MLX_C_AUTODIFF_H
#define MLX_C_AUTODIFF_H

#include <stddef.h>
#include "facies_gan.h"
#include "generator.h"
#include "discriminator.h"
#include <mlx/c/ops.h>

typedef struct AGValue AGValue;
typedef struct AGNode AGNode;

/* Create an AGValue that wraps an existing mlx_array (does not take ownership). */
AGValue *ag_value_from_array(mlx_array *arr, int requires_grad);
void ag_value_free(AGValue *v);

/* Create an AGValue that takes ownership of the provided mlx_array. Useful
	for wrapping temporary arrays created during forward passes so the autodiff
	runtime can free them later. */
AGValue *ag_value_from_new_array(mlx_array *arr, int requires_grad);

/* Register an AGValue as a temporary value to be freed by `ag_reset_tape()`
	(used for wrappers created inside forward helpers). */
void ag_register_temp_value(AGValue *v);

/* Access underlying array */
mlx_array *ag_value_array(AGValue *v);

/* Create constant scalar value */
AGValue *ag_scalar_float(float f);

/* Basic ops (record on global tape) */
AGValue *ag_add(AGValue *a, AGValue *b);
AGValue *ag_sub(AGValue *a, AGValue *b);
AGValue *ag_mul(AGValue *a, AGValue *b);
AGValue *ag_tanh(AGValue *a);
AGValue *ag_square(AGValue *a);
AGValue *ag_sum_axis(AGValue *a, int axis, int keepdims);
/* Mean over all elements, returns scalar. Backward broadcasts grad/size to input shape. */
AGValue *ag_mean(AGValue *a);
/* Sum over all elements, returns scalar. Backward broadcasts grad (ones) to input shape. */
AGValue *ag_sum(AGValue *a);
AGValue *ag_transpose(AGValue *a);
AGValue *ag_matmul(AGValue *a, AGValue *b);
AGValue *ag_divide(AGValue *a, AGValue *b);
AGValue *ag_sqrt(AGValue *a);

/* Create an AGValue containing ones with the same shape as the given AGValue's array. */
AGValue *ag_ones_like(AGValue *a);

/* AG-compatible LeakyReLU: returns piecewise x for x>=0 else negative_slope*x.
	Implemented using sqrt(x^2) to avoid needing a separate abs/where op. */
AGValue *ag_leaky_relu(AGValue *a, float negative_slope);

/* Convolution wrapper; records op with stride/padding/group parameters */
AGValue *ag_conv2d(AGValue *input, AGValue *weight, int stride0, int stride1, int pad0, int pad1, int dilation0, int dilation1, int groups);

/* Transposed convolution AG op (used for conv backward in create_graph mode) */
AGValue *ag_conv_transpose2d(AGValue *input, AGValue *weight, int stride0, int stride1, int pad0, int pad1, int dilation0, int dilation1, int output_pad0, int output_pad1, int groups);

/* Backprop that builds a differentiable graph of gradients (create_graph=True).
	It populates AGValue->grad_ag fields for values reachable from `output`.
	After calling this, you can use those `grad_ag` AGValues in further AG ops
	to construct higher-order losses, then call the regular `ag_backward` on the
	final scalar loss to compute parameter gradients. */
int ag_backward_create_graph(AGValue *output);

/* Backpropagate from given value (compute grads for requires_grad values). */
int ag_backward(AGValue *output);

/* Utility: zero gradients, get grad array pointer (caller owns copy) */
mlx_array *ag_value_get_grad(AGValue *v);
void ag_zero_grad_all(void);

/* Clear all grad_ag pointers (call after create-graph backward to allow
   subsequent regular backwards on the same tape). */
void ag_clear_grad_ag_all(void);

/* Return AGValue representing symbolic gradient produced by create_graph pass.
	Caller must not free the returned AGValue (it's owned by tape temporaries). */
AGValue *ag_value_get_grad_ag(AGValue *v);

/* Upsample (nearest/repeat) to target spatial dims (NHWC). Only integer
	scaling factors are supported. */
AGValue *ag_upsample(AGValue *a, int out_h, int out_w, const char *mode, int align_corners);

/* Slice: similar to mlx_slice. start/stop arrays are pointers to ints of length `ndim`.
	keepdims is ignored; implement slicing by copying underlying mlx_slice result. */
AGValue *ag_slice(AGValue *a, const int *start, const int *stop, int ndim);

/* Pad: pad along given axes (axes array length = n_axes), low_pad/high_pad arrays length = n_axes */
AGValue *ag_pad(AGValue *a, const int *axes, int n_axes, const int *low_pad, int low_len, const int *high_pad, int high_len, float pad_val, const char *mode);

/* Tile: repeat along each axis; reps length = ndim */
AGValue *ag_tile(AGValue *a, const int *reps, int ndim);

/* Concatenate: concatenate an array of AGValue* along axis. `parts` is array of length n_parts. */
AGValue *ag_concatenate(AGValue **parts, int n_parts, int axis);

/* Collect grads for an array of AGValue parameters.
	Allocates *out_grads as an array of mlx_array* (caller must free each and the array).
	Returns 0 on success. */
int ag_collect_grads(AGValue **params, int n, mlx_array ***out_grads);

/* Reset and free the tape (frees AGNode/AGValue structures). */
void ag_reset_tape(void);

#endif
