#include <stdio.h>
#include "custom_layer.h"

int main(void) {
    printf("Creating color quantization module...\n");
    MLXColorQuantization *q = mlx_colorquant_create(0.1f);
    if (!q) {
        fprintf(stderr, "Failed to create module\n");
        return 1;
    }

    /* In a full integration you'd pass a real mlx_array_t pointer here.
       We just exercise create/free to ensure the library compiles and links. */
    mlx_colorquant_free(q);
    printf("OK\n");
    return 0;
}
