# Makefile to build a shared library for the mlx-c custom layers
# Adjust MLX_INC and MLX_LIB to match your system's mlx-c installation paths

MLX_INC ?= /usr/local/include
MLX_LIB ?= /usr/local/lib
CC ?= gcc
CFLAGS ?= -O2 -fPIC -I$(MLX_INC)
LDFLAGS ?= -shared -L$(MLX_LIB) -lmlx_c

TARGET = libcustom_layer.so
SRCS = custom_layer.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c custom_layer.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
