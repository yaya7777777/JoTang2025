#指定编译器
CC = gcc

#编译选项
CFLAGS = -g -O2 -Wall -Werror


# 头文件包含路径
INCLUDES = -I./include
#源文件目录
SRCS = hello.c lib/add.c lib/mu.c
#目标文件目录
OBJS = $(SRCS:.c=.o)
#可执行文件名
TARGET = demo.out
#编译规则
all: $(TARGET)
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDES)
#编译源文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDES)
#清理目标文件和可执行文件
clean:
	del /Q /F hello.o lib\add.o lib\mu.o demo.out

