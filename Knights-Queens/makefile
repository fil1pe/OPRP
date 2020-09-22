# GNU Makefile

CC = nvcc
CCFLAGS =
LDFLAGS =
TARGET = main o_clean

all: $(TARGET)

%.o: %.cu
	$(CC) $(CCFLAGS) -c $< -lm

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@

o_clean:
	rm *.o

clean:
	rm -f *~ $(TARGET)