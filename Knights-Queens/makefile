# GNU Makefile

CC = nvcc
CCFLAGS =
LDFLAGS =
TARGET = main o_clean

all: $(TARGET)

%.o: %.cu
	$(CC) $(CCFLAGS) -c $<

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@

main: main.cu wtime.o knights.o queens.o board.o
	$(CC) $(CCFLAGS) wtime.o board.o queens.o knights.o main.cu -o $@

o_clean:
	rm *.o

clean:
	rm -f *~ $(TARGET)