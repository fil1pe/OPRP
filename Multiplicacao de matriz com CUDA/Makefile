# GNU Makefile

CC = nvcc
CCFLAGS =
LDFLAGS =
TARGET = main o_clean help

all: $(TARGET)

%.o: %.cu
	$(CC) $(CCFLAGS) -c $< -lm

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@

o_clean:
	rm *.o

help:
	@echo
	@echo
	@echo "####### Exemplo de Execução GPU #######"
	@echo "./main"
	@echo
	@echo

clean:
	rm -f *~ $(TARGET)