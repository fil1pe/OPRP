# GNU Makefile
# Revisão: ago/2019

CC = gcc
CCFLAGS = -Wall -O3
LDFLAGS = -pthread
TARGET = main gmat help

all: $(TARGET)

%.o: %.c
	$(CC) $(CCFLAGS) -c $<

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@

main: main.c matrizv3.o toolsv3.o matriz-operacoesv3.o
		$(CC) $(CCFLAGS) matriz-operacoesv3.o matrizv3.o toolsv3.o main.c matriz-operacoes-pthread.c -o $@ $(LDFLAGS)

gmat: matrizv3.o toolsv3.o gera_matrizv3.c
		$(CC) $(CCFLAGS) matrizv3.o toolsv3.o gera_matrizv3.c -o $@ $(LDFLAGS)

help:
	@echo
	@echo
	@echo "####### Exemplo de Execução #######"
	@echo "./main mat_a3x4.example mat_b4x3.example 4"

clean:
	rm -f *.o *~ $(TARGET) *.map *.result
