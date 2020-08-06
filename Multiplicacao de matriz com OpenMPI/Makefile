# GNU Makefile

CC = mpicc
CCFLAGS = -Wall -O3
LDFLAGS = 
TARGET = main_matrizv3 gera_matrizv3 o_clean help

all: $(TARGET)

%.o: %.c
	$(CC) $(CCFLAGS) -c $<

%: %.o
	$(CC) $(LDFLAGS) $^ -o $@

main_matrizv3: main.c matrizv3.o toolsv3.o matriz-operacoesv3.o
	$(CC) $(CCFLAGS) matriz-operacoesv3.o matrizv3.o toolsv3.o main.c matriz-operacoes-openmpi.c -o $@ $(LDFLAGS)

gera_matrizv3: matrizv3.o toolsv3.o gera_matrizv3.c
	$(CC) $(CCFLAGS) matrizv3.o toolsv3.o gera_matrizv3.c -o $@ $(LDFLAGS)

o_clean:
	rm *.o

help:
	@echo
	@echo
	@echo "####### Exemplo de Execução #######"
	@echo "./main_matrizv3 mat_a3x4.example mat_b4x3.example"
	@echo
	@echo

clean:
	rm -f *~ $(TARGET) *.map *.result
