SRC=np1
SRC1=matmul
polly_path=/home/delaram/Desktop/llvm-project/build
opt=$(polly_path)/bin/opt
lib=$(polly_path)/lib/LLVMPolly.so

#LD_LIBRARY_PATH= $(polly_path)/lib


all:
#	$(polly_path)/bin/clang -emit-llvm -S computation_functions.c -o computation.ll
	$(polly_path)/bin/clang -S -emit-llvm -Xclang -disable-O0-optnone -fno-discard-value-names   $(SRC).c -o $(SRC).ll
#	$(polly_path)/bin/llvm-link -S computation.ll $(SRC).ll > $(SRC).ll
	$(opt) -S -load $(lib) -polly-canonicalize -polly-prepare -instcombine -simplifycfg $(SRC).ll -o $(SRC).preopt.ll
	$(opt) -S -load $(lib) -polly-detect -stats  $(SRC).preopt.ll -o $(SRC).scheduled.ll --polly-process-unprofitable
	$(opt) -S -load $(lib)  -polly-pipeline-loops=yes -polly-opt-isl $(SRC).preopt.ll -polly-ast -polly-pipeline=true -polly-codegen -polly-pipeline-build=yes -polly-pipeline-codegen=yes  -o $(SRC).scheduled.ll

run:
	$(polly_path)/bin/clang  -O1 -S -emit-llvm functions_no_block_test.c -o functions.ll -fopenmp -I $(polly_path)/projects/openmp/runtime/src
	$(polly_path)/bin/llvm-link -S functions.ll $(SRC).scheduled.ll > final.ll
	
	$(opt) -S -load $(lib) -instcombine -simplifycfg final.ll -o final.ll
	$(polly_path)/bin/clang -O3 -Wno-everything final.ll -o param.bin -lgmp -lm -fopenmp=libiomp5 -L $(polly_path)/lib 
#	 ./param.bin 

orig:
	$(polly_path)/bin/clang $(SRC).c -O3 -o orig.bin  -fopenmp=libiomp5 -lgmp -lm
#	./orig.bin 

parallel:
	$(polly_path)/bin/clang -S -emit-llvm -Xclang -disable-O0-optnone -fno-discard-value-names   $(SRC).c -o $(SRC).ll
	$(opt) -S -load $(lib) -polly-canonicalize -polly-prepare  -instcombine -simplifycfg $(SRC).ll -o $(SRC).preopt.ll
	$(opt) -S -load $(lib)  -polly-opt-isl $(SRC).preopt.ll -polly-parallel=true -polly-ast -polly-codegen  -o $(SRC).scheduled.ll
	$(polly_path)/bin/clang $(SRC).scheduled.ll -o par.bin -lgomp -lgmp
#	./par.bin

clean:
	rm orig*
	rm param*
	rm *.ll
	rm *.bin
	rm *.txt
