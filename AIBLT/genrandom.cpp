#include "genrandom.h"
using namespace std;

randomgen::randomgen(uint32_t sd) {
	sfmt_init_gen_rand(&st,sd);
}

int randomgen::getint(int n){
	return sfmt_genrand_uint32(&st) % n;
}

double randomgen::getreal(){
	return sfmt_genrand_real1(&st);
}
