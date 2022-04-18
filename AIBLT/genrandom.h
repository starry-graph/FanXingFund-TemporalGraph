#ifndef __GEN_RANDOM_H__
#define __GEN_RANDOM_H__

//#include<random>
#include "sfmt/SFMT.h"


class randomgen{
	sfmt_t st;
	public:
	randomgen(uint32_t sd);
	int getint(int n);
	double getreal();
};

#endif