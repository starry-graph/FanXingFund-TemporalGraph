#ifndef __INF_GRAPH_H__
#define __INF_GRAPH_H__

#include<vector>
#include<set>
#include<map>
#include<utility>
#include "genrandom.h"

struct configargs{
	double epsilon,l,wsum,eps_p,lam_p,alv;  // alv: alpha for alpha-T-RIP, ignored for other types
	int k,t,type,evalcnt,checkpre,verbose;
	//type 1: k-R-IMM
	//type 2: alpha-T-RIP
	//type 3: OS-RP
	//type 4: OS-IP
	void showargs(){
		puts("Config:");
		printf("epsilon = %.2f, l = %.2f, k = %d, t = %d, type = %d, wsum = %.2f, evalcnt = %d, verbose = %d\n",epsilon,l,k,t,type,wsum,evalcnt,verbose);
		if (type==2) printf("alpha = %.4f\n",alv);
	}
};

class infgraph{
	std::vector<std::vector<int>> eb,ef;  //backward and forward edges for each node
	std::vector<std::vector<double>> wb,wf; //edge weights

	std::vector<std::map<int,std::vector<int>>> rset;
	std::vector<std::set<int>> bset;

	std::vector<std::pair<int,int>> bres;
	std::vector<int> seeds,st,rv,srv,rcov,vis;
	std::vector<double> thv;
	int n,rsetnum,rsetcov;
	randomgen rg,graph_rg;

	public:
	infgraph(uint32_t randseed, uint32_t graphranseed);
	void addedge(int x,int y);
	int genrip(int v, const configargs &args);
	void buildripset(int num, const configargs &args);
	double immselect(int num, const configargs &args,int ts);
	double osipselect(const configargs &args, const std::vector<int> &rem);
	int getnodesnum();
	void setweights(const configargs &args);
	void genrv(int ts, const configargs &args);
	bool dotimestep(int ts);
	void genthv();
	void insertnewpairs();
	void readseeds(FILE *cof);
	void printans(const configargs &args, FILE *ansf);
	void resetstate(const configargs &args);
	int eval(int setst, const configargs &args);
	int countinf();
	int geth0();
	void resetrset(const configargs &args);
	double getnewcov(const configargs &args);
	void osipchangerem(std::vector<int> &rem);
	double insertnewpair_at(int x);
};

#endif
