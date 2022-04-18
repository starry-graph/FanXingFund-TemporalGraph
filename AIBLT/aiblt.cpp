#include<cstdio>
#include<cstring>
#include<cmath>
#include<cassert>
#include<algorithm>
#include<climits>
#include<ctime>
#include<string>
#include "infgraph.h"
#include "genrandom.h"
using namespace std;

typedef long long ll;

double sqr(double x){
	return x*x;
}

double log2(int n){
	return log(n) / log(2);
}

double logcnk(int n, int k) {
	double ans = 0;
	for (int i = n - k + 1; i <= n; i++){
		ans += log(i);
	}
	for (int i = 1; i <= k; i++){
		ans -= log(i);
	}
	return ans;
}

double opt_est(infgraph &g, const configargs &args, int ts){
	int n=g.getnodesnum(), lim=ceil(log2(n-1));
	for (int i = 1; i <= lim ; i++){
		double x = n / pow(2.0, i);
		ll the_i = ceil(args.lam_p / x);
		assert(the_i <= INT_MAX);
		g.buildripset(the_i,args);
		
		double cov = g.immselect(args.k,args,ts);
		if (n*cov > (1+args.eps_p)/x){  // note this line was changed from original
			double OPT_prime = n*cov / (1+args.eps_p);
			return OPT_prime;
		}
	}
	printf("**WARNING** fail to find LB at the end of the for loop, returning LB=1\n");
	return 1;
}

double opt_est_osip(infgraph &g, const configargs &args, const vector<int> &rem){
	int n=g.getnodesnum(), lim=ceil(log2(n-1));
	for (int i = 1; i <= lim ; i++){
		double x = n / pow(2.0, i);
		ll the_i = ceil(args.lam_p / x);
		assert(the_i <= INT_MAX);
		g.buildripset(the_i,args);
		
		double cov = g.osipselect(args,rem);
		if (n*cov > (1+args.eps_p)/x){  // note this line was changed from original
			double OPT_prime = n*cov / (1+args.eps_p);
			return OPT_prime;
		}
	}
	printf("**WARNING** fail to find LB at the end of the for loop, returning LB=1\n");
	return 1;
}

double solve_krimm(infgraph &g, configargs &args, FILE* af){
	int n=g.getnodesnum();
	double res=0;
	args.checkpre=1;
	args.l+=log(2*args.t)/log(n);
	double eps_0=exp(1-exp(-1))*args.epsilon/2;
	args.eps_p=sqrt(2)*eps_0;
	double alpha=sqrt(args.l*log(n) + log(2) + log(args.t));
	double beta=sqrt((1-exp(-1))*logcnk(n, args.k)+sqr(alpha));
	args.lam_p=(2 + 2*args.eps_p/3)*(args.l * log(n) + log(args.t) + logcnk(n, args.k) + log(log2(n))) * n / sqr(args.eps_p);
	double lam_star=2 * n * sqr((1-exp(-1)) * alpha + beta) / sqr(eps_0);

	for (int reps=0; reps<args.evalcnt; reps++){
		g.resetstate(args);
		g.genthv();  //generate a LT instance (threshold values)
		
		int i,ores=g.eval(0,args);
		for (i=1;i<=args.t;i++){
			if (args.verbose){
				printf("** Time step %d/%d\n",i,args.t);
			}
			g.genrv(i,args);
			double lb = opt_est(g,args,i);
			ll theta = ceil(lam_star / lb);
			assert(theta <= INT_MAX);
			g.buildripset(theta,args);
			g.immselect(args.k,args,i);
			g.insertnewpairs();
			if (!g.dotimestep(i)) break;
		}
		// simulate to the end
		if (i>args.t){
			while (g.dotimestep(i)) i++;
		}
		int nres=g.countinf();
		double rr=(double)(ores-nres)/ores;
		res+=rr;
		printf("Test #%d: No imm-set: %d, with imm-set: %d, reduction ratio: %.6f\n",reps,ores,nres,rr);fflush(stdout);
		if (args.verbose){
			g.printans(args,af);
		}
	}
	
	return res/args.evalcnt;
}

double solve_atrip(infgraph &g, configargs &args, FILE* af){
	int n=g.getnodesnum();
	long long tot_rem=0;
	double res=0,lim,opt;
	args.checkpre=0;
	for (int reps=0; reps<args.evalcnt; reps++){
		g.resetstate(args);
		g.genthv();  //generate a LT instance (threshold values)
		
		int i,ores=g.eval(0,args);
		int h0,rem=args.k*args.t;
		for (i=1;i<=args.t;i++){
			if (args.verbose){
				printf("** Time step %d/%d\n",i,args.t);
			}
			g.genrv(i,args);
			if (i==1){
				h0=g.geth0();
				lim=(1-args.alv)*h0;
				opt=(2+args.epsilon)*h0*log(2*rem*pow(n,args.l+1))/sqr(args.epsilon);
			}
			g.resetrset(args);
			g.buildripset(opt,args);
			double fm=g.getnewcov(args);
			g.immselect(rem,args,i);
			int cur=0;
			while (rem>0){
				if (n * fm <= lim && rem>0){
					fm = g.insertnewpair_at(cur);
					if (fm<0) break;
					rem--;
					cur++;
					continue;
				}
				break;
			}
			if (!g.dotimestep(i)) break;
		}
		// simulate to the end
		if (i>args.t){
			while (g.dotimestep(i)) i++;
		}
		int nres=g.countinf();
		double rr=(double)(ores-nres)/ores;
		res+=rr;
		printf("Test #%d: No imm-set: %d, with imm-set: %d, reduction ratio: %.6f, remaining budget: %d\n",reps,ores,nres,rr,rem);fflush(stdout);
		tot_rem+=rem;
		if (args.verbose){
			g.printans(args,af);
		}
	}
	printf("----> Average remaining budget: %.6f\n",(double)tot_rem/args.evalcnt);
	return res/args.evalcnt;
}

double solve_osrp(infgraph &g, configargs &args, FILE* af){
	int n=g.getnodesnum();
	double res=0;
	args.checkpre=1;
	args.l+=log(2*args.t)/log(n);
	double eps_0=exp(1-exp(-1))*args.epsilon/2;
	args.eps_p=sqrt(2)*eps_0;
	double alpha=sqrt(args.l*log(n) + log(2) + log(args.t));
	double beta=sqrt((1-exp(-1))*logcnk(n, args.k)+sqr(alpha));
	args.lam_p=(2 + 2*args.eps_p/3)*(args.l * log(n) + log(args.t) + logcnk(n, args.k) + log(log2(n))) * n / sqr(args.eps_p);
	double lam_star=2 * n * sqr((1-exp(-1)) * alpha + beta) / sqr(eps_0);

	for (int reps=0; reps<args.evalcnt; reps++){
		g.resetstate(args);
		g.genthv();  //generate a LT instance (threshold values)

		int i,ores=g.eval(0,args);
		g.genrv(1,args);
		for (i=1;i<=args.t;i++){
			if (args.verbose){
				printf("** Time step %d/%d\n",i,args.t);
			}
			double lb = opt_est(g,args,i);
			ll theta = ceil(lam_star / lb);
			assert(theta <= INT_MAX);
			g.buildripset(theta,args);
			g.immselect(args.k,args,i);
			g.insertnewpairs();
		}
		int nres=g.eval(1,args);
		double rr=(double)(ores-nres)/ores;
		res+=rr;
		printf("Test #%d: No imm-set: %d, with imm-set: %d, reduction ratio: %.6f\n",reps,ores,nres,rr);fflush(stdout);
		if (args.verbose){
			g.printans(args,af);
		}
	}
	
	return res/args.evalcnt;
}

double solve_osip(infgraph &g, configargs &args, FILE* af){
	int n=g.getnodesnum();
	double res=0;
	args.checkpre=1;
	args.l+=log(2)/log(n);
	args.eps_p=sqrt(2)*args.epsilon;
	double alpha=sqrt(args.l*log(n) + log(2));
	double beta=sqrt(args.t * logcnk(n, args.k)/2 + sqr(alpha));
	args.lam_p=(2 + 2*args.eps_p/3)*(args.l * log(n) + args.t * logcnk(n, args.k) + log(log2(n))) * n / sqr(args.eps_p);
	double lam_star=2 * n * args.t * sqr(alpha/2 + beta) / sqr(args.epsilon);
	vector<int> rem(args.t+1,0);

	for (int reps=0; reps<args.evalcnt; reps++){
		g.resetstate(args);
		g.genthv();  //generate a LT instance (threshold values)

		int i,ores=g.eval(0,args);
		g.genrv(1,args);
		for (i=1;i<=args.t;i++) rem[i]=args.k;

		double lb = opt_est_osip(g,args,rem);
		ll theta = ceil(lam_star / lb);
		assert(theta <= INT_MAX);
		g.buildripset(theta,args);
		g.osipselect(args,rem);
		g.osipchangerem(rem);
		g.insertnewpairs();

		int nres=g.eval(1,args);
		double rr=(double)(ores-nres)/ores;
		res+=rr;
		printf("Test #%d: No imm-set: %d, with imm-set: %d, reduction ratio: %.6f\n",reps,ores,nres,rr);fflush(stdout);
		if (args.verbose){
			g.printans(args,af);
		}
	}
	
	return res/args.evalcnt;
}

void show_original(infgraph &g, configargs &args, FILE* af){
	int n=g.getnodesnum();
	for (int reps=0; reps<args.evalcnt; reps++){
		g.resetstate(args);
		g.genthv();  //generate a LT instance (threshold values)

		int ores=g.eval(1,args);
		printf("Test #%d: No imm-set: %d\n",reps,ores);
		if (args.verbose){
			g.printans(args,af);
		}
	}
}

void readgraph(infgraph &g,const configargs &args){
	map<int,int> st;
	int i,n=0,m;
	scanf("%d",&m);
	for (i=1;i<=m;i++){
		int t1,t2;
		scanf("%d%d",&t1,&t2);
		if (!st.count(t1)) st[t1]=n++;
		if (!st.count(t2)) st[t2]=n++;
		g.addedge(st[t1],st[t2]);
	}
	if (args.verbose){
		FILE *mp = fopen("nodemapping.out","w");
		fprintf(mp,"Node mapping: (node number in dataset file, mapped node number):\n");
		for (auto z:st) fprintf(mp,"%d %d\n",z.first,z.second);
		fclose(mp);
	}
	g.setweights(args);
}

int main(int argc, char **argv){
	assert(argc==2);
	string casename(argv[1]);
	string cofname=casename+".in";
	string oufname=casename+".log";
	string ansname=casename+".out";

	FILE *cof=fopen(cofname.c_str(),"r");
	freopen(oufname.c_str(),"w",stdout);
	assert(cof!=NULL);
	char ch[111];
	fscanf(cof,"%s",ch);
	printf("Config file name: %s\n",cofname.c_str());
	printf("Data file name: %s\n",ch);
	assert(freopen(ch,"r",stdin)!=NULL);

	//read config
	uint32_t randseed,graphranseed;
	fscanf(cof,"%s%u",ch,&graphranseed);
	fscanf(cof,"%s%u",ch,&randseed);
	printf("Random seed for Generating graph weights = %u\n",graphranseed);
	if (randseed==0){
		randseed=time(NULL);
		printf("Random seed=0, generating it from time(), seed= %u\n",randseed);
	}else{
		printf("Random seed= %u\n",randseed);
	}
	infgraph g(randseed, graphranseed);
	configargs args;
	fscanf(cof,"%s%lf",ch,&args.l);
	fscanf(cof,"%s%lf",ch,&args.epsilon);
	fscanf(cof,"%s%d",ch,&args.type);
	fscanf(cof,"%s%d",ch,&args.k);
	fscanf(cof,"%s%d",ch,&args.t);
	fscanf(cof,"%s%lf",ch,&args.wsum);
	fscanf(cof,"%s%lf",ch,&args.alv);
	fscanf(cof,"%s%d",ch,&args.evalcnt);
	fscanf(cof,"%s%d",ch,&args.verbose);
	args.showargs();
	readgraph(g,args);
	g.readseeds(cof);
	fclose(cof);
	fflush(stdout);fflush(stderr);

	printf("Reading data done, starting clock.\n");fflush(stdout);
	clock_t t0 = clock();
	double res = -1;
	FILE *af;
	if (args.verbose){
		af=fopen(ansname.c_str(),"w");
		assert(af!=NULL);
	}
	assert(args.type>=0&&args.type<=4);
	if (args.type==0){
		show_original(g,args,af);
	}
	if (args.type==1){
		res = solve_krimm(g,args,af);
	}
	if (args.type==2){
		res = solve_atrip(g,args,af);
	}
	if (args.type==3){
		res = solve_osrp(g,args,af);
	}
	if (args.type==4){
		res = solve_osip(g,args,af);
	}
	clock_t t2 = clock();
	printf("Done, CPU clock() time elapsed (sec): %.3f\n",1.0*(t2-t0)/CLOCKS_PER_SEC);fflush(stdout);
	printf("Average reduction ratio: %.6f\n",res);
	return 0;
}
