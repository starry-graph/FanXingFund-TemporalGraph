#include "genrandom.h"
#include "infgraph.h"
#include<vector>
#include<cassert>
#include<queue>
#include<map>
#include<set>
#include<algorithm>
#include<utility>
using namespace std;

infgraph::infgraph(uint32_t randseed, uint32_t graphranseed) : rg(randseed), n(0), rsetnum(0), rsetcov(0), graph_rg(graphranseed){

}

void infgraph::addedge(int x,int y){
	n=max(n,1+max(x,y));
	while (eb.size()<n){
		vector<int> ev;
		eb.push_back(ev);ef.push_back(ev);
	}
	eb[y].push_back(x);
}

void infgraph::readseeds(FILE *cof){
	char ch[111];
	int i,j,k;
	fscanf(cof,"%s%d",ch,&k);
	fscanf(cof,"%d",&j);
	assert(k>=0&&k<=n);
	printf("number of initial N-inf seeds = %d\n",k);
	if (!j){
		puts("randomly generating initial seeds:");
		for (i=0;i<n;i++) seeds.push_back(i);
		for (i=0;i<k;i++){
			j=rg.getint(n-i);
			swap(seeds[i],seeds[j+i]);
		}
		j=n-k;
		for (i=0;i<j;i++) seeds.pop_back();
		for (auto z:seeds) printf("%d ",z);
		puts("");
	}else{
		puts("reading initial seeds from file:");
		for (i=0;i<k;i++){
			fscanf(cof,"%d",&j);
			printf("%d ",j);
			seeds.push_back(j);
		}
		puts("");
	}
}

int infgraph::genrip(int v, const configargs &args){
	int i,j,tv;
	vector<int> res;
	while (1){
		j=eb[v].size();vis[v]=1;
		res.push_back(v);
		if (st[v]>=0) break;
		double t=rg.getreal();
		tv=-1;
		for (i=0;i<j;i++){
			t-=wb[v][i];
			if (t<=0){
				tv=eb[v][i];
				break;
			}
		}
		v=tv;
		if (v==-1||vis[v]) break;
	}
	for (auto z:res) vis[z]=0; //revert vis to all 0's
	if (v==-1||st[v]<0){
		return 1;
		//rsetnum++;rcov.push_back(0);
		//return 0;
	}
	j=res.size();
	int ots=st[v];
	if (args.checkpre){
		for (i=j-1;i>=0;i--){
			int k=ots+(j-i-1);
			if (k<=args.t&&bset[k].count(res[i])>0) return 2;
		}
	}
	rsetnum++;rcov.push_back(0);
	for (i=j-1;i>=0;i--){
		int k=ots+(j-i-1);
		if (k>args.t) continue;
		if (rset[k].count(res[i])==0){
			vector<int> tmp0;
			rset[k][res[i]]=tmp0;
		}
		rset[k][res[i]].push_back(rsetnum-1);
	}
	return 0;
}

void infgraph::buildripset(int num, const configargs &args){
	vector<int> w=rv;
	int i,k=w.size();
	while (rsetnum < num){
		if (k==0){
			if (args.verbose) printf("**Warning: no possible RIP set.**\n");
			return;
		}
		i=rg.getint(k);
		genrip(w[i],args);
	}
	//printf("** Finished building rset, num = %d\n",rsetnum);
}

static bool immselect_cmp(const pair<int,int> &a, const pair<int,int> &b){
	return a.first>b.first;
}

double infgraph::immselect(int num, const configargs &args,int ts){
	vector<pair<int,int>> w;
	int ans=rsetcov;
	int i,j,k=rv.size();
	bres.clear();
	for (i=0;i<k;i++){
		j=0;
		if (rset[ts].count(rv[i])>0){
			for (auto z:rset[ts][rv[i]])
				if (!rcov[z]) j++;
		}
		w.push_back(make_pair(j,rv[i]));
		//printf("** %d %d\n",rv[i],j);
	}
	sort(w.begin(),w.end(),immselect_cmp);
	if (w.size()<num){
		if (args.verbose) printf("**Warning: not enough nodes in V' to select.**\n");
		num=w.size();
	}
	for (i=0;i<num;i++){
		ans+=w[i].first;
		bres.push_back(make_pair(w[i].second,ts));
	}
	return (double)ans/rsetnum;
}

double infgraph::osipselect(const configargs &args, const vector<int> &in_rem){
	pair<int,int> c;
	vector<int> b;
	int ans=rsetcov;
	int i,j,k=rv.size(),w;
	bres.clear();
	vector<set<int>> sp;
	vector<int> rem = in_rem;

	for (i=0;i<=args.t;i++){
		set<int> tmp0;
		sp.push_back(tmp0);
	}

	while (1){
		w=-1;
		for (int ts=1;ts<=args.t;ts++){
			if (!rem[ts]) continue;
			for (i=0;i<k;i++){
				if (sp[ts].count(rv[i])>0) continue; // node already selected
				j=0;
				if (rset[ts].count(rv[i])>0){
					for (auto z:rset[ts][rv[i]])
						if (!rcov[z]) j++;
				}
				if (w<j) w=j,c=make_pair(rv[i],ts);
			}
		}
		if (w==-1) break;
		bres.push_back(c);sp[c.second].insert(c.first);
		ans+=w;rem[c.second]--;
		for (auto y:rset[c.second][c.first])
			if (rcov[y]==0){
				rcov[y]=1;
				b.push_back(y);
			}
	}

	// revert rcov  status
	for (auto z:b) rcov[z]=0;

	return (double)ans/rsetnum;
}

int infgraph::getnodesnum(){
	return n;
}

void infgraph::setweights(const configargs &args){
	vector<double> evd;
	set<int> iset;
	map<int,vector<int>> imap;
	int i,j;
	double ws=args.wsum;  // wsum < 0 means it is randomly generated for each node
	FILE *gf;
	if (args.verbose){
		gf=fopen("graphweights.out","w");
		fprintf(gf,"%d\n",n);
	}
	for (i=0;i<n;i++) wb.push_back(evd),wf.push_back(evd);
	for (i=0;i<n;i++) vis.push_back(0),st.push_back(-2),thv.push_back(0);
	for (i=0;i<=args.t;i++) rset.push_back(imap),bset.push_back(iset);
	for (i=0;i<n;i++){
		double sum=0;
		int num=eb[i].size();
		for (j=0;j<num;j++){
			double tt=graph_rg.getreal();
			sum+=tt;
			wb[i].push_back(tt);
		}
		double wsum=(ws<0)?graph_rg.getreal():ws;
		double tsum=0;
		for (j=0;j<num;j++){
			wb[i][j]/=sum;wb[i][j]*=wsum;
			ef[eb[i][j]].push_back(i);
			wf[eb[i][j]].push_back(wb[i][j]);
			if (args.verbose){
				fprintf(gf, "%d %d %.6lf ", eb[i][j], i, tsum);
				tsum += wb[i][j];
				fprintf(gf, "%.6lf\n", tsum);
			}
		}
	}
	if (args.verbose) fclose(gf);
}

void infgraph::printans(const configargs &args, FILE *ansf){
	puts("Infected nodes: (node, timestep)");
	fprintf(ansf,"Infected-nodes-(node,timestamp)\n");
	for (int i=0;i<n;i++){
		if (st[i]<0) continue;
		printf("(%d, %d) ",i,st[i]);fprintf(ansf,"%d %d\n",i,st[i]);
		puts("");
	}
	puts("Imm-pairs selected: (node, timestep)");
	fprintf(ansf,"Blocked-nodes-(node,timestamp)\n");
	for (int i=1;i<=args.t;i++){
		for (auto z:bset[i]) printf("(%d, %d) ",z,i),fprintf(ansf,"%d %d\n",z,i);
		puts("");
	}
}

void infgraph::genrv(int ts, const configargs &args){
	int i,j,k;
	rv.clear();
	vector<int> c(n,0);
	queue<int> q;
	if (ts==1){
		for (i=0;i<n;i++) st[i]=-2;
		srv=seeds;
		for (auto z:seeds) c[z]=1,st[z]=0;
		for (i=0;i<srv.size();i++){
			j=srv[i];
			for (auto z:ef[j])
				if (!c[z]){
					c[z]=1;st[z]=-1;
					srv.push_back(z);
				}
		}
		for (i=0;i<n;i++) c[i]=0;
		if (args.verbose){
			printf("Node reachable from seeds at the begining (only consider graph structure):");
			for (auto z:srv) printf(" %d",z);
			puts("");
		}
	}
	for (i=0;i<n;i++)
		if (st[i]!=-1&&st[i]<ts-1) q.push(i);
	while (!q.empty()){
		j=q.front();q.pop();
		for (auto z:ef[j]){
			if (st[z]!=-1) continue;
			if (++c[z]==eb[z].size()){          // all incoming neighbors activated, or confirmed can't be activated
				q.push(z);
				st[z]=-2;   // confirmed can't be activated
			}
			assert(c[z]<=eb[z].size());
		}
	}
	for (auto z:srv)
		if (st[z]==-1) rv.push_back(z);
	if (args.verbose){
		printf("Node reachable from seeds at the timestep %d:",ts);
		for (auto z:srv)
			if (st[z]==-1) printf(" %d",z);
		puts("");
	}
}

bool infgraph::dotimestep(int ts){
	int flag=0,tlim=bset.size();
	for (auto z:srv)
		if (st[z]==ts-1){
			int k=ef[z].size();
			for (int i=0;i<k;i++) thv[ef[z][i]]-=wf[z][i];
		}
	for (auto z:srv)
		if (thv[z]<=0){
			assert(st[z]!=-2);
			if (ts<tlim&&bset[ts].count(z)>0){
				// node is blocked at the critital time step, can't be activated any more
				thv[z] = 1000;
				continue;
			}
			if (st[z]==-1) st[z]=ts,flag=1;  //node activated at timestep ts
		}
	return flag;   // is at least one node activated in this timestep, if not, no need to continue
}

void infgraph::genthv(){
	for (int i=0;i<n;i++) thv[i]=graph_rg.getreal();
}

void infgraph::insertnewpairs(){
	for (auto z:bres){
		int v=z.first,t=z.second;
		bset[t].insert(v);
		if (rset[t].count(v)==0) continue;
		for (auto y:rset[t][v])
			if (rcov[y]==0){
				rcov[y]=1;rsetcov++;
			}
	}
}

double infgraph::insertnewpair_at(int x){
	if (x>=bres.size()) return -1;
	auto z=bres[x];
	int v=z.first,t=z.second;
	bset[t].insert(v);
	if (rset[t].count(v)==0) return (double)rsetcov/rsetnum;
	for (auto y:rset[t][v])
		if (rcov[y]==0){
			rcov[y]=1;rsetcov++;
		}
	return (double)rsetcov/rsetnum;
}

void infgraph::resetstate(const configargs &args){
	int i;
	for (i=0;i<=args.t;i++) rset[i].clear(),bset[i].clear();
	for (i=0;i<n;i++) vis[i]=0,st[i]=-2;
	rcov.clear();bres.clear();
	rsetnum=rsetcov=0;
}

int infgraph::eval(int setst, const configargs &args){
	int i,j,k;
	vector<int> q,c(n,-1);
	vector<double> t=thv;
	for (auto z:seeds) c[z]=0,q.push_back(z);
	for (i=0;i<q.size();i++){
		j=q[i];k=ef[j].size();
		//if (c[j]>=args.t) break;
		for (int y=0;y<k;y++){
			int x=ef[j][y];
			t[x]-=wf[j][y];
			if (t[x]>0||c[x]>=0) continue;
			if (c[j]+1<=args.t&&bset[c[j]+1].count(x)>0){
				// node is blocked at the critital time step, can't be activated any more
				thv[x] = 1000;
				continue;
			}
			c[x]=c[j]+1;q.push_back(x);
		}
	}
	if (setst) st=c;
	return q.size();
}

int infgraph::countinf(){
	int res=0;
	for (int i=0;i<n;i++)
		if (st[i]>=0) res++;
	return res;
}

int infgraph::geth0(){
	return srv.size();
}

void infgraph::resetrset(const configargs &args){
	for (int i=0;i<=args.t;i++) rset[i].clear();
	rcov.clear();
	rsetnum=rsetcov=0;
}

double infgraph::getnewcov(const configargs &args){
	bres.clear();
	for (int i=0;i<=args.t;i++){
		for (auto z:bset[i]) bres.push_back(make_pair(z,i));
	}
	insertnewpairs();
	return (double)rsetcov/rsetnum;
}

void infgraph::osipchangerem(vector<int> &rem){
	for (auto z:bres){
		int t=z.second;
		rem[t]--;
		assert(rem[t]>=0);
	}
}
