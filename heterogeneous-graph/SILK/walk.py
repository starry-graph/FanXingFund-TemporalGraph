import numpy as np
import networkx as nx
import random
import math

class RWGraph():
    def __init__(self, dic_G):
        self.G = dic_G

    def walk(self, walk_length, start, alpha, target_type, type_att):
        # Simulate a random walk starting from start node.
        G = self.G
        length = 0
        change_probability = 0
        
        type = list()
        type.append(start)
        walk = [start]
        while len(walk) < walk_length:
            cur = walk[-1]
            candidates = []
            if cur[0] == target_type:
                change_probability = 1 - math.pow(alpha, length)
                r = random.uniform(0, 1)
                '''
                order = list(type_att[cur])
                if r > change_probability:
                    candidates.extend([e for e in G[cur] if (e[0] == order[0])])
                    if not candidates:
                        candidates.extend([e for e in G[cur] if (e[0] == order[1])])

                else:
                    if order[0] not in type:
                        candidates.extend([e for e in G[cur] if (e[0] == order[0])])
                    if not candidates:
                        if order[1] not in type:
                            candidates.extend([e for e in G[cur] if (e[0] == order[1])])
                '''
                order = list(type_att[cur])
                rr = random.uniform(0, 1)
                if r > change_probability:
                    if order[0] > rr:
                        candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                        if not candidates:
                            candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                    else:
                        candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                        if not candidates:
                            candidates.extend([e for e in G[cur] if (e[0] == 'd')])

                else:
                    if order[0] > rr:
                        if 'd' not in type:
                            candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                        if not candidates:
                            if 'a' not in type:
                                candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                    else:
                        if 'a' not in type:
                            candidates.extend([e for e in G[cur] if (e[0] == 'a')])
                        if not candidates:
                            if 'd' not in type:
                                candidates.extend([e for e in G[cur] if (e[0] == 'd')])
                
                if candidates:
                    weights = []
                    next_list = []
                    for candidate in candidates:
                        weights.append((candidate,G.degree(candidate)))
                    weights = sorted(weights,key=lambda x:x[1],reverse = True)
                    for i in range(int(len(weights)/3)+1):
                        next_list.extend([weights[i][0]])
                    next = random.choice(next_list)
                    if next[0] == type[-1]:
                        length = length + 1
                    else:
                        type[:1] = []
                        type.append(next[0])
                        length = 0
                    walk.append(next)
                else:
                    break
            else:
                candidates.extend([e for e in G[cur]])
                if candidates:
                    next = random.choice(candidates)
                    walk.append(next)
                else:
                    break
        return walk

        
    def simulate_walks(self, num_walks, walk_length, alpha, type, type_att):
        G = self.G
        walks = []
        paths = []
        all_walks = []
        #nodes = list(n for n in G.nodes() if(n[0] == type))
        nodes = list(n for n in G.nodes())
        # print('Walk iteration:')
        for walk_iter in range(num_walks):
            print('---walk num:', walk_iter)
            random.shuffle(nodes)
            for node in nodes:
                walk = self.walk(walk_length=walk_length, start=node, alpha = alpha, target_type = type, type_att = type_att)
                all_walks.append([str(n) for n in walk])
                walkn = []
                pathn = []
                for n in walk:
                    if n[0] == type:
                        walkn.append(n)
                    else:
                        pathn.append(n[0])
                walks.append([str(n) for n in walkn])
                paths.append([str(t) for t in pathn])

        return all_walks, walks, paths


