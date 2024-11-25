import numpy as np

class SimpleGraph:
    def __init__(self, V, E):
        self.vertices = V
        self.edge_set = E
        self.outgoing = [set() for _ in range(V)]
        self.incoming = [set() for _ in range(V)]
        
        for v1, v2 in self.edge_set:
            if v1 < 0 or v1 >= V:
                raise Exception(f'invalid vertex id {v1} for graph with {V} vertices')
            if v2 < 0 or v2 >= V:
                raise Exception(f'invalid vertex id {v2} for graph with {V} vertices')
            
            self.outgoing[v1].add(v2)
            self.incoming[v2].add(v1)

        self.bidirectional = [self.incoming[v].union(self.outgoing[v]) for v in range(V)]

    def is_connected(self):
        if self.vertices <= 1:
            return True
        # DFS to see if from 0 we can get to all other nodes, ignoring directionality of edges
        seen = [False]*self.vertices
        stack = [0]
        while len(stack) > 0:
            v = stack.pop()
            if not seen[v]:
                seen[v] = True
                for v2 in self.bidirectional[v]:
                    stack.append(v2)
        return all(seen)

    def is_strongly_connected(self):
        if self.vertices <= 1:
            return True
        # DFS to see if from 0 we can reach all other nodes, respecting direction
        seen = [False]*self.vertices
        stack = [0]
        while len(stack) > 0:
            v = stack.pop()
            if not seen[v]:
                seen[v] = True
                for v2 in self.outgoing[v]:
                    stack.append(v2)
        if not all(seen):
            return False
        # DFS to see if all edges can get to 0
        seen = [False]*self.vertices
        stack = [0]
        while len(stack) > 0:
            v = stack.pop()
            if not seen[v]:
                seen[v] = True
                for v2 in self.incoming[v]:
                    stack.append(v2)
        return all(seen)
        
    def adj_matrix(self):
        table = []
        for v1 in range(self.vertices):
            row = []
            for v2 in range(self.vertices):
                row.append(1 if v2 in self.outgoing[v1] else 0)
            table.append(row)
        return np.array(table)


    def subgraph(self, V):
        v_map = {}
        v_set = set([V])
        E = set()
        for i, v in enumerate(V):
            v_map[v] = i
            for v2 in self.outgoing(v):
                if v2 in v_set:
                    E.add((v_map[v], v_map[v2]))
        return SimpleGraph(len(V), E)
    
    # Tarjan's algorithm for findind strongly connected components
    # modified slightly to only return components made of vertices with index >= s
    def strongly_connected_components(self, s=0):
        idx = 0
        index = [None]*self.vertices
        lowlink = [None]*self.vertices
        onstack = [False]*self.vertices
        stack = []
        sccs = {}
        def scc_helper(v):
            nonlocal idx, index, lowlink, onstack, stack, sccs
            
            index[v] = idx
            lowlink[v] = idx
            idx += 1
            stack.append(v)
            onstack[v] = True

            for w in self.outgoing[v]:
                if w < s:
                    continue
                if index[w] is None:
                    scc_helper(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif onstack[w]:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc = []
                w = stack.pop()
                onstack[w] = False
                scc.append(w)
                while w != v:
                    w = stack.pop()
                    onstack[w] = False
                    scc.append(w)
                min_vert = min(scc)
                sccs[min_vert] = scc
                
        for v in range(s, self.vertices):
            if index[v] is None:
                scc_helper(v)
        return sccs

    # finds all elementary cycles of a graph
    # this is an implementation of an algorithm by D.B Johnson
    # taken almost verbatim from paper "Finding All the Elementary Circuits of a Directed Graph"
    def elementary_cycles(self):
        cycles = []
        blocked = [False]*self.vertices
        B = [set()]*self.vertices
        stack = []
        
        def circuit(v):
            nonlocal cycles, blocked, B, stack
            
            def unblock(u):
                nonlocal blocked, B

                blocked[u] = False
                
                ws = [w for w in B[u]]
                for w in ws:
                    if w in B[u]:
                        B[u].remove(w)
                        unblock(w)

            f = False
            stack.append(v)
            blocked[v] = True
            for w in self.outgoing[v]:
                if w == s:
                    cycles.append(stack + [s])
                    f = True
                elif not blocked[w]:
                    if circuit(w):
                        f = True
            if f:
                unblock(v)
            else:
                for w in self.outgoing[v]:
                    if v not in B[w]:
                        B[w].add(v)
            stack.pop()
            return f

        s = 0
        while s < self.vertices-1:
            sccs = self.strongly_connected_components(s)
            scc = sccs[s]
            if len(scc) > 1:
                for i in scc:
                    blocked[i] = False
                    B[i] = set()
                dummy = circuit(s)
                s += 1
            else:
                s = self.vertices-1
        return cycles

class OrientedCycleUnion:
    def __init__(self, vertices, edges):
        self.vertices = frozenset(vertices)
        self.edges = frozenset(edges)
    
    def is_compatible(self, other):
        for v1, v2 in self.edges:
            if (v2, v1) in other.edges:
                return False
        return True

    def union(self, other, check=False):
        if check:
            if not self.is_compatible(other):
                raise Exception('attempted to union incompatible Oriented Cycle Unions')
            
        vertices = self.vertices.union(other.vertices)
        edges = self.edges.union(other.edges)
        
        return OrientedCycleUnion(vertices, edges)

    # eq and hash functions so these can be added to a set
    def __eq__(self, other):
        return self.vertices == other.vertices and self.edges == other.edges

    def __hash__(self):
        return hash((self.vertices, self.edges))
    
def random_simple_graph(num_v, num_e, directed=True):
    V = num_v
    E = set()
    for _ in range(num_e):
        v1, v2 = tuple(np.random.choice(range(num_v), 2, replace=False))
        if (v1, v2) not in E:
            E.add((v1, v2))
            if not directed:
                E.add((v2, v1))
    return SimpleGraph(V, E)

# backtracking helper function for subgraph search
# receives the current ocu that has been built so far
def backtrack_find_subgraphs(ocus, v, k, current_ocu, subgraphs, compatibility, used_stack):
    # if we have >= k edges, we don't need to backtrack further
    if len(current_ocu.edges) >= k:
        return

    # if the current_ocu is a valid subgraph, add it to our set
    if len(current_ocu.edges) > 0 and v in current_ocu.vertices:
        subgraphs.add(current_ocu)

    current_idx = len(used_stack) - 1
    
    # loop over all next ocus to union with the current one
    # i.e. don't backtrack on all the ocus we don't union, just the next one we do union
    for next_offset in range(1, len(ocus)-current_idx):
        next_idx = current_idx + next_offset
        next_ocu = ocus[next_idx]
        # if they're not compatible don't backtrack
        if not compatibility[current_idx][next_idx]:
            continue

        new_ocu = current_ocu.union(next_ocu)

        # if we don't get any new edges, don't backtrack and continue
        if len(new_ocu.edges) == len(current_ocu.edges):
            continue
            
        used_stack = [False] * (next_offset - 1) + [True]
        backtrack_find_subgraphs(ocus, v, k, new_ocu, subgraphs, compatibility, used_stack)
        used_stack = used_stack[:current_idx+1]

# backtracking solution to find all subgraphs
# ocus is the set of all elementary cycles implemented as OrientedCycleUnion objects
# compatibility is a precomputed compatibility table to prevent unions that would result in bidirectional edges
# v is the vertex required to be in the subgraph
# k is the upper bound (exclusive) on number of edges
def find_subgraphs(ocus, compatibility, v, k):
    # subgraphs is the set of all subgraphs that will be populated by backtracking
    subgraphs = set()

    # contains true or false for each ocu seen so far indicating whether or not it is contained in the union
    used_stack = []

    # loop over all ocus so that each gets to act as the smallest 
    for initial_ocu in ocus:
        used_stack.append(True)

        backtrack_find_subgraphs(ocus, v, k, initial_ocu, subgraphs, compatibility, used_stack)
        
        used_stack.pop()
        used_stack.append(False)
    return subgraphs

if __name__ == "__main__":
    # rejection sampling to find strongly connected graph
    g = random_simple_graph(10, 20, directed=True)
    while not g.is_strongly_connected():
        g = random_simple_graph(10, 20, directed=True)

    print('Adjacency matrix of random undirected strongly connected simple graph')
    print(g.adj_matrix())

    print('finding all elementary cycles')
    cycles = g.elementary_cycles()
    print(f'found {len(cycles)} elementary cycles')
    
    # convert each cycle into OrientedCycleUnion
    ocus = []
    for cycle in cycles:
        vertices = set()
        edges = set()
        for v in cycle[:-1]:
            vertices.add(v)
        for v1, v2 in zip(cycle[:-1], cycle[1:]):
            edges.add((v1, v2))
            
        ocus.append(OrientedCycleUnion(vertices, edges))

    print('determining oriented union compatability between cycles')
    compatibility = []
    for c1 in ocus:
        row = []
        for c2 in ocus:
            row.append(c1.is_compatible(c2))
        compatibility.append(row)

    k = 10
    v = 0
    print(f'finding all oriented subgraphs containing vertex {v} with no sources or sinks with 0 < |E| < {k}')
    subgraphs = find_subgraphs(ocus, compatibility, v, k)

    # I'm just printing the number here, but all the subgraphs exist in the form of OrientedCycleUnions
    print(f'found {len(subgraphs)} subgraphs')

    
    
