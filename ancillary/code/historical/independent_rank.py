"""Independent F2 limit-to-colimit rank of frame/union H1 zigzags.

No persistence library is used for the rank calculation. This verifies the
number of full-window interval summands reported by the barcode implementation.
"""
import itertools
import numpy as np


def reduce_rows(rows):
    piv={}
    for row in rows:
        x=int(row)
        while x:
            k=x.bit_length()-1
            if k in piv:x^=piv[k]
            else:piv[k]=x;break
    return piv


def generalized_rank(dims,arrows):
    """arrows=(source,target,map columns encoded in target basis)."""
    offsets=np.r_[0,np.cumsum(dims)].astype(int).tolist();D=offsets[-1]
    relations=[];constraints=[]
    for src,dst,cols in arrows:
        assert len(cols)==dims[src]
        assert all(0<=c<(1<<dims[dst]) for c in cols)
        for k,c in enumerate(cols):relations.append((1<<(offsets[src]+k))^(int(c)<<offsets[dst]))
        for k in range(dims[dst]):
            row=1<<(offsets[dst]+k)
            for j,c in enumerate(cols):
                if c>>k&1:row^=1<<(offsets[src]+j)
            constraints.append(row)
    piv=reduce_rows(constraints);null=[]
    for free in range(D):
        if free in piv:continue
        x=1<<free
        for k in sorted(piv):
            if (piv[k]&x).bit_count()%2:x^=1<<k
        assert all((row&x).bit_count()%2==0 for row in constraints)
        null.append(x)
    projections=[x&((1<<dims[0])-1) for x in null]
    return len(reduce_rows(relations+projections))-len(reduce_rows(relations))


class H1Space:
    def __init__(self,n,pairs,active,triangles):
        self.pivots={};self.labels={};self.cycles=[]
        for c in triangles:self._insert(int(c),0)
        parent=list(range(n));forest=[[] for _ in range(n)];nonforest=[]
        def find(v):
            while parent[v]!=v:
                parent[v]=parent[parent[v]];v=parent[v]
            return v
        for k in sorted(active):
            u,v=map(int,pairs[k]);a,b=find(u),find(v)
            if a==b:nonforest.append((k,u,v))
            else:parent[a]=b;forest[u].append((v,k));forest[v].append((u,k))
        for k,u,v in nonforest:
            todo=[(u,-1,0)]
            while todo:
                w,prev,chain=todo.pop()
                if w==v:break
                todo.extend((z,w,chain^(1<<idx)) for z,idx in forest[w] if z!=prev)
            chain^=1<<k
            residual,_=self.reduce(chain)
            if residual:
                self.cycles.append(residual)
                self._insert(residual,1<<(len(self.cycles)-1))

    def reduce(self,x):
        coord=0
        while x:
            p=x.bit_length()-1
            if p not in self.pivots:break
            x^=self.pivots[p];coord^=self.labels[p]
        return x,coord

    def _insert(self,x,label):
        while x:
            p=x.bit_length()-1
            if p in self.pivots:x^=self.pivots[p];label^=self.labels[p]
            else:self.pivots[p]=x;self.labels[p]=label;return

    def coordinates(self,cycle):
        residual,c=self.reduce(cycle)
        assert residual==0,'Included cycle not in target cycle space'
        return c


def frame_triangles(n,pairs,active):
    adj=[set() for _ in range(n)];index={}
    for k in active:
        i,j=map(int,pairs[k]);adj[i].add(j);adj[j].add(i);index[i,j]=k
    tri=set()
    for i,j in index:
        for k in adj[i]&adj[j]:
            if k>j:tri.add((1<<index[i,j])^(1<<index[i,k])^(1<<index[j,k]))
    return tri


def rank_from_edge_history(n,pairs,presence,bridge='union'):
    """presence has shape (edges,frames). Union is simplex union, not clique completion."""
    act=[set(map(int,np.flatnonzero(presence[:,t]))) for t in range(presence.shape[1])]
    tri=[frame_triangles(n,pairs,x) for x in act]
    frames=[H1Space(n,pairs,x,y) for x,y in zip(act,tri)]
    assert bridge in ['union','intersection']
    nodes=[];arrows=[]
    for t,frame in enumerate(frames):
        nodes.append(frame)
        if t+1<len(frames):
            ae=act[t]|act[t+1] if bridge=='union' else act[t]&act[t+1]
            at=tri[t]|tri[t+1] if bridge=='union' else tri[t]&tri[t+1]
            nodes.append(H1Space(n,pairs,ae,at))
    for t in range(len(frames)-1):
        for endpoint in [2*t,2*t+2]:
            src,dst=(endpoint,2*t+1) if bridge=='union' else (2*t+1,endpoint)
            arrows.append((src,dst,[nodes[dst].coordinates(c) for c in nodes[src].cycles]))
    dims=[len(x.cycles) for x in nodes]
    return generalized_rank(dims,arrows),dims


class Subspace:
    """A basis and coordinates for an image subspace, using Python-int F2 vectors."""
    def __init__(self,vectors):
        self.pivots={};self.labels={};self.basis=[]
        for v in vectors:
            x,c=self.reduce(v)
            if x:
                p=x.bit_length()-1;self.pivots[p]=x
                self.labels[p]=1<<len(self.basis);self.basis.append(x)

    def reduce(self,x):
        c=0
        while x:
            p=x.bit_length()-1
            if p not in self.pivots:break
            x^=self.pivots[p];c^=self.labels[p]
        return x,c

    def coordinates(self,x):
        residual,c=self.reduce(x);assert residual==0;return c


def image_rank_from_edge_history(n,pairs,presence,support):
    """Rank of im[H1(region)->H1(full)] as a zigzag module, not isolated-region H1."""
    support=set(support)
    allowed={i for i,(u,v) in enumerate(pairs) if u in support and v in support}
    act=[set(map(int,np.flatnonzero(presence[:,t]))) for t in range(presence.shape[1])]
    local=[x&allowed for x in act]
    tri=[frame_triangles(n,pairs,x) for x in act]
    lt=[frame_triangles(n,pairs,x) for x in local]
    whole=[];region=[]
    for t in range(len(act)):
        whole.append(H1Space(n,pairs,act[t],tri[t]));region.append(H1Space(n,pairs,local[t],lt[t]))
        if t+1<len(act):
            whole.append(H1Space(n,pairs,act[t]|act[t+1],tri[t]|tri[t+1]))
            region.append(H1Space(n,pairs,local[t]|local[t+1],lt[t]|lt[t+1]))
    images=[Subspace([w.coordinates(c) for c in r.cycles]) for w,r in zip(whole,region)]
    arrows=[]
    for t in range(len(act)-1):
        for src in [2*t,2*t+2]:
            dst=2*t+1;cols=[]
            for image_vector in images[src].basis:
                chain=0
                for k,c in enumerate(whole[src].cycles):
                    if image_vector>>k&1:chain^=c
                cols.append(images[dst].coordinates(whole[dst].coordinates(chain)))
            arrows.append((src,dst,cols))
    dims=[len(x.basis) for x in images]
    return generalized_rank(dims,arrows),dims
