import itertools
import numpy as np
from scipy.stats import kendalltau
import audit_order as a
import dionysus as d


def dionysus_h1(n,edges):
    edges=[tuple(e) for e in edges];edge_set=set(edges)
    simplices=[[i] for i in range(n)]+[list(e) for e in edges]
    simplices += [list(t) for t in itertools.combinations(range(n),3)
                  if all(e in edge_set for e in itertools.combinations(t,2))]
    filt=d.Filtration([d.Simplex(s,0.) for s in simplices]);filt.sort()
    hom=d.homology_persistence(filt,prime=2);diags=d.init_diagrams(hom,filt)
    return sum(np.isinf(x.death) for x in diags[1]) if len(diags)>1 else 0


def test_known_complexes():
    for n,e,b in [(4,[(0,1),(0,3),(1,2),(2,3)],1),
                  (3,[(0,1),(0,2),(1,2)],0),(5,[(0,1),(2,3)],0),
                  (4,list(itertools.combinations(range(4),2)),0),
                  (7,[(0,1),(0,3),(0,4),(0,6),(1,2),(2,3),(4,5),(5,6)],2)]:
        assert a.h1_basis(n,e)==b==dionysus_h1(n,e)
        beta,reps=a.h1_basis(n,e,True);assert beta==len(reps)
        for rep in reps:
            degrees=np.bincount(np.asarray(rep).ravel(),minlength=n)
            assert np.all(degrees%2==0)


def test_random_complexes():
    rng=np.random.default_rng(901)
    for n in range(4,20):
        all_edges=np.asarray(list(itertools.combinations(range(n),2)))
        for p in [.15,.4,.7]:
            edges=all_edges[rng.random(len(all_edges))<p]
            assert a.h1_basis(n,edges)==dionysus_h1(n,edges)


def test_short_pulses():
    x=np.array([[1,1,0,1,1,1,0,1]],bool)
    assert a.drop_short(x).tolist()==[[False,False,False,True,True,True,False,False]]


def test_observed_only_and_ties():
    x=np.array([[-1,1,2,3,4,499],[0,2,1,3,4,499]])
    m,c=a.pair_matrix(x,observed_only=True)
    assert c[0,1]==4
    assert np.isclose(m[0,1],kendalltau([1,2,3,4],[2,1,3,4]).statistic)
    assert np.isnan(a.pair_matrix(np.full((2,5),499),observed_only=True)[0]).all()
