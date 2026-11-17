import numpy as np
from regional_continuity import coverage,describe
from inspect_temporal import triangle_pivots,nontrivial_cycle


def test_occupancy_does_not_imply_continuity():
    # Same beta=1 at every sample; first sequence replaces its class halfway.
    beta=np.ones(40,int);q=np.zeros(40)
    w,_=describe(np.array([0,10,20,30]),np.array([10,20,30,40]),beta,q)
    assert all(x['hidden_replacement'] and x['constant_one'] for x in w)
    w,_=describe(np.array([0]),np.array([40]),beta,q)
    assert not any(x['hidden_replacement'] for x in w)
    assert coverage(np.array([0]),np.array([40]),0,40)==1


def test_cycle_nontrivial_not_just_closed():
    square=[(0,1),(0,3),(1,2),(2,3)]
    lookup,piv=triangle_pivots(4,square)
    assert nontrivial_cycle(square,lookup,piv,4)
    filled=sorted(square+[(0,2)])
    lookup,piv=triangle_pivots(4,filled)
    assert not nontrivial_cycle(square,lookup,piv,4)
