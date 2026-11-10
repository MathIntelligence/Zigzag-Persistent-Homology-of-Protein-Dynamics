import numpy as np
from independent_rank import generalized_rank,rank_from_edge_history,image_rank_from_edge_history


def test_one_surviving_interval():
    assert generalized_rank([1,1,1],[(0,1,[1]),(2,1,[1])])==1


def test_replaced_interval_same_frame_dimensions():
    assert generalized_rank([1,2,1],[(0,1,[1]),(2,1,[2])])==0


def test_change_of_basis_and_partial_survival():
    assert generalized_rank([2,2,2],[(0,1,[3,1]),(2,1,[1,2])])==2
    assert generalized_rank([2,1,2],[(0,1,[1,0]),(2,1,[0,1])])==1
    assert generalized_rank([0,1,1],[(0,1,[]),(2,1,[1])])==0


def test_actual_disjoint_square_replacement():
    pairs=np.array([(0,1),(0,3),(1,2),(2,3),(4,5),(4,7),(5,6),(6,7)])
    presence=np.zeros((8,2),bool);presence[:4,0]=True;presence[4:,1]=True
    rank,dims=rank_from_edge_history(8,pairs,presence)
    assert rank==0 and dims==[1,2,1]
    rank,dims=rank_from_edge_history(8,pairs,presence,bridge='intersection')
    assert rank==0 and dims==[1,0,1]
    presence[:4,:]=True;presence[4:,:]=False
    assert rank_from_edge_history(8,pairs,presence)[0]==1
    assert rank_from_edge_history(8,pairs,presence,bridge='intersection')[0]==1


def test_region_hole_filled_by_external_residue():
    # The four-cycle region is coned off by an omitted fifth vertex in the full complex.
    pairs=np.array([(0,1),(0,3),(0,4),(1,2),(1,4),(2,3),(2,4),(3,4)])
    presence=np.ones((8,2),bool)
    rank,dims=image_rank_from_edge_history(5,pairs,presence,[0,1,2,3])
    assert rank==0 and dims==[0,0,0]


def test_region_image_preserved_when_global_hole_real():
    pairs=np.array([(0,1),(0,3),(1,2),(2,3),(3,4)])
    presence=np.ones((5,2),bool)
    rank,dims=image_rank_from_edge_history(5,pairs,presence,[0,1,2,3])
    assert rank==1 and dims==[1,1,1]
