import functools as ft
import itertools as it
import more_itertools as mit
from typing import List, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import linprog


from utils import smith_normal_form


class SimplicialComplex:
    def __init__(self, faces: List[Set]):
        self.maximal_faces = [face for face in faces if sum(face <= other_face for other_face in faces) == 1]
        self.vertices = sorted(list({v for face in self.maximal_faces for v in face}))

        self.graph = nx.from_edgelist([edge for max_face in self.maximal_faces for edge in it.combinations(max_face, 2)])
        self.graph.add_nodes_from([node for face in self.maximal_faces if len(face) == 1 for node in face]) # add isolates

    def __repr__(self):
        return f'{self.__class__.__name__}({self.maximal_faces})'
    
    def __contains__(self, item):
        if isinstance(item, self.__class__):
            return all(any(item_face <= face for face in self.maximal_faces) for item_face in item.maximal_faces)
        elif isinstance(item, set):
            return any(item <= face for face in self.maximal_faces)
        else:
            raise NotImplementedError(f'Cannot check if {item} is contained in {self.__class__.__name__}')
    
    def __and__(self, other):
        if isinstance(other, self.__class__):
            shared_faces = [self_face & other_face for (self_face, other_face) in it.product(self.maximal_faces, other.maximal_faces)]
            # remove duplicate faces so maximal faces are calculated correctly
            # https://stackoverflow.com/a/32296966            
            return self.__class__([set(unique_face) for unique_face in set(frozenset(face) for face in shared_faces)])
        elif isinstance(other, set):
            return self.__class__([other & face for face in self.maximal_faces])
        else:
            raise NotImplementedError(f'Cannot intersect {other} and {self}')
        
    def __or__(self, other):
        if isinstance(other, self.__class__):
            # remove duplicate faces so maximal faces are calculated correctly
            # https://stackoverflow.com/a/32296966
            return self.__class__([set(unique_face) for unique_face in set(frozenset(face) for face in self.maximal_faces + other.maximal_faces)])
        elif isinstance(other, set):
            return self.__class__(self.maximal_faces + [other])
        else:
            raise NotImplementedError(f'Cannot union {other} and {self}')

    def restriction(self, vertices):
        """Returns the simplicial complex induced by a set of vertices."""    
        return SimplicialComplex([set(subset) for subset in mit.powerset(vertices) if set(subset) in self])

    @ft.cached_property
    def dim(self):
        return max(len(face) for face in self.maximal_faces) - 1

    @ft.cached_property
    def is_pure(self):
        return len(set(len(face) for face in self.maximal_faces)) == 1

    @ft.cached_property
    def is_connected(self):
        if not self.maximal_faces:
            return True
        return nx.is_connected(self.graph)

    @ft.cached_property
    def num_components(self):
        if not self.maximal_faces:
            return 1
        return nx.number_connected_components(self.graph)

    @ft.cached_property
    def is_shellable(self):
        """See shellable iff inductive and pure at each step...?"""
        raise NotImplementedError

    def k_faces(self, k):
        k_faces = {face for max_face in self.maximal_faces for face in it.combinations(max_face, k+1)}
        k_faces = [set(unique_face) for unique_face in set(frozenset(face) for face in k_faces)]
        return [sorted(list(face)) for face in k_faces]
    
    @staticmethod
    def boundary(face):
        return [face[:idx] + face[idx+1:] for idx, _ in enumerate(face)]

    def boundary_matrix(self, k):
        """Returns the differential mapping k-chains to (k-1)-chains."""
        k_faces = self.k_faces(k)
        k_minus_one_faces = self.k_faces(k-1)
        boundary_faces = [self.boundary(face) for face in k_faces]
        boundary_matrix = np.zeros((len(k_minus_one_faces), len(k_faces)))
        for k_minus_one_face in k_minus_one_faces:
            for idx, k_face in enumerate(k_faces):
                boundaries = self.boundary(k_face)
                boundary_indices = [k_minus_one_faces.index(b_face) for b_face in boundaries]
                for i, b_idx in enumerate(boundary_indices):
                    boundary_matrix[b_idx, idx] = 1
        # boundary_matrix = [[(-1)**idx if face in boundary_face else 0 for idx, face in enumerate(k_minus_one_faces)] for boundary_face in boundary_faces]
        return np.array(boundary_matrix)

    def betti(self, k: int):
        if k == 0:
            return self.num_components
        elif k == self.dim:
            return sum(1 for face in self.maximal_faces if len(face)-1 == self.dim) - np.count_nonzero(smith_normal_form(self.boundary_matrix(k)) == 1)
        else:
            reduced_k_matrix = smith_normal_form(self.boundary_matrix(k))
            reduced_kplusone_matrix = smith_normal_form(self.boundary_matrix(k+1))
            return reduced_k_matrix.shape[1] - np.count_nonzero(reduced_k_matrix == 1) - np.count_nonzero(reduced_kplusone_matrix == 1)

    @ft.cached_property
    def betti_numbers(self):
        return {k: self.betti(k) for k in range(self.dim + 1)}

    @ft.cached_property
    def reduced_betti_numbers(self):
        return {k: betti_number-1 if k == 0 else betti_number for k, betti_number in self.betti_numbers.items()}
    
    @ft.cached_property
    def f_vector(self):
        return (1,) + tuple(len(self.k_faces(k)) for k in range(self.dim+1))
    
    @ft.cached_property
    def h_vector(self):
        # Stanley's trick can be made into a triangluar array
        # https://stackoverflow.com/a/27682124
        stanleys_trick_array = np.array([0 for _ in range(sum(range(1, self.dim+4))-1)])
        offset = lambda r: (r*(r+1)) // 2
        array_index = lambda r,c: offset(r) + c
        for i, f_i in enumerate(self.f_vector):
            # column of ones
            idx = array_index(i, 0)
            stanleys_trick_array[idx] = 1
            # row of f-vector
            idx = array_index(self.dim+2, i)
            stanleys_trick_array[idx] = f_i
        # compute h-vector via Stanley's trick
        for i in reversed(range(1, self.dim+2)):
            for j in range(1, i+1):
                idx = array_index(i, j)
                left_idx = array_index(i, j-1)
                below_idx = array_index(i+1, j)
                stanleys_trick_array[idx] = stanleys_trick_array[below_idx] - stanleys_trick_array[left_idx]
        # indices on diagonal correspond to triangular numbers: https://oeis.org/A000096
        return tuple(stanleys_trick_array[(i*(i+3)) // 2] for i in range(self.dim+2))


    @ft.cached_property
    def is_acyclic(self):
        return not any(self.reduced_betti_numbers.values())
    
    def draw(self, show: bool=True, **kwargs):
        nx.draw(self.graph, pos=nx.spring_layout(self.graph), **kwargs)
        if show:
            plt.show()


class MultidegreeComplex(SimplicialComplex):
    def __init__(self, multidegree: np.ndarray, semigroup_generators: np.ndarray):
        super().__init__(self._compute_max_faces(multidegree, semigroup_generators))
        self.multidegree = multidegree
        self.semigroup_generators = semigroup_generators

    @staticmethod
    def is_in_cone(lattice_points, target_point):
        return linprog(np.zeros(lattice_points.shape[1]), A_eq=lattice_points, b_eq=target_point, bounds=(0, None), integrality=1).success
    
    def _compute_max_faces(self, multidegree, semigroup_generators):
        # faces are represented with indices
        all_faces = [set(face) for face in mit.powerset(range(semigroup_generators.shape[1]))
                                    if self.is_in_cone(semigroup_generators, multidegree - semigroup_generators[:, face].sum(axis=1))]
        all_faces = [face for face in all_faces if face]
        return [face for face in all_faces if sum(face <= other_face for other_face in all_faces) == 1]


class AlexanderDual(SimplicialComplex):
    def __init__(self, dual_maximal_faces):
        vertex_set = set(mit.flatten(dual_maximal_faces))
        super().__init__([set(A) for A in mit.powerset(vertex_set) if not any(vertex_set - set(A) <= dual_face for dual_face in dual_maximal_faces)])
        self.dual_maximal_faces = dual_maximal_faces


class StanleyReisnerComplex(SimplicialComplex):
    def __init__(self, nonfaces):
        vertex_set = set(mit.flatten(nonfaces))
        super().__init__([vertex_set - nonface for nonface in nonfaces])
        self.nonfaces = nonfaces


def main():
    delta = SimplicialComplex([{1}, 
                               {2,3}, {3,8}, {8,2}, 
                               {4,5}, {5,8}, {8,4}, 
                               {6,7}, {7,8}, {8,6}, 
                               {9,10}, {10,18}, {18,17}, {17,9},
                               {11,12}, {12,13}, {13,18}, {18,11},
                               {14,15}, {15,16}, {16,18}, {18,14},
                               {19,20}, {20,21}, {21,22}, {22,23}, {23,19}])
    delta.draw()
    # for U in it.combinations(delta.vertices, r=4):
    #     rest_delta = delta.restriction(set(U))
    #     # print(f'{set(U)=}')
    #     # print(f'{rest_delta.maximal_faces=}')
    #     print(f'{rest_delta.betti_numbers=}')
    #     # rest_delta.draw(with_labels=True)


if __name__ == '__main__':
    main()