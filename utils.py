import itertools as it
from pathlib import Path
import re

import numpy as np


class BettiTable:
    def __init__(self, lattice_points, res_data):
        """The betti table of a resuolution coming from the ideal of lattice points.
        
        Following the Macaulay2 convention, entry (i,j) corresponds to the rank of
        R(-i-j) in the free resolution, i.e., the rank of the free module in the 
        i-th summand that is generated in degree j.
        """
        homogeneous_factor = np.sum(lattice_points[:,0])
        regularity = max((np.sum(summand) // homogeneous_factor) - res_idx for res_idx, summands in res_data.items() for summand in summands)
        self.table = np.zeros((regularity+1, max(res_data.keys())+1), dtype=int)
        if 0 not in res_data:
            self.table[0, 0] = 1
        for res_idx, summands in res_data.items():
            for degree, degree_summands in it.groupby(summands, lambda summand: np.sum(summand)):
                homogeneous_degree = (degree // homogeneous_factor) - res_idx
                self.table[homogeneous_degree, res_idx] = len(list(degree_summands))

    def __str__(self):
        column_widths = [6] + [max(len(str(res_idx)), len(str(np.sum(self.table[:, res_idx])))) for res_idx in range(self.table.shape[1])]
        first_column = ['', 'total:'] + [f'{deg}:' for deg in range(self.table.shape[0])]
        table_str = ' '.join([f'{first_column[0]:>{column_widths[0]}}'] + [f'{k:>{column_widths[k+1]}}' for k in range(self.table.shape[1])]) + '\n' + \
                    ' '.join([f'{first_column[1]:>{column_widths[0]}}'] + [f'{np.sum(self.table[:, res_idx]):>{column_widths[res_idx+1]}}' for res_idx in range(self.table.shape[1])]) + '\n'
        m2_betti = self.table.copy().astype(str)
        m2_betti[m2_betti == '0'] = '.'
        table_str += '\n'.join(' '.join([f'{first_column[row_idx+2]:>{column_widths[0]}}'] + [f'{betti:>{column_widths[res_idx]}}' for res_idx, betti in enumerate(m2_betti[row_idx, :].tolist(), start=1)]) for row_idx in range(self.table.shape[0]))
        return table_str
    

def find_smallest_entry(matrix):
    """Finds the index of the smallest entry in a matrix."""

    # need to look at absolute value for smallest element
    abs_matrix = abs(matrix)
    
    # find the indices of the smallest element
    # https://stackoverflow.com/a/45002906
    i, j = np.where(abs_matrix == np.min(abs_matrix[abs_matrix != 0]))
    i, j = i[0], j[0]   # need to select only one entry

    return i, j


def step_one(matrix):
    """Performs Step 1 of Artin's algorithm.
    
    1. "move smallest absolute value to top left"
    Find an entry with smallest absolute value.
    Permute rows and columns to move it to the top left entry.
    If this number is negative, multiply first row by -1.
    """

    # row-column swaps based on smallest nonzero entry
    i, j = find_smallest_entry(matrix)
    matrix[[0,i], :] = matrix[[i,0], :]
    matrix[:, [0,j]] = matrix[:, [j,0]]

    # make sure a_11 is positive
    matrix[0, :] = np.sign(matrix[0, 0]) * matrix[0, :]

    return matrix


def row_reduce(matrix):
    """Performs row reduction for Step 2 of Artin's algorithm."""

    quotients = np.floor_divide(matrix[1:, 0], matrix[0, 0])
    matrix[1:, :] = matrix[1:, :] - np.diagflat(quotients) @ np.broadcast_to(matrix[0, :], matrix[1:, :].shape)

    return matrix


def column_reduce(matrix):
    """Performs column reduction for Step 2 of Artin's algorithm."""

    quotients = np.floor_divide(matrix[0, 1:], matrix[0, 0])
    matrix[:, 1:] = matrix[:, 1:] - np.broadcast_to(matrix[:, 0], matrix[:, 1:].T.shape).T @ np.diagflat(quotients)

    return matrix


def step_two(matrix):
    """Performs Step 2 of Artin's algorithm.
    
    "try to clear out first column"
    Perform row operations to clear first column.
    If a row operation produces a nonzero entry whose absolute value is smaller than a_11, go back to step 1.
    Do the same process to clear out the first row as well.
    """

    # clear out first column
    matrix = row_reduce(matrix)
    while np.any(matrix[1:, 0]):
        matrix = step_one(matrix)
        matrix = row_reduce(matrix)

    # clear out first row
    matrix = column_reduce(matrix)
    while np.any(matrix[0, 1:]):
        matrix = step_one(matrix)
        matrix = column_reduce(matrix)

    return matrix


def step_three(matrix):
    """Performs Step 3 of Artin's algorithm.
    
    3. "Assume that a_11 is the only nonzero entry in the first row and column, but that some entry b of M is not divisible by a_11.
    We add the column of A that contains b to column 1.
    This produces an entry b in the first column.
    We go back to Step 2.
    Division with remainder produces a smaller nonzero matrix entry, sending us back to Step 1.
    """

    # check if all entries of M are divisible by a_11
    i, j = np.where(np.remainder(matrix[1:, 1:], matrix[0, 0]) != 0)
    
    # if there is some entry not divisible by a_11, add that column to the first column
    if i.size > 0 or j.size > 0:
        i, j = i[0], j[0]
        matrix[:, 0] = matrix[:, 0] + matrix[:, j]
    
        # go back to Step 2
        matrix = step_two(matrix)

    return matrix


def check_divisible(dividend, divisor):
    """Checks if dividend is a multiple of divisor."""

    # special case to avoid runtime warning
    if divisor == 0:
        divisible = True
    elif dividend % divisor != 0:
        divisible = False
    else:
        divisible = True

    return divisible
    

def smith_normal_form(matrix):
    """Reduces a matrix with integer entries according to the algorithm outlined by Artin on page 420 (432).
    
    1. "move smallest absolute value to top left"
       Find an entry with smallest absolute value.
       Permute rows and columns to move it to the top left entry.
       If this number is negative, multiply first row by -1.

    2. "try to clear out first column"
       Perform row operations to clear first column.
       If a row operation produces a nonzero entry whose absolute value is smaller than a_11, go back to step 1.
       Do the same process to clear out the first row as well.

    3. "Assume that a_11 is the only nonzero entry in the first row and column, but that some entry b of M is not divisible by a_11.
        We add the column of A that contains b to column 1.
        This produces an entry b in the first column.
        We go back to Step 2.
        Division with remainder produces a smaller nonzero matrix entry, sending us back to Step 1.
    """
    
    min_size = min(matrix.shape)

    # go through pivot columns/rows
    for i in range(min_size):
        submatrix = matrix[i:, i:]

        divisible = check_divisible(submatrix[-1, -1], submatrix[0, 0])
        # exits if all entries in first row and first column are 0 (except a_11)
        while np.any(submatrix[1:, 0]) or np.any(submatrix[0, 1:]) or not divisible:
            submatrix = step_one(submatrix)
            submatrix = step_two(submatrix)
            submatrix = step_three(submatrix)

            # have to also check that last diagonal entry is divisible by first diagonal entry
            divisible = check_divisible(submatrix[-1, -1], submatrix[0, 0])
        
        # reassign row/column-reduced submatrix to orginal submatrix
        matrix[i:, i:] = submatrix

    # check sign of bottom right entry (final step)
    matrix[-1, -1] = np.sign(matrix[-1, -1]) * matrix[-1, -1]

    return matrix


def read_resolution(filename: Path=Path().cwd()/'output.txt', 
                    is_initial: bool=False, is_polarized: bool=False):
    """Reads an input file that encodes the data from a multigraded free resolution.
    
    The data is given as (i, degrees_i) in each line of the file.
    For each i, degrees_i is a list of the multidegrees appearing in the i-th
    step of the minimal free resolution.

    If the ideal resolved is monomial, then an additional line is in the resolution
    file with the ideal printed as a list of generators.

    Parameters
    ----------
    filename : Path
        the file with the data of the resolution
    is_initial : bool
        is the ideal an initial ideal
    is_polarized : bool
        is the ideal the polarization of an ideal

    Returns
    -------
    lattice_points : List[List[int]]
        the lattice points generating the lattice ideal
    ideal : List[Set[int]]
        the list of generators of the ideal where each set corresponds to a product of monomials
        returns None if is_monomial is False.
    res_data : Dict[int, List[List[int]]]
        key-value pairs are index-multidegrees pairs
    """
    with open(filename, 'r') as f:
        raw_res_data = f.readlines()
    # read lattice points
    lattice_points = eval(raw_res_data[0].strip().replace('{', '[').replace('}', ']'))
    # read ideal (the ideal is assumed to be squarefree, monomial)
    ideal = None
    if is_initial:
        stripped_ideal = raw_res_data[1].strip()[1:-1].replace('x_', '').split(', ')
        filtered_ideal = [set(map(int, gen.split('*'))) for gen in stripped_ideal if '^' not in gen]
        ideal = [set(unique_gen) for unique_gen in set(frozenset(gen) for gen in filtered_ideal)]
    elif is_polarized:
        ideal_string = raw_res_data[1].strip()[1:-1]
        split_string = ideal_string.split('}, ')
        split_string = [part + '}' for part in split_string[:-1]] + [split_string[-1]]
        gens = []
        for part in split_string:
            numbers = re.findall(r'\d+', part)
            gens.append({(int(numbers[0]), int(numbers[1])), (int(numbers[2]), int(numbers[3]))})
        ideal = [set(unique_gen) for unique_gen in set(frozenset(gen) for gen in gens)]
    # read resolution
    res_idx = 1 if not (is_initial or is_polarized) else 2
    res_data = {}
    for raw_line in raw_res_data[res_idx:]:
        line = raw_line.strip().removeprefix('(').removesuffix(')').replace('{', '[').replace('}', ']').split(', ', maxsplit=1)
        idx, multidegrees = int(line[0]), eval(line[1])
        res_data[idx] = multidegrees
    return lattice_points, ideal, res_data