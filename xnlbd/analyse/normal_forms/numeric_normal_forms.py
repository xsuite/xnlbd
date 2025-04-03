from __future__ import annotations

from typing import Sequence

import numpy as np

from .polynom_base import Map, Polynom, Term


class NormalForm4D:
    """
    Class representing the 4D normal form obtained from a 4D one-turn map of
    an accelerator lattice. It includes the conjugating transformation and its
    inverse up to the desired order, the normal form U of the map, as well as
    the interpolating Hamiltonian.
    """

    def __init__(
        self,
        complex_map: Map,
        max_map_order: int,
        max_nf_order: int,
        res_space_dim: int,
        res_case: int,
        res_eig: list[complex] | None = None,
        res_basis1: list[int] | None = None,
        res_basis2: list[int] | None = None,
    ) -> None:
        """
        Initialiser for the NormalForm4D class.

        Input:
            - complex_map: Map, 4D polynomial one-turn map in complex normalised
              Courant-Snyder coordinates
            - max_map_order: integer, maximum polynomial order in the complex
              map
            - max_nf_order: integer, desired maximum polynomial order in the
              normal form
            - res_space_dim: integer, dimension of the resonant space, 0 if
              nonresonant, 1 if single resonance, 2 if double resonance
            - res_case: integer, 0 for nonresonant normal forms, 1 for exactly
              resonant normal forms, 2 for quasiresonant normal forms
            - res_eig: list of complex resonant eigenvalues,
              i.e. [$e^{2i\pi Q_{x,res}}$, $e^{-2i\pi Q_{x,res}}$,
              $e^{2i\pi Q_{y,res}}$, $e^{-2i\pi Q_{y,res}}$], only needed if
              res_case is 1 or 2
            - res_basis1: list or integers, [n, m] which satisfy the resonance
              condition n*Q_x+m*Q_y=p, only needed if res_space_dim is 1 or 1
            - res_basis2: list or integers, [n, m] which satisfy the resonance
              condition n*Q_x+m*Q_y=p for a second resonance, only needed if
              res_space_dim is 2

        Output:
            -
        """

        # One-turn map in complex real space coordinates
        self._complex_one_turn_map: Map | None = None
        # Map order
        self.f_order: int = 0

        # Normal form order
        self.nf_order: int = 0
        # Various orders for computation
        self.nord: int = 0
        self.N_dim: int = 0
        self.N_dim_H: int = 0

        # Dimension of the resonant space of exponents
        # (0: nonresonant case)
        # (1: single resonance)
        # (2: double resonance)
        self.res_space_dim: int = 0
        # Resonance case
        # (1: nonresonant or exactly resonant case)
        # (2: quasi-resonant case)
        self.res_case: int = 2
        # True linear eigenvalues
        self.lin_eig: list[complex] = [0] * 4
        # Resonant eigenvalues
        self.res_eig: list[complex] = [0] * 4
        # First basis array of the resonant space (used if res_space_dim = 1, 2)
        self.res_basis1: list[int] = [0] * 2
        # First basis array of the resonant space (used if res_space_dim = 2)
        self.res_basis2: list[int] = [0] * 2

        # Field
        self._A: list[list[complex]] | None = None
        # One-turn-map in vector form
        self._F: list[list[complex]] | None = None
        # Transformation from complex real space to normal form coordinates
        self._Psi: list[list[complex]] | None = None
        self.Psi: Map | None = None
        # Transformation from normal form coordinates to complex real space
        self._Phi: list[list[complex]] | None = None
        self.Phi: Map | None = None
        # Normal form
        self._U: list[list[complex]] | None = None
        self.U: Map | None = None
        # Hamiltonian
        self._H: list[complex] | None = None
        self.H: Polynom | None = None

        # Dictionaries to keep track of indices of terms and derivatives
        self._idx_dict: dict[tuple[int, int, int, int], int] = {}
        self._inv_idx_dict: dict[int, tuple[int, int, int, int]] = {}
        self._idx_deriv: list[list[int]] | None = None

        # Intermediate results
        self._Resto1: list[list[complex]] | None = None
        self._Resto2: list[list[complex]] | None = None

        if res_case != 0 and res_eig is None:
            raise ValueError(
                "The resonant eigenvalues must be provided in the resonant and quasiresonant case!"
            )
        if res_space_dim == 1 and res_basis1 is None:
            raise ValueError(
                "The first resonant basis (res_basis1) must be provided in the 1D resonance case!"
            )
        if res_space_dim == 2:
            if res_basis1 is None or res_basis2 is None:
                raise ValueError(
                    "Both the first and second resonant bases (res_basis1, res_basis2) must be provided in the 2D resonance case!"
                )
        self.res_space_dim = res_space_dim
        self.res_case = res_case
        if res_eig is not None:
            self.res_eig = res_eig
        if res_basis1 is not None:
            self.res_basis1[0] = res_basis1[0]
            self.res_basis2[0] = res_basis1[1]
        if res_basis2 is not None:
            self.res_basis1[1] = res_basis2[0]
            self.res_basis2[1] = res_basis2[1]

        self._complex_one_turn_map = complex_map

        self.nf_order = max_nf_order
        self.nord = np.max([max_map_order, max_nf_order]) + 1
        self.N_dim = int(
            (self.nord + 4) * (self.nord + 3) * (self.nord + 2) * (self.nord + 1) / 24
        )
        self.N_dim_H = int(
            (self.nord + 5) * (self.nord + 4) * (self.nord + 3) * (self.nord + 2) / 24
        )

        self._A = [[0 + 0j] * self.N_dim for i in range(4)]
        self._F = [[0 + 0j] * self.N_dim for i in range(4)]
        self._Psi = [[0 + 0j] * self.N_dim for i in range(4)]
        self._Phi = [[0 + 0j] * self.N_dim for i in range(4)]
        self._U = [[0 + 0j] * self.N_dim for i in range(4)]
        self._H = [0 + 0j] * self.N_dim_H

        i = 0
        for nord in range(0, self.nord + 2):
            for n1 in range(0, nord + 1):
                for n2 in range(0, nord - n1 + 1):
                    for n3 in range(0, nord - n1 - n2 + 1):
                        n4 = nord - n1 - n2 - n3
                        self._idx_dict[(n1, n2, n3, n4)] = i
                        self._inv_idx_dict[i] = (n1, n2, n3, n4)
                        i += 1
        i = 0
        self._idx_deriv = [[0] * self.N_dim for i in range(4)]
        for nord in range(0, self.nord + 1):
            for n1 in range(0, nord + 1):
                for n2 in range(0, nord - n1 + 1):
                    for n3 in range(0, nord - n1 - n2 + 1):
                        n4 = nord - n1 - n2 - n3
                        self._idx_deriv[0][i] = self._idx_dict[(n1 + 1, n2, n3, n4)]
                        self._idx_deriv[1][i] = self._idx_dict[(n1, n2 + 1, n3, n4)]
                        self._idx_deriv[2][i] = self._idx_dict[(n1, n2, n3 + 1, n4)]
                        self._idx_deriv[3][i] = self._idx_dict[(n1, n2, n3, n4 + 1)]
                        i += 1

        self.f_order = max_map_order
        self._init_F()
        for i in range(1, 5):
            self.lin_eig[i - 1] = self._F[i - 1][5 - i]

        self._Phi[0][4] = 1.0
        self._Phi[1][3] = 1.0
        self._Phi[2][2] = 1.0
        self._Phi[3][1] = 1.0

        self._Resto1 = [[0] * self.N_dim for i in range(4)]
        self._Resto2 = [[0] * self.N_dim for i in range(4)]

    def _init_F(self) -> None:
        """
        Function that initialises the internal representation of the complex
        one-turn map for the normal form calculation.

        Input:
            -

        Output:
            -
        """

        if self._complex_one_turn_map is None:
            raise ValueError("Complex one-turn map is not provided!")
        else:
            if self._F is None:
                self._F = [[0 + 0j] * self.N_dim for i in range(4)]
            for i in range(len(self._complex_one_turn_map.x_poly.terms)):
                idx = self._idx_dict[
                    (
                        self._complex_one_turn_map.x_poly.terms[i].x_exp,
                        self._complex_one_turn_map.x_poly.terms[i].px_exp,
                        self._complex_one_turn_map.x_poly.terms[i].y_exp,
                        self._complex_one_turn_map.x_poly.terms[i].py_exp,
                    )
                ]
                self._F[0][idx] = self._complex_one_turn_map.x_poly.terms[i].coeff
            for i in range(len(self._complex_one_turn_map.px_poly.terms)):
                idx = self._idx_dict[
                    (
                        self._complex_one_turn_map.px_poly.terms[i].x_exp,
                        self._complex_one_turn_map.px_poly.terms[i].px_exp,
                        self._complex_one_turn_map.px_poly.terms[i].y_exp,
                        self._complex_one_turn_map.px_poly.terms[i].py_exp,
                    )
                ]
                self._F[1][idx] = self._complex_one_turn_map.px_poly.terms[i].coeff
            for i in range(len(self._complex_one_turn_map.y_poly.terms)):
                idx = self._idx_dict[
                    (
                        self._complex_one_turn_map.y_poly.terms[i].x_exp,
                        self._complex_one_turn_map.y_poly.terms[i].px_exp,
                        self._complex_one_turn_map.y_poly.terms[i].y_exp,
                        self._complex_one_turn_map.y_poly.terms[i].py_exp,
                    )
                ]
                self._F[2][idx] = self._complex_one_turn_map.y_poly.terms[i].coeff
            for i in range(len(self._complex_one_turn_map.py_poly.terms)):
                idx = self._idx_dict[
                    (
                        self._complex_one_turn_map.py_poly.terms[i].x_exp,
                        self._complex_one_turn_map.py_poly.terms[i].px_exp,
                        self._complex_one_turn_map.py_poly.terms[i].y_exp,
                        self._complex_one_turn_map.py_poly.terms[i].py_exp,
                    )
                ]
                self._F[3][idx] = self._complex_one_turn_map.py_poly.terms[i].coeff

    def _init_prod_4(
        self, max_q: int, max_p: int, max_r: int
    ) -> tuple[list[int], list[list[int]], list[int], list[int]]:
        """
        Subroutine to initialise the address arrays for the product of two
        polynomials in 4 variables.
        """

        N_ps = int(
            (self.nord + 8)
            * (self.nord + 7)
            * (self.nord + 6)
            * (self.nord + 5)
            * self.N_dim
            / 1680
        )
        n_pairs = [0] * N_ps
        indp = [0] * N_ps
        indq = [0] * N_ps
        n_entry = [[0] * (self.nord + 2) for i in range(self.N_dim + 1)]
        n_pairs[0] = 0
        l_count = 0
        max_p = np.min([max_r, max_p])
        max_q = np.min([max_r, max_q])
        ip = 0
        for j in range(0, max_r + 1):  # [2]
            for k1 in range(0, j + 1):  # [2]
                for k2 in range(0, j - k1 + 1):  # [2]
                    for k3 in range(0, j - k1 - k2 + 1):  # [2]
                        ip = ip + 1
                        n_entry[ip][np.max([0, j - max_q])] = 0
                        k4 = j - k1 - k2 - k3
                        l_ps = 0
                        for np_ in range(
                            np.max([0, j - max_q]), np.min([j, max_p]) + 1
                        ):  # [4]
                            nq = j - np_
                            for j1 in range(
                                np.max([0, k1 - nq]), np.min([np_, k1]) + 1
                            ):  # [3]
                                i1 = k1 - j1
                                for j2 in range(
                                    np.max([0, k2 - nq + i1]),
                                    np.min([np_ - j1, k2]) + 1,
                                ):  # [3]
                                    i2 = k2 - j2
                                    for j3 in range(
                                        np.max([0, k3 - nq + i1 + i2]),
                                        np.min([np_ - j1 - j2, k3]) + 1,
                                    ):  # [3]
                                        i3 = k3 - j3
                                        j4 = np_ - j1 - j2 - j3
                                        i4 = k4 - j4
                                        l_count = l_count + 1
                                        l_ps = l_ps + 1
                                        indp[l_count] = self._idx_dict[(j1, j2, j3, j4)]
                                        indq[l_count] = self._idx_dict[(i1, i2, i3, i4)]
                            n_entry[ip][np_ + 1] = l_ps
                        n_pairs[ip] = l_count

        return n_pairs, n_entry, indp, indq

    def _prodo(
        self,
        Q: list[complex],
        P: list[complex],
        R: list[complex],
        max_q: int,
        min_p: int,
        min_r: int,
        max_r: int,
        n_pairs: list[int],
        n_entry: list[list[int]],
        indp: list[int],
        indq: list[int],
    ) -> list[complex]:
        """
        Subroutine to compute the product of two polynomials in 4 variables.
        """

        max_nt = self._idx_dict[(max_r, 0, 0, 0)]
        for j in range(0, max_nt + 1):
            R[j] = 0.0
        for nord in range(min_r, max_r + 1):
            min_j = self._idx_dict[(0, 0, 0, nord)] + 1
            max_j = self._idx_dict[(nord, 0, 0, 0)] + 1
            min_pp = np.max([min_p, nord - max_q])
            for j in range(min_j, max_j + 1):
                for nj in range(
                    n_pairs[j - 1] + n_entry[j][min_pp] + 1, n_pairs[j] + 1
                ):
                    R[j - 1] = R[j - 1] + Q[indq[nj]] * P[indp[nj]]

        return R

    def _lie_derivative(
        self,
        A: list[list[complex]],
        C: list[list[complex]],
        R: list[list[complex]],
        n_a: int,
        min_c: int,
        n_c: int,
        min_r: int,
        max_r: int,
        n_pairs: list[int],
        n_entry: list[list[int]],
        indp: list[int],
        indq: list[int],
    ) -> list[list[complex]]:
        """
        Subroutine to compute the Lie-derivative of C with respect to the
        vector field A.
        """
        if self._idx_deriv is None:
            raise ValueError

        D = [[complex(0, 0)] * self.N_dim for i in range(4)]
        B = [complex(0, 0)] * self.N_dim

        max_nt = self._idx_dict[(max_r, 0, 0, 0)]
        for j in range(0, 4):
            i_count = 0
            for nord in range(0, n_c):
                for n1 in range(0, nord + 1):
                    for n2 in range(0, nord - n1 + 1):
                        for n3 in range(0, nord - n1 - n2 + 1):
                            n4 = nord - n1 - n2 - n3
                            D[0][i_count] = (n1 + 1) * C[j][self._idx_deriv[0][i_count]]
                            D[1][i_count] = (n2 + 1) * C[j][self._idx_deriv[1][i_count]]
                            D[2][i_count] = (n3 + 1) * C[j][self._idx_deriv[2][i_count]]
                            D[3][i_count] = (n4 + 1) * C[j][self._idx_deriv[3][i_count]]
                            i_count += 1

            B = self._prodo(
                A[0],
                D[0],
                B,
                n_a,
                min_c - 1,
                min_r,
                max_r,
                n_pairs,
                n_entry,
                indp,
                indq,
            )
            for i_count in range(0, max_nt + 1):
                R[j][i_count] = B[i_count]
            B = self._prodo(
                A[1],
                D[1],
                B,
                n_a,
                min_c - 1,
                min_r,
                max_r,
                n_pairs,
                n_entry,
                indp,
                indq,
            )
            for i_count in range(0, max_nt + 1):
                R[j][i_count] = R[j][i_count] + B[i_count]
            B = self._prodo(
                A[2],
                D[2],
                B,
                n_a,
                min_c - 1,
                min_r,
                max_r,
                n_pairs,
                n_entry,
                indp,
                indq,
            )
            for i_count in range(0, max_nt + 1):
                R[j][i_count] = R[j][i_count] + B[i_count]
            B = self._prodo(
                A[3],
                D[3],
                B,
                n_a,
                min_c - 1,
                min_r,
                max_r,
                n_pairs,
                n_entry,
                indp,
                indq,
            )
            for i_count in range(0, max_nt + 1):
                R[j][i_count] = R[j][i_count] + B[i_count]

        return R

    def _lie_transformation(
        self,
        A: list[list[complex]],
        S: list[list[complex]],
        C: list[list[complex]],
        max_a: int,
        max_r: int,
    ) -> list[list[complex]]:
        """
        Soubroutine to compute the Lie transformation.
        """

        R = [[complex(0, 0)] * self.N_dim for i in range(4)]
        Q = [[complex(0, 0)] * self.N_dim for i in range(4)]

        max_nt = self._idx_dict[(max_r, 0, 0, 0)]
        for i_term in range(0, max_nt + 1):
            for j in range(0, 4):
                Q[j][i_term] = S[j][i_term]
                C[j][i_term] = Q[j][i_term]
        n_pairs, n_entry, indp, indq = self._init_prod_4(max_a, max_r - 1, max_r)

        min_r = 1
        min_c = 0
        for nord in range(1, max_r):
            min_r = min_r + 1
            min_c = min_c + 1

            R = self._lie_derivative(
                A, Q, R, max_a, min_c, max_r, min_r, max_r, n_pairs, n_entry, indp, indq
            )
            for i_term in range(0, max_nt + 1):
                for j in range(0, 4):
                    Q[j][i_term] = R[j][i_term] / (nord)
            for i_term in range(0, max_nt + 1):
                for j in range(0, 4):
                    C[j][i_term] = C[j][i_term] + Q[j][i_term]

        return C

    @staticmethod
    def _comp_lin(
        P: list[list[complex]], vec: list[complex], Q: list[list[complex]], Max: int
    ) -> list[list[complex]]:
        """ """

        rap14 = vec[0] / vec[3]
        rap24 = vec[1] / vec[3]
        rap34 = vec[2] / vec[3]
        i_count = 1
        eval_ = complex(1.0, 0)

        for nord in range(1, Max + 1):
            eval_ = vec[3] * eval_
            eval1 = eval_
            for n1 in range(0, nord + 1):
                eval2 = eval1
                for n2 in range(0, nord - n1 + 1):
                    eval3 = eval2
                    for n3 in range(0, nord - n1 - n2 + 1):
                        n4 = nord - n1 - n2 - n3
                        for j in range(0, 4):
                            Q[j][i_count] = P[j][i_count] * eval3
                        i_count += 1
                        eval3 *= rap34
                    eval2 *= rap24
                eval1 *= rap14

        return Q

    @staticmethod
    def fmod(a: int, p: int) -> int:
        return a - (int(a / p) * p)

    def _solve_hom_eq(
        self,
        Phi: list[list[complex]],
        U: list[list[complex]],
        Resto: list[list[complex]],
        max_r: int,
    ) -> tuple[
        list[list[complex]],
        list[list[complex]],
        list[list[complex]],
    ]:
        """
        Subroutine to solve the homological equation.
        """
        if self._A is None:
            raise ValueError

        k = [[1, -1, 0, 0], [0, 0, 1, -1]]
        C = [[0] * self.N_dim for i in range(4)]
        D = [[0] * self.N_dim for i in range(4)]

        n_term1 = self._idx_dict[(max_r - 1, 0, 0, 0)]
        n_term2 = self._idx_dict[(max_r, 0, 0, 0)] + 1

        # Loop over all terms of order n discriminating the terms that are
        # integer combinations of the resonant vectors
        i_norm1 = self.res_basis1[0] ** 2 + self.res_basis2[0] ** 2
        i_norm2 = self.res_basis1[1] ** 2 + self.res_basis2[1] ** 2
        i_prodv = (
            self.res_basis1[0] * self.res_basis2[1]
            - self.res_basis2[0] * self.res_basis1[1]
        )
        i_count = n_term1
        pd1 = complex(1.0, 0)
        pd4 = self.lin_eig[3] ** max_r
        rap14 = self.lin_eig[0] / self.lin_eig[3]
        rap24 = self.lin_eig[1] / self.lin_eig[3]
        rap34 = self.lin_eig[2] / self.lin_eig[3]
        for n1 in range(0, max_r + 1):
            pd2 = pd1
            for n2 in range(0, max_r - n1 + 1):
                pd3 = pd2
                for n3 in range(0, max_r - n1 - n2 + 1):
                    n4 = max_r - n1 - n2 - n3
                    i_count = i_count + 1
                    pd = pd3 * pd4
                    for j in range(0, 4):
                        pden = pd - self.lin_eig[j]
                        k1 = n1 - n2 - k[0][j]
                        k2 = n3 - n4 - k[1][j]
                        i_mod1 = 1
                        i_mod2 = 1
                        if i_norm2 != 0:
                            i_prodv1 = k1 * self.res_basis2[0] - k2 * self.res_basis1[0]
                            i_prodv2 = k1 * self.res_basis2[1] - k2 * self.res_basis1[1]
                            i_mod2 = NormalForm4D.fmod(-i_prodv1, i_prodv)
                            i_mod1 = NormalForm4D.fmod(i_prodv2, i_prodv)
                        else:
                            if i_norm1 != 0:
                                iscal1 = (
                                    k1 * self.res_basis1[0] + k2 * self.res_basis2[0]
                                )
                                iscal2 = (
                                    k1 * self.res_basis2[0] - k2 * self.res_basis1[0]
                                )
                                i_mod1 = NormalForm4D.fmod(iscal1, i_norm1)
                                i_mod2 = iscal2
                        if (k1 == 0 and k2 == 0) or (i_mod1 == 0 and i_mod2 == 0):
                            self._A[j][i_count] = 0
                            U[j][i_count] = Resto[j][i_count] / self.lin_eig[j]
                        else:
                            if Resto[j][i_count] == 0:
                                self._A[j][i_count] = 0
                            else:
                                self._A[j][i_count] = Resto[j][i_count] / pden
                            Phi[j][i_count] = Phi[j][i_count] + self._A[j][i_count]
                            U[j][i_count] = 0
                    pd3 = pd3 * rap34
                pd2 = pd2 * rap24
            pd1 = pd1 * rap14
        return Phi, U, self._A

    def _inverse(
        self,
        Phi: list[list[complex]],
        Psi: list[list[complex]],
        max_r: int,
        A: list[list[complex]],
    ) -> list[list[complex]]:
        """
        Subroutine to compute the inverse.
        """

        C = [[complex(0, 0)] * self.N_dim for i in range(4)]
        R = [[complex(0, 0)] * self.N_dim for i in range(4)]

        n_term = self._idx_dict[(max_r, 0, 0, 0)]
        for j in range(0, 4):
            for i_term in range(0, n_term + 1):
                C[j][i_term] = 0
                Psi[j][i_term] = 0
                A[j][i_term] = -A[j][i_term]
            C[0][4] = 1.0
            C[1][3] = 1.0
            C[2][2] = 1.0
            C[3][1] = 1.0
            Psi[0][4] = 1.0
            Psi[1][3] = 1.0
            Psi[2][2] = 1.0
            Psi[3][1] = 1.0

        n_pairs, n_entry, indp, indq = self._init_prod_4(max_r, max_r - 1, max_r)
        min_r = 1
        min_c = 0
        for nord in range(1, max_r):
            min_r = min_r + 1
            min_c = min_c + 1
            R = self._lie_derivative(
                A, C, R, max_r, min_c, max_r, min_r, max_r, n_pairs, n_entry, indp, indq
            )
            for i_term in range(0, n_term + 1):
                for j in range(0, 4):
                    C[j][i_term] = R[j][i_term] / nord
            for i_term in range(5, n_term + 1):
                for j in range(0, 4):
                    Psi[j][i_term] += C[j][i_term]

        return Psi

    def _hamiltonian(
        self,
        A: list[list[complex]],
        H: list[complex],
        max_r: int,
    ) -> list[complex]:
        """
        Subroutine to compute the Hamiltonian.
        """

        if self._idx_deriv is None:
            raise ValueError

        max_nt = self._idx_dict[(max_r, 0, 0, 0)]

        # Initialisation of the linear part
        A[3][1] = np.log(self.lin_eig[3])
        A[2][2] = np.log(self.lin_eig[2])
        A[1][3] = np.log(self.lin_eig[1])
        A[0][4] = np.log(self.lin_eig[0])
        for n in range(1, max_r + 1):
            i_count_start = self._idx_dict[(0, 0, 0, n)]
            i_count = i_count_start
            for n1 in range(0, n + 1):
                for n2 in range(0, n - n1 + 1):
                    for n3 in range(0, n - n1 - n2 + 1):
                        H[self._idx_deriv[0][i_count]] = -A[1][i_count] / (n1 + 1)
                        i_count = i_count + 1
            i_count = i_count_start
            for n2 in range(0, n + 1):
                for n3 in range(0, n - n2 + 1):
                    H[self._idx_deriv[1][i_count]] = A[0][i_count] / (n2 + 1)
                    i_count = i_count + 1
            i_count = i_count_start
            for n3 in range(0, n + 1):
                H[self._idx_deriv[2][i_count]] = -A[3][i_count] / (n3 + 1)
                i_count = i_count + 1
            i_count = i_count_start
            H[self._idx_deriv[3][i_count]] = A[2][i_count] / (n + 1)

        return H

    def compute_normal_form(self):
        """
        Main routine for computing normal forms.
        """

        for nord in range(2, self.nf_order + 1):
            n_f = np.min([nord, self.f_order])
            n_phi = nord - 1
            n_u = nord - 1

            n_term = self._idx_dict[(nord, 0, 0, 0)]

            for j in range(0, 4):
                for i_term in range(0, n_term + 1):
                    self._Psi[j][i_term] = 0
            self._Psi[0][4] = 1.0
            self._Psi[1][3] = 1.0
            self._Psi[2][2] = 1.0
            self._Psi[3][1] = 1.0

            self._Phi = self._lie_transformation(
                self._A, self._Psi, self._Phi, n_phi, nord
            )

            # Compute the Lie transformation by _A of the map _F up to order
            # nord and store it in _Resto1
            self._Resto1 = self._lie_transformation(
                self._A, self._F, self._Resto1, n_phi, nord
            )

            # Lie transformation of Phi by using the interpolating field _U of
            # the normal form: first, compose with the linear part,
            # second, use the Lie transformation of the nonlinear part
            self._Psi = NormalForm4D._comp_lin(self._Phi, self.lin_eig, self._Psi, nord)
            self._Resto2 = self._lie_transformation(
                self._U, self._Psi, self._Resto2, n_u, nord
            )

            # Compute the remainder of the functional equation at the order nord
            n_term1 = int(nord * (nord + 1) * (nord + 2) * (nord + 3) / 24)
            n_term2 = int((nord + 1) * (nord + 2) * (nord + 3) * (nord + 4) / 24)
            for i_count in range(n_term1, n_term2 + 1):
                for j in range(0, 4):
                    self._Resto1[j][i_count] = (
                        self._Resto1[j][i_count] - self._Resto2[j][i_count]
                    )

            # Solve the homological equation, to compute _U, _A and _Phi
            self._Phi, self._U, self._A = self._solve_hom_eq(
                self._Phi, self._U, self._Resto1, nord
            )

        self._Psi = self._inverse(self._Phi, self._Psi, self.nf_order, self._A)

        # Save Phi and Psi
        Phi_x_poly = Polynom(terms=[])
        Phi_px_poly = Polynom(terms=[])
        Phi_y_poly = Polynom(terms=[])
        Phi_py_poly = Polynom(terms=[])
        for i in range(0, 4):
            curr_terms = []
            for key in self._inv_idx_dict.keys():
                try:
                    curr_terms.append(
                        Term(
                            coeff=self._Phi[i][key],
                            x_exp=self._inv_idx_dict[key][0],
                            px_exp=self._inv_idx_dict[key][1],
                            y_exp=self._inv_idx_dict[key][2],
                            py_exp=self._inv_idx_dict[key][3],
                        )
                    )
                except IndexError:
                    continue
            if i == 0:
                Phi_x_poly.terms.extend(curr_terms)
            elif i == 1:
                Phi_px_poly.terms.extend(curr_terms)
            elif i == 2:
                Phi_y_poly.terms.extend(curr_terms)
            else:
                Phi_py_poly.terms.extend(curr_terms)
        Phi_x_poly.remove_zero_terms()
        Phi_px_poly.remove_zero_terms()
        Phi_y_poly.remove_zero_terms()
        Phi_py_poly.remove_zero_terms()
        self.Phi = Map(
            x_poly=Phi_x_poly,
            px_poly=Phi_px_poly,
            y_poly=Phi_y_poly,
            py_poly=Phi_py_poly,
        )
        Psi_x_poly = Polynom(terms=[])
        Psi_px_poly = Polynom(terms=[])
        Psi_y_poly = Polynom(terms=[])
        Psi_py_poly = Polynom(terms=[])
        for i in range(0, 4):
            curr_terms = []
            for key in self._inv_idx_dict.keys():
                try:
                    curr_terms.append(
                        Term(
                            coeff=self._Psi[i][key],
                            x_exp=self._inv_idx_dict[key][0],
                            px_exp=self._inv_idx_dict[key][1],
                            y_exp=self._inv_idx_dict[key][2],
                            py_exp=self._inv_idx_dict[key][3],
                        )
                    )
                except IndexError:
                    continue
            if i == 0:
                Psi_x_poly.terms.extend(curr_terms)
            elif i == 1:
                Psi_px_poly.terms.extend(curr_terms)
            elif i == 2:
                Psi_y_poly.terms.extend(curr_terms)
            else:
                Psi_py_poly.terms.extend(curr_terms)
        Psi_x_poly.remove_zero_terms()
        Psi_px_poly.remove_zero_terms()
        Psi_y_poly.remove_zero_terms()
        Psi_py_poly.remove_zero_terms()
        self.Psi = Map(
            x_poly=Psi_x_poly,
            px_poly=Psi_px_poly,
            y_poly=Psi_y_poly,
            py_poly=Psi_py_poly,
        )

        n_coef = self._idx_dict[(self.nf_order, 0, 0, 0)] + 1
        if self.res_case == 2:
            # Compute the normal form associated to the field U and store it in
            # Phi for the moment
            for j in range(0, 4):
                for i_term in range(0, n_coef):
                    self._Psi[j][i_term] = 0
            self._Psi[0][4] = 1
            self._Psi[1][3] = 1
            self._Psi[2][2] = 1
            self._Psi[3][1] = 1

            self._Phi = self._lie_transformation(
                self._U, self._Psi, self._Phi, self.nf_order, self.nf_order
            )

            # Redefine the map F for the resonant case
            for j in range(0, 4):
                ratio = self.lin_eig[j] / self.res_eig[j]
                for i_count in range(0, n_coef):
                    self._F[j][i_count] = self._Phi[j][i_count] * ratio
                    self._U[j][i_count] = 0
                    self._Phi[j][i_count] = 0
                    self._A[j][i_count] = 0
            self.res_basis1[0] = 0
            self.res_basis2[0] = 0
            self.res_basis1[1] = 0
            self.res_basis2[1] = 0

            # Initialise _Phi and _U and compute the nonresonant normal form
            for j in range(0, 4):
                self._Phi[j][0] = 0
                self._U[j][0] = 0
            for n_count in range(1, 5):
                self.lin_eig[n_count - 1] = self._F[n_count - 1][5 - n_count]
                for j in range(0, 4):
                    self._U[j][n_count] = 0
                    self._Phi[j][n_count] = 0
                    self._A[j][n_count] = 0
            self._Phi[0][4] = 1
            self._Phi[1][3] = 1
            self._Phi[2][2] = 1
            self._Phi[3][1] = 1

            # Main loop
            for nord in range(2, self.nf_order + 1):
                n_f = nord
                n_phi = nord - 1
                n_u = nord - 1

                # Write _Phi as a Lie transformation up to order nord
                n_term = self._idx_dict[(nord, 0, 0, 0)]
                for j in range(0, 4):
                    for i_term in range(0, n_term + 1):
                        self._Psi[j][i_term] = 0
                self._Psi[0][4] = 1
                self._Psi[1][3] = 1
                self._Psi[2][2] = 1
                self._Psi[3][1] = 1

                # Compute the Lie transformation of the vector field A: i.e. the
                # normalizing transformation Phi
                self._Phi = self._lie_transformation(
                    self._A, self._Psi, self._Phi, n_phi, nord
                )
                self._Resto1 = self._lie_transformation(
                    self._A, self._F, self._Resto1, n_phi, nord
                )
                self._Psi = self._comp_lin(self._Phi, self.lin_eig, self._Psi, nord)
                self._Resto2 = self._lie_transformation(
                    self._U, self._Psi, self._Resto2, n_u, nord
                )
                n_term1 = int(nord * (nord + 1) * (nord + 2) * (nord + 3) / 24)
                n_term2 = int((nord + 1) * (nord + 2) * (nord + 3) * (nord + 4) / 24)
                for i_count in range(n_term1, n_term2 + 1):
                    for j in range(0, 4):
                        self._Resto1[j][i_count] = (
                            self._Resto1[j][i_count] - self._Resto2[j][i_count]
                        )

                # Solve the homological equation
                self._Phi, self._U, self._A = self._solve_hom_eq(
                    self._Phi, self._U, self._Resto1, nord
                )

            # Change the sign of A to return to the resonant variables by
            # means of a Lie transformation
            for j in range(0, 4):
                for i_count in range(1, n_coef):
                    self._A[j][i_count] = -self._A[j][i_count]
            # Add  the linear part to the field U
            self._U[3][1] = np.log(self.lin_eig[3])
            self._U[2][2] = np.log(self.lin_eig[2])
            self._U[1][3] = np.log(self.lin_eig[1])
            self._U[0][4] = np.log(self.lin_eig[0])
            # Back to resonant variables
            n_pairs_, n_entry_, indp_, indq_ = self._init_prod_4(
                self.nf_order, self.nf_order - 1, self.nf_order
            )
            min_r = 1
            min_c = 1
            self._Psi = self._lie_derivative(
                self._U,
                self._Phi,
                self._Psi,
                self.nf_order,
                min_c,
                self.nf_order,
                min_r,
                self.nf_order,
                n_pairs_,
                n_entry_,
                indp_,
                indq_,
            )
            self._U = self._lie_transformation(
                self._A, self._Psi, self._U, self.nf_order, self.nf_order
            )

        self._H = self._hamiltonian(self._U, self._H, self.nf_order + 1)

        U_x_poly = Polynom(terms=[])
        U_px_poly = Polynom(terms=[])
        U_y_poly = Polynom(terms=[])
        U_py_poly = Polynom(terms=[])
        for i in range(0, 4):
            curr_terms = []
            for key in self._inv_idx_dict.keys():
                try:
                    curr_terms.append(
                        Term(
                            coeff=self._U[i][key],
                            x_exp=self._inv_idx_dict[key][0],
                            px_exp=self._inv_idx_dict[key][1],
                            y_exp=self._inv_idx_dict[key][2],
                            py_exp=self._inv_idx_dict[key][3],
                        )
                    )
                except IndexError:
                    continue
            if i == 0:
                U_x_poly.terms.extend(curr_terms)
            elif i == 1:
                U_px_poly.terms.extend(curr_terms)
            elif i == 2:
                U_y_poly.terms.extend(curr_terms)
            else:
                U_py_poly.terms.extend(curr_terms)
        U_x_poly.remove_zero_terms()
        U_px_poly.remove_zero_terms()
        U_y_poly.remove_zero_terms()
        U_py_poly.remove_zero_terms()
        self.U = Map(
            x_poly=U_x_poly, px_poly=U_px_poly, y_poly=U_y_poly, py_poly=U_py_poly
        )

        self.H = Polynom(terms=[])
        curr_terms = []
        for key in self._inv_idx_dict.keys():
            try:
                curr_terms.append(
                    Term(
                        coeff=self._H[key],
                        x_exp=self._inv_idx_dict[key][0],
                        px_exp=self._inv_idx_dict[key][1],
                        y_exp=self._inv_idx_dict[key][2],
                        py_exp=self._inv_idx_dict[key][3],
                    )
                )
            except IndexError:
                continue
        self.H.terms = curr_terms
        self.H.remove_zero_terms()
        self.H.collect_terms()

    def norm_to_nf(
        self,
        x_norm: np.ndarray,
        px_norm: np.ndarray,
        y_norm: np.ndarray,
        py_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to transform from complex normalised Courant-Snyder
        coordinates to normal form coordinates.

        Input:
            - x_norm: array-like, Courant-Snyder normalised x coordinates
            - px_norm: array-like, Courant-Snyder normalised px coordinates
            - y_norm: array-like, Courant-Snyder normalised y coordinates
            - py_norm: array-like, Courant-Snyder normalised py coordinates

        Output:
            - tupe of 4 arry-like objects, the values of the 4 complex normal
              form coordinates
        """

        z1 = x_norm - 1j * px_norm
        z1s = x_norm + 1j * px_norm
        z2 = y_norm - 1j * py_norm
        z2s = y_norm + 1j * py_norm

        if self.Psi is None:
            raise ValueError("Conjugating function has not been evaluated!")

        return self.Psi.substitute(z1, z1s, z2, z2s)

    def nf_to_norm(
        self,
        zeta1: np.ndarray,
        zeta1s: np.ndarray,
        zeta2: np.ndarray,
        zeta2s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Function to transform from normal form coordinates to complex
        normalised Courant-Snyder coordinates.

        Input:
            - zeta1: array-like, first complex normal form coordinates
            - zeta1s: array-like, conjugate of first complex normal form
              coordinates
            - zeta2: array-like, second complex normal form coordinates
            - zeta2s: array-like, conjugate of second complex normal form
              coordinates

        Output:
            - tupe of 4 arry-like objects, the values of the 4 normalised
              Courant-Snyder coordinates
        """

        if self.Phi is None:
            raise ValueError("Conjugating function has not been evaluated!")

        complex_norm = self.Phi.substitute(zeta1, zeta1s, zeta2, zeta2s)

        x_norm = np.asarray((complex_norm[0] + complex_norm[1]) / 2.0, dtype=float)
        px_norm = np.asarray(1j * (complex_norm[0] - complex_norm[1]) / 2.0, dtype=float)
        y_norm = np.asarray((complex_norm[2] + complex_norm[3]) / 2.0, dtype=float)
        py_norm = np.asarray(1j * (complex_norm[2] - complex_norm[3]) / 2.0, dtype=float)

        return (x_norm, px_norm, y_norm, py_norm)
