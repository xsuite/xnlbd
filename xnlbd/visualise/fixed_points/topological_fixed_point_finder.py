import copy
import itertools
import warnings
from typing import Tuple, Union

import numpy as np
import xobjects as xo  # type: ignore
import xtrack as xt  # type: ignore
from xtrack import Line  # type: ignore
from xtrack.particles.particles import Particles  # type: ignore
from xtrack.twiss import TwissTable  # type: ignore

from xnlbd.tools import NormedParticles


class FPFinder:
    def __init__(
        self,
        line: Line,
        order: int,
        planes: str,
        tol: float,
        co_guess: Union[dict[str, float], None] = None,
        verbose: int = 0,
    ):
        """
        Initialiser of the FPFinder class.

        Input:
            - line: xsuite line
            - order: integer representing the order of the fixed point
            - planes: string representing in which planes to search for the fixed point,
              should be 'H' for the horizontal plane (2D), 'V' for the vertical plane (2D),
              'L' for the longitudinal plane (2D), 'HV' for both transverse planes (4D) and
              'HVL' for all planes (6D)
            - tol: absolute tolerance for the accuracy of the fixed point
            - verbose: integer, 0 for non-verbose, 1 and 2 for different levels of verbosity

        Output:
            - no output

        Comment:
            - works only on CPU
        """

        # Check the requested plane
        if planes not in ["H", "V", "L", "HV", "HVL"]:
            raise ValueError(
                "Incorrect plane requested! Must be 'H', 'V', 'L', 'HV' or 'HVL'."
            )
        elif planes == "H":
            self.F_n = self._F_H
            self.n = 2
        elif planes == "V":
            self.F_n = self._F_V
            self.n = 2
        elif planes == "L":
            self.F_n = self._F_L
            self.n = 2
        elif planes == "HV":
            self.F_n = self._F_HV
            self.n = 4
        else:
            self.F_n = self._F_HVL
            self.n = 6
        self.planes = planes

        self.line: Line = line.copy()
        self.line.discard_tracker()
        self.line.build_tracker(_context=xo.ContextCpu())
        if co_guess == None:
            co_guess = {
                "x": 0.0,
                "px": 0.0,
                "y": 0.0,
                "py": 0.0,
                "zeta": 0.0,
                "ptau": 0.0,
            }
        self.twiss: TwissTable = line.twiss(
            continue_on_closed_orbit_error=False, co_guess=co_guess
        )
        self.order: int = order
        self.tol: float = tol
        self.verbose: int = verbose
        self.M: np.ndarray = self._M_n()
        self.edges: list[Tuple[int, int]] = self._construct_edges()

        self.max_allowed_grid_points: int = 100000

        # These variables will be set by the function that runs
        # the whole algorithm
        self.delta0: float = 0.0

        self.grid_part: Union[Particles, None] = None
        self.grid_part_norm: Union[NormedParticles, None] = None
        self.hs_part: Union[Particles, None] = None
        self.hs_part_norm: Union[NormedParticles, None] = None
        self.gbs_part: Union[Particles, None] = None
        self.gbs_part_norm: Union[NormedParticles, None] = None
        self.grid_num_pts: int = 0

    def _F_H(self, point: np.ndarray) -> np.ndarray:
        """
        Function to compute change in horizontal coordinates after
        given number of turns in machine.

        Input:
            - point: array of normalised horizontal coordinates

        Output:
            - array of change in normalised horizontal coordinates
        """

        try:
            num_points = point.shape[1]
        except IndexError:
            num_points = 1

        if num_points == 1:
            if not isinstance(self.gbs_part, Particles) or not isinstance(
                self.gbs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_gbs_parts()
            self.gbs_part_norm.x_norm = np.asarray([point[0]])
            self.gbs_part_norm.px_norm = np.asarray([point[1]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)

            self.line.track(
                self.gbs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.gbs_part_norm.phys_to_norm(self.gbs_part)

            return np.asarray(
                [
                    self.gbs_part_norm.x_norm[0] - point[0],
                    self.gbs_part_norm.px_norm[0] - point[1],
                ]
            )
        elif num_points == int(10**self.n):
            if not isinstance(self.hs_part, Particles) or not isinstance(
                self.hs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_hs_parts()
            self.hs_part_norm.x_norm = copy.deepcopy(point[0])
            self.hs_part_norm.px_norm = copy.deepcopy(point[1])
            self.hs_part = self.hs_part_norm.norm_to_phys(self.hs_part)

            self.line.track(
                self.hs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.hs_part_norm.phys_to_norm(self.hs_part)

            return np.asarray(
                [
                    self.hs_part_norm.x_norm - point[0],
                    self.hs_part_norm.px_norm - point[1],
                ]
            )
        else:
            if not isinstance(self.grid_part, Particles) or not isinstance(
                self.grid_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_grid_parts()
            self.grid_part_norm.x_norm = copy.deepcopy(point[0])
            self.grid_part_norm.px_norm = copy.deepcopy(point[1])
            self.grid_part = self.grid_part_norm.norm_to_phys(self.grid_part)

            self.line.track(
                self.grid_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.grid_part_norm.phys_to_norm(self.grid_part)

            return np.asarray(
                [
                    self.grid_part_norm.x_norm - point[0],
                    self.grid_part_norm.px_norm - point[1],
                ]
            )

    def _F_V(self, point: np.ndarray) -> np.ndarray:
        """
        Function to compute change in vertical coordinates after
        given number of turns in machine.

        Input:
            - point: array of normalised vertical coordinates

        Output:
            - array of change in normalised vertical coordinates
        """

        try:
            num_points = point.shape[1]
        except IndexError:
            num_points = 1

        if num_points == 1:
            if not isinstance(self.gbs_part, Particles) or not isinstance(
                self.gbs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_gbs_parts()
            self.gbs_part_norm.y_norm = np.asarray([point[0]])
            self.gbs_part_norm.py_norm = np.asarray([point[1]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)

            self.line.track(
                self.gbs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.gbs_part_norm.phys_to_norm(self.gbs_part)

            return np.asarray(
                [
                    self.gbs_part_norm.y_norm[0] - point[0],
                    self.gbs_part_norm.py_norm[0] - point[1],
                ]
            )
        elif num_points == int(10**self.n):
            if not isinstance(self.hs_part, Particles) or not isinstance(
                self.hs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_hs_parts()
            self.hs_part_norm.y_norm = copy.deepcopy(point[0])
            self.hs_part_norm.py_norm = copy.deepcopy(point[1])
            self.hs_part = self.hs_part_norm.norm_to_phys(self.hs_part)

            self.line.track(
                self.hs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.hs_part_norm.phys_to_norm(self.hs_part)

            return np.asarray(
                [
                    self.hs_part_norm.y_norm - point[0],
                    self.hs_part_norm.py_norm - point[1],
                ]
            )
        else:
            if not isinstance(self.grid_part, Particles) or not isinstance(
                self.grid_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_grid_parts()
            self.grid_part_norm.y_norm = copy.deepcopy(point[0])
            self.grid_part_norm.py_norm = copy.deepcopy(point[1])
            self.grid_part = self.grid_part_norm.norm_to_phys(self.grid_part)

            self.line.track(
                self.grid_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.grid_part_norm.phys_to_norm(self.grid_part)

            return np.asarray(
                [
                    self.grid_part_norm.y_norm - point[0],
                    self.grid_part_norm.py_norm - point[1],
                ]
            )

    def _F_L(self, point: np.ndarray) -> np.ndarray:
        """
        Function to compute change in longitudinal coordinates after
        given number of turns in machine.

        Input:
            - point: array of normalised longitudinal coordinates

        Output:
            - array of change in normalised longitudinal coordinates
        """

        try:
            num_points = point.shape[1]
        except IndexError:
            num_points = 1

        if num_points == 1:
            if not isinstance(self.gbs_part, Particles) or not isinstance(
                self.gbs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_gbs_parts()
            self.gbs_part_norm.zeta_norm = np.asarray([point[0]])
            self.gbs_part_norm.pzeta_norm = np.asarray([point[1]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)

            self.line.track(
                self.gbs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.gbs_part_norm.phys_to_norm(self.gbs_part)

            return np.asarray(
                [
                    self.gbs_part_norm.zeta_norm[0] - point[0],
                    self.gbs_part_norm.pzeta_norm[0] - point[1],
                ]
            )
        elif num_points == int(10**self.n):
            if not isinstance(self.hs_part, Particles) or not isinstance(
                self.hs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_hs_parts()
            self.hs_part_norm.zeta_norm = copy.deepcopy(point[0])
            self.hs_part_norm.pzeta_norm = copy.deepcopy(point[1])
            self.hs_part = self.hs_part_norm.norm_to_phys(self.hs_part)

            self.line.track(self.hs_part, num_turns=self.order)

            self.hs_part_norm.phys_to_norm(self.hs_part)

            return np.asarray(
                [
                    self.hs_part_norm.zeta_norm - point[0],
                    self.hs_part_norm.pzeta_norm - point[1],
                ]
            )
        else:
            if not isinstance(self.grid_part, Particles) or not isinstance(
                self.grid_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_grid_parts()
            self.grid_part_norm.zeta_norm = copy.deepcopy(point[0])
            self.grid_part_norm.pzeta_norm = copy.deepcopy(point[1])
            self.grid_part = self.grid_part_norm.norm_to_phys(self.grid_part)

            self.line.track(self.grid_part, num_turns=self.order)

            self.grid_part_norm.phys_to_norm(self.grid_part)

            return np.asarray(
                [
                    self.grid_part_norm.zeta_norm - point[0],
                    self.grid_part_norm.pzeta_norm - point[1],
                ]
            )

    def _F_HV(self, point: np.ndarray) -> np.ndarray:
        """
        Function to compute change in transverse coordinates after
        given number of turns in machine.

        Input:
            - point: array of normalised transverse coordinates

        Output:
            - array of change in normalised transverse coordinates
        """

        try:
            num_points = point.shape[1]
        except IndexError:
            num_points = 1

        if num_points == 1:
            if not isinstance(self.gbs_part, Particles) or not isinstance(
                self.gbs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_gbs_parts()
            self.gbs_part_norm.x_norm = np.asarray([point[0]])
            self.gbs_part_norm.px_norm = np.asarray([point[1]])
            self.gbs_part_norm.y_norm = np.asarray([point[2]])
            self.gbs_part_norm.py_norm = np.asarray([point[3]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)

            self.line.track(
                self.gbs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.gbs_part_norm.phys_to_norm(self.gbs_part)

            return np.asarray(
                [
                    self.gbs_part_norm.x_norm[0] - point[0],
                    self.gbs_part_norm.px_norm[0] - point[1],
                    self.gbs_part_norm.y_norm[0] - point[2],
                    self.gbs_part_norm.py_norm[0] - point[3],
                ]
            )
        elif num_points == int(10**self.n):
            if not isinstance(self.hs_part, Particles) or not isinstance(
                self.hs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_hs_parts()
            self.hs_part.at_turn = np.zeros(num_points)
            self.hs_part.at_element = np.zeros(num_points)
            self.hs_part.state = np.ones(num_points)
            self.hs_part_norm.x_norm = copy.deepcopy(point[0])
            self.hs_part_norm.px_norm = copy.deepcopy(point[1])
            self.hs_part_norm.y_norm = copy.deepcopy(point[2])
            self.hs_part_norm.py_norm = copy.deepcopy(point[3])
            self.hs_part = self.hs_part_norm.norm_to_phys(self.hs_part)

            self.line.track(
                self.hs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.hs_part_norm.phys_to_norm(self.hs_part)

            return np.asarray(
                [
                    self.hs_part_norm.x_norm - point[0],
                    self.hs_part_norm.px_norm - point[1],
                    self.hs_part_norm.y_norm - point[2],
                    self.hs_part_norm.py_norm - point[3],
                ]
            )
        else:
            if not isinstance(self.grid_part, Particles) or not isinstance(
                self.grid_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_grid_parts()
            self.grid_part.at_turn = np.zeros(num_points)
            self.grid_part.at_element = np.zeros(num_points)
            self.grid_part.state = np.ones(num_points)
            self.grid_part_norm.x_norm = copy.deepcopy(point[0])
            self.grid_part_norm.px_norm = copy.deepcopy(point[1])
            self.grid_part_norm.y_norm = copy.deepcopy(point[2])
            self.grid_part_norm.py_norm = copy.deepcopy(point[3])
            self.grid_part = self.grid_part_norm.norm_to_phys(self.grid_part)

            self.line.track(
                self.grid_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.grid_part_norm.phys_to_norm(self.grid_part)

            return np.asarray(
                [
                    self.grid_part_norm.x_norm - point[0],
                    self.grid_part_norm.px_norm - point[1],
                    self.grid_part_norm.y_norm - point[2],
                    self.grid_part_norm.py_norm - point[3],
                ]
            )

    def _F_HVL(self, point: np.ndarray) -> np.ndarray:
        """
        Function to compute change in all 6 coordinates after
        given number of turns in machine.

        Input:
            - point: array of normalised coordinates

        Output:
            - array of change in normalised coordinates
        """

        try:
            num_points = point.shape[1]
        except IndexError:
            num_points = 1

        if num_points == 1:
            if not isinstance(self.gbs_part, Particles) or not isinstance(
                self.gbs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_gbs_parts()
            self.gbs_part_norm.x_norm = np.asarray([point[0]])
            self.gbs_part_norm.px_norm = np.asarray([point[1]])
            self.gbs_part_norm.y_norm = np.asarray([point[2]])
            self.gbs_part_norm.py_norm = np.asarray([point[3]])
            self.gbs_part_norm.zeta_norm = np.asarray([point[4]])
            self.gbs_part_norm.pzeta_norm = np.asarray([point[5]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)

            self.line.track(self.gbs_part, num_turns=self.order)

            self.gbs_part_norm.phys_to_norm(self.gbs_part)

            return np.asarray(
                [
                    self.gbs_part_norm.x_norm[0] - point[0],
                    self.gbs_part_norm.px_norm[0] - point[1],
                    self.gbs_part_norm.y_norm[0] - point[2],
                    self.gbs_part_norm.py_norm[0] - point[3],
                    self.gbs_part_norm.zeta_norm[0] - point[4],
                    self.gbs_part_norm.pzeta_norm[0] - point[5],
                ]
            )
        elif num_points == int(10**self.n):
            if not isinstance(self.hs_part, Particles) or not isinstance(
                self.hs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_hs_parts()
            self.hs_part_norm.x_norm = copy.deepcopy(point[0])
            self.hs_part_norm.px_norm = copy.deepcopy(point[1])
            self.hs_part_norm.y_norm = copy.deepcopy(point[2])
            self.hs_part_norm.py_norm = copy.deepcopy(point[3])
            self.hs_part_norm.zeta_norm = copy.deepcopy(point[4])
            self.hs_part_norm.pzeta_norm = copy.deepcopy(point[5])
            self.hs_part = self.hs_part_norm.norm_to_phys(self.hs_part)

            self.line.track(
                self.hs_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.hs_part_norm.phys_to_norm(self.hs_part)

            return np.asarray(
                [
                    self.hs_part_norm.x_norm - point[0],
                    self.hs_part_norm.px_norm - point[1],
                    self.hs_part_norm.y_norm - point[2],
                    self.hs_part_norm.py_norm - point[3],
                    self.hs_part_norm.zeta_norm - point[4],
                    self.hs_part_norm.pzeta_norm - point[5],
                ]
            )
        else:
            if not isinstance(self.grid_part, Particles) or not isinstance(
                self.grid_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            self._reset_grid_parts()
            self.grid_part_norm.x_norm = copy.deepcopy(point[0])
            self.grid_part_norm.px_norm = copy.deepcopy(point[1])
            self.grid_part_norm.y_norm = copy.deepcopy(point[2])
            self.grid_part_norm.py_norm = copy.deepcopy(point[3])
            self.grid_part_norm.zeta_norm = copy.deepcopy(point[4])
            self.grid_part_norm.pzeta_norm = copy.deepcopy(point[5])
            self.grid_part = self.grid_part_norm.norm_to_phys(self.grid_part)

            self.line.track(
                self.grid_part, num_turns=self.order, freeze_longitudinal=True
            )

            self.grid_part_norm.phys_to_norm(self.grid_part)

            return np.asarray(
                [
                    self.grid_part_norm.x_norm - point[0],
                    self.grid_part_norm.px_norm - point[1],
                    self.grid_part_norm.y_norm - point[2],
                    self.grid_part_norm.py_norm - point[3],
                    self.grid_part_norm.zeta_norm - point[4],
                    self.grid_part_norm.pzeta_norm - point[5],
                ]
            )

    def _bin_coeffs(self, i: int) -> np.ndarray:
        """
        Function that returns the binary coefficients of an integer
        number including zeros up to the coefficient of 2^(n-1).

        Input:
            - i: the integer whose binary coefficients are needed

        Output:
            - an array of integers
        """

        binary = bin(i)[2:]

        reversed_binary = list(map(int, binary[::-1]))

        all_coeffs = np.hstack(
            (reversed_binary, np.zeros(self.n - len(reversed_binary)))
        )

        return all_coeffs

    def _M_n(self) -> np.ndarray:
        """
        Funtion to return a 2^n x n array of integers (1 or -1) based on
        binary coefficients of the row and column indices.

        Input:
            - no input

        Output:
            - a 2^n x n array of integers (1 or -1)
        """

        rows = int(2**self.n)
        cols = self.n

        matrix = np.zeros((rows, cols))

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                matrix[i - 1, j - 1] = 2 * self._bin_coeffs(i - 1)[self.n - j] - 1

        return matrix

    def _get_neighbour_idx(self, i: int) -> list[int]:
        """
        Function to find the neighbouring vertices of a
        given vertex of an n-polygon.

        Input:
            - i: integer representing the vertex of the
                 n-polygon whose neighbours are seeked

        Output:
            - a list of integers representing the vertices
              that are neighbours of the input vertex
        """

        neighbour_idx = []

        for j in range(1, self.n + 1):
            neighbour_idx.append(
                int(
                    i
                    - 2 ** (self.n - j) * (2 * self._bin_coeffs(i - 1)[self.n - j] - 1)
                )
            )

        return neighbour_idx

    def _construct_edges(self) -> list[Tuple[int, int]]:
        """
        Function that computes all vertex pairs of an
        n-polygon who represent a valid edge.

        Input:
            -

        Output:
            - a list of tuples, each containing n integers
        """

        edges = []

        neighbours = self._get_neighbour_idx(1)
        for j in range(self.n):
            edges.append((1, neighbours[j]))

        for i in range(2, int(2**self.n) + 1):
            neighbours = self._get_neighbour_idx(i)
            for j in range(self.n):
                if ((i, neighbours[j]) not in edges) and (
                    (neighbours[j], i) not in edges
                ):
                    edges.append((i, neighbours[j]))

        return edges

    def _get_longest_diagonal(self, n_poly: np.ndarray) -> Tuple[int, int]:
        """
        Function that computes end vertices of the
        longest diagonal of an n-polygon.

        Input:
            - n_poly: 2^n x n array representing an
                      n-polygon

        Output:
            - tuple of integers
        """

        diagonals = []
        lengths = []

        for i in range(1, int(2**self.n) + 1):
            curr_neighbours = self._get_neighbour_idx(i)
            for j in range(1, int(2**self.n) + 1):
                if (
                    (i != j)
                    and (j not in curr_neighbours)
                    and ((i, j) not in diagonals)
                    and ((j, i) not in diagonals)
                ):
                    diagonals.append((i, j))
                    length = np.sqrt(
                        np.sum(
                            (n_poly[i - 1, :] - n_poly[j - 1, :])
                            * (n_poly[i - 1, :] - n_poly[j - 1, :])
                        )
                    )
                    lengths.append(length)

        idx = np.argmax(np.array(lengths))

        return diagonals[idx]

    def _sign(self, psi: Union[float, np.ndarray]) -> np.ndarray:
        """
        Function that returns 1 if the input is bigger than or
        equal to zero, otherwise it returns -1.

        Input:
            - psi: a real scalar value, or an array of real
              scalars

        Output:
            - array of integers (1 or -1), could be of length one
        """

        return np.where(psi >= 0, int(1), int(-1))

    def _sign_n_poly(self, n_poly: np.ndarray) -> np.ndarray:
        """
        Function that, for each point (row) in the input n-polygon
        (represented by a 2^n x n array), returns 1 if the input
        n-dimensional function applied to the point returns a value
        bigger than or equal to zero, and -1 otherwise.

        Input:
            - n_poly: n-dimensional polygon represented by a 2^n x n
            array

        Output:
            - a 2^n x n array of integers (1 or -1)
        """

        sign_F_n_poly = np.zeros(n_poly.shape)
        for i in range(n_poly.shape[0]):
            sign_F_n_poly[i, :] = self._sign(self.F_n(n_poly[i, :]))

        return sign_F_n_poly

    def _construct_n_grid(
        self, limits: list, num_pts: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Function that generates equally spaced points between
        the given limits.

        Input:
            - limits: list of minimum and maximum values along
                      each dimension
            - num_pts: number of points between minimum and
                       maximum in each dimension, default `10`

        Output:
            - a dimensions x num_pts^2 array, the
              flattened grid coordinates
            - a float representing the diagonal distance to the
              next point on the grid
        """

        if len(limits) != self.n:
            raise ValueError(
                f"Length of the limit list does not match number of dimensions ({self.n})!"
            )

        arrays = []
        dist = 0
        for i in range(self.n):
            arrays.append(
                np.linspace(limits[i][0], limits[i][1], num_pts, endpoint=True)
            )
            dist += (arrays[-1][1] - arrays[-1][0]) ** 2
        rad = np.sqrt(dist)

        grid_coords = list(np.meshgrid(*arrays))
        for i in range(self.n):
            grid_coords[i] = grid_coords[i].flatten()

        return np.asarray(grid_coords), rad

    def _grid_signs(self, grid_coords: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """
        Function that evaluates the sign vector of the functions
        F_n on each grid point.

        Input:
            - grid_coords: a dimensions x (num_pts x num_pts) array,
                           the flattened grid coordinates
            - shape: shape of original grid

        Output:
            - array of shape "shape"
        """
        vals = np.ones((self.n, np.prod(shape))) * np.nan
        for i in range(int(round(np.prod(shape) / self.grid_num_pts))):
            vals[:, i * self.grid_num_pts : (i + 1) * self.grid_num_pts] = self.F_n(
                grid_coords[:, i * self.grid_num_pts : (i + 1) * self.grid_num_pts]
            )

        signs = []
        for i in range(self.n):
            curr_signs = np.where(vals[i] >= 0, int(1), int(-1))
            curr_signs = curr_signs.reshape([shape[0], np.prod(shape[1:])])
            for j in range(1, self.n - 1):
                curr_signs = curr_signs.reshape(
                    np.hstack((shape[: j + 1], np.prod(shape[j + 1 :])))
                )
            signs.append(curr_signs)

        return np.asarray(signs)

    def _find_sign_change_per_component(self, signs: np.ndarray) -> list[np.ndarray]:
        """
        Function that evaluates where the components of the function
        F_n change sign on the grid.

        Input:
            - signs: an arrays giving the signs of F_n
                     components on the grid

        Output:
            - list of arrays containing the integer indices where the
              signs change
        """

        n_pad_list = [[0, 0]] * self.n
        n_pad = np.asarray(n_pad_list)

        changes = []
        for i in range(self.n):
            diffs = []
            for j in range(self.n):
                n_pad[j][1] = 1
                sg_ext = np.pad(signs[i], pad_width=n_pad, mode="edge")
                n_pad[j][1] = 0

                curr_diff = np.diff(sg_ext, axis=j)
                diffs.append(np.where(curr_diff != 0, 1, 0))

            full_diff = copy.deepcopy(diffs[0])
            for j in range(1, self.n):
                full_diff += diffs[j]
            full_diff = full_diff.flatten()

            changes.append(np.where(full_diff != 0)[0])

        return changes

    def _find_intersection_of_all_colours(
        self, grid_coords: np.ndarray, changes: list[np.ndarray]
    ) -> np.ndarray:
        """
        Function that returns the coordinate on the grid at which
        the all components of F_n change sign.

        Input:
            - grid_coords: the flattened grid coordinates, has as
                           many rows as dimensions
            - changes: list of arrays containing the integer indices
                       where the signs change

        Output:
            - array of coordinates of intersection point in each
              dimension
        """

        intersect_idxs = np.intersect1d(changes[0], changes[1])

        for i in range(2, self.n):
            intersect_idxs = np.intersect1d(intersect_idxs, changes[i])

        if len(intersect_idxs) == 0:
            return np.asarray([])

        intersect_coords = []
        for i in range(self.n):
            intersect_coords.append(np.mean(grid_coords[i][intersect_idxs]))

        return np.asarray(intersect_coords)

    def _construct_init_n_poly_from_sampling_on_n_sphere(
        self, centre: np.ndarray, rad: float
    ) -> np.ndarray:
        """
        Function that returns the coordinates of an n-polygon that
        contains the fixed point and is proper based on a rough
        initial guess.

        Input:
            - centre: array of coordinates in each dimension that
                      correspond to the rough eastimate of the fixed
                      point
            - rad: radius of n-sphere on which points will be sampled

        Output:
            - array containing coordinates in each dimension of
              2^dimensions number of points representing the n-polygon
        """

        num_points = 10**self.n

        # https://mathworld.wolfram.com/HyperspherePointPicking.html
        X = np.random.normal(size=(num_points, self.n))
        points = np.transpose(rad / np.sqrt(np.sum(X**2, 1, keepdims=True)) * X)
        for i in range(self.n):
            points[i] += centre[i]

        vals = self.F_n(points)
        signs_list = []
        for i in range(self.n):
            signs_list.append(np.where(vals[i] >= 0, int(1), int(-1)))
        signs = np.asarray(signs_list)

        init_points = np.ones(self.M.shape) * np.nan

        for i in range(self.M.shape[0]):
            for j in range(signs.shape[1]):
                if np.all(self.M[i, :] == signs[:, j]):
                    init_points[i, :] = points[:, j]
                    break

        return init_points

    def _construct_proper_n_poly(
        self, limits: list[list[float]], num_pts: int
    ) -> np.ndarray:
        """
        Function that returns the coordinates of an n-polygon that
        contains the fixed point and is proper.

        Input:
            - limits: list of minimum and maximum values along
                      each dimension between which to seach for a
                      rough guess of the fixed point
            - num_pts: number of points between minimum and
                       maximum in each dimension

        Output:
            - array containing coordinates in each dimension of
              2^dimensions number of points representing the n-polygon
        """

        cartesian_product = itertools.product(*limits)
        region_corners = [np.array(combination) for combination in cartesian_product]
        for i in range(len(region_corners)):
            _ = self.F_n(region_corners[i])
            if not isinstance(self.gbs_part, Particles) or not isinstance(
                self.gbs_part_norm, NormedParticles
            ):
                raise ValueError("Particle objects are not properly initialised!")
            if self.gbs_part._num_active_particles != 1:
                raise ValueError(
                    "Search region is (partially) unstable, the \
initial proper n-polygon cannot be constructed! Check the limits you provided \
and try again!"
                )

        shape = np.ones(self.n, dtype=int) * int(num_pts)

        grid_coords, rad = self._construct_n_grid(limits, num_pts)

        signs = self._grid_signs(grid_coords, shape)

        changes = self._find_sign_change_per_component(signs)

        intersect = self._find_intersection_of_all_colours(grid_coords, changes)

        if len(intersect) == 0:
            return np.asarray([])

        init_n_poly = self._construct_init_n_poly_from_sampling_on_n_sphere(
            intersect, rad
        )

        trial = 2
        while np.any(np.isnan(init_n_poly)):
            init_n_poly = self._construct_init_n_poly_from_sampling_on_n_sphere(
                intersect, trial * rad
            )
            trial += 1
            if trial > 100:
                raise RuntimeError(
                    "Unexpected failure to construct proper \
n-polygon."
                )

        return init_n_poly

    def _check_accuracy(self, n_poly: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Function to check if the accuracy with which
        fixed point is estimated is good enough.

        Input:
            - n_poly: 2^n x n array of float representing an
                      n-polygon
            - F_n: the function to apply to each row of the
                   n-polygon before computing the sign

        Output:
            - a boolean that is True if the accuracy is satisfied
            - the final estimate for the fixed point
        """

        diagonal = self._get_longest_diagonal(n_poly)

        fp_estimate = (n_poly[diagonal[0] - 1, :] + n_poly[diagonal[1] - 1, :]) / 2.0

        accuracies = self.F_n(fp_estimate)
        if not isinstance(self.gbs_part, Particles) or not isinstance(
            self.gbs_part_norm, NormedParticles
        ):
            raise ValueError("Particle objects are not properly initialised!")
        if self.gbs_part._num_active_particles != 1:
            raise RuntimeError(
                "The particle gets lost, the fixed point cannot be found."
            )

        if np.all(np.abs(accuracies) <= self.tol):
            return True, fp_estimate
        else:
            return False, fp_estimate

    def _apply_generalised_bisection(
        self, n_poly_in: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """
        Function to apply bisection in n-dimensions along
        the edges of an n-polygon.

        Input:
            - n_poly_in: 2^n x n array of floats representing
                        the starting n-polygon
            - F_n: the function to apply to each row of the
                n-polygon before computing the sign

        Output:
            - a boolean indicating the success of the operation,
            i.e. True if at least one vertex was replaced,
            False otherwise
            - a 2^n x n array of floats representing the new
            n-polygon
        """

        if self.verbose > 1:
            print("APPLYING GENERALISED BISECTION")
        n_poly = copy.deepcopy(n_poly_in)

        num_vert_replaced = 0
        for edge in self.edges:
            i = edge[0] - 1
            j = edge[1] - 1

            if self.verbose > 1:
                print("Trying to bisect edge", edge[0], "-", edge[1])

            X_i = n_poly[i, :]
            X_j = n_poly[j, :]
            X_mid = (X_j + X_i) / 2.0

            sign_F_n_X_i = self._sign(self.F_n(X_i))
            sign_F_n_X_j = self._sign(self.F_n(X_j))
            sign_F_n_X_mid = self._sign(self.F_n(X_mid))

            if np.all(sign_F_n_X_i == sign_F_n_X_mid):
                n_poly[i, :] = copy.deepcopy(X_mid)
                num_vert_replaced += 1
                if self.verbose > 1:
                    print("vertex", i + 1, "replaced at level 0")
                    print("Current n-poly:")
                    print(n_poly)
                    print(self._sign_n_poly(n_poly))
            elif np.all(sign_F_n_X_j == sign_F_n_X_mid):
                n_poly[j, :] = copy.deepcopy(X_mid)
                num_vert_replaced += 1
                if self.verbose > 1:
                    print("vertex", j + 1, "replaced at level 0")
                    print("Current n-poly:")
                    print(n_poly)
                    print(self._sign_n_poly(n_poly))
            else:
                X_u = copy.deepcopy(X_mid)
                sign_F_n_X_u = self._sign(self.F_n(X_u))
                for vert_num in range(1, int(2**self.n)):
                    if vert_num != edge[0] and vert_num != edge[1]:
                        X_u = n_poly[vert_num - 1, :]
                        sign_F_n_X_u = self._sign(self.F_n(X_u))
                        if np.all(sign_F_n_X_u == sign_F_n_X_mid):
                            break
                X_mid = 2 * X_mid - X_u
                sign_F_n_X_mid = self._sign(self.F_n(X_mid))

                if np.all(sign_F_n_X_i == sign_F_n_X_mid):
                    n_poly[i, :] = copy.deepcopy(X_mid)
                    num_vert_replaced += 1
                    if self.verbose > 1:
                        print("vertex", i + 1, "replaced at level 1")
                        print("Current n-poly:")
                        print(n_poly)
                        print(self._sign_n_poly(n_poly))
                elif np.all(sign_F_n_X_j == sign_F_n_X_mid):
                    n_poly[j, :] = copy.deepcopy(X_mid)
                    num_vert_replaced += 1
                    if self.verbose > 1:
                        print("vertex", j + 1, "replaced at level 1")
                        print("Current n-poly:")
                        print(n_poly)
                        print(self._sign_n_poly(n_poly))
                else:
                    X_u = copy.deepcopy(X_mid)
                    sign_F_n_X_u = self._sign(self.F_n(X_u))
                    for vert_num in range(1, int(2**self.n)):
                        if vert_num != edge[0] and vert_num != edge[1]:
                            X_u = n_poly[vert_num - 1, :]
                            sign_F_n_X_u = self._sign(self.F_n(X_u))
                            if np.all(sign_F_n_X_u == sign_F_n_X_mid):
                                break
                    X_mid = 2 * X_mid - X_u
                    sign_F_n_X_mid = self._sign(self.F_n(X_mid))

                    if np.all(sign_F_n_X_i == sign_F_n_X_mid):
                        n_poly[i, :] = copy.deepcopy(X_mid)
                        num_vert_replaced += 1
                        if self.verbose > 1:
                            print("vertex", i + 1, "replaced at level 2")
                            print("Current n-poly:")
                            print(n_poly)
                            print(self._sign_n_poly(n_poly))
                    elif np.all(sign_F_n_X_j == sign_F_n_X_mid):
                        n_poly[j, :] = copy.deepcopy(X_mid)
                        num_vert_replaced += 1
                        if self.verbose > 1:
                            print("vertex", j + 1, "replaced at level 2")
                            print("Current n-poly:")
                            print(n_poly)
                            print(self._sign_n_poly(n_poly))
                    else:
                        continue
        if self.verbose > 1:
            print("Current n-poly:")
            print(n_poly)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        return True if num_vert_replaced > 0 else False, n_poly

    def _reset_gbs_parts(self) -> None:
        """
        Function to reset the particle used for performing the generalised
        bisection.

        Input:
            - no input

        Output:
            - no output
        """
        if not isinstance(self.gbs_part, Particles) or not isinstance(
            self.gbs_part_norm, NormedParticles
        ):
            raise ValueError("Particle objects are not properly initialised!")
        self.gbs_part = self.twiss.particle_on_co.copy()
        self.gbs_part_norm.phys_to_norm(self.gbs_part)

    def _reset_hs_parts(self) -> None:
        """
        Function to reset the particle used for sampling on the hypersphere.

        Input:
            - no input

        Output:
            - no output
        """
        if not isinstance(self.hs_part, Particles) or not isinstance(
            self.hs_part_norm, NormedParticles
        ):
            raise ValueError("Particle objects are not properly initialised!")
        hs_num_pts = 10**self.n
        self.hs_part = xt.Particles(
            p0c=self.twiss.particle_on_co.p0c[0],
            mass0=self.twiss.particle_on_co.mass0,
            x=np.ones(hs_num_pts) * self.twiss.particle_on_co.x[0],
            px=np.ones(hs_num_pts) * self.twiss.particle_on_co.px[0],
            y=np.ones(hs_num_pts) * self.twiss.particle_on_co.y[0],
            py=np.ones(hs_num_pts) * self.twiss.particle_on_co.py[0],
            zeta=np.ones(hs_num_pts) * self.twiss.particle_on_co.zeta[0],
            ptau=np.ones(hs_num_pts) * self.twiss.particle_on_co.ptau[0],
        )
        self.hs_part_norm.phys_to_norm(self.hs_part)

    def _reset_grid_parts(self) -> None:
        """
        Function to reset the particle used for sampling on a grid.

        Input:
            - no input

        Output:
            - no output
        """
        if not isinstance(self.grid_part, Particles) or not isinstance(
            self.grid_part_norm, NormedParticles
        ):
            raise ValueError("Particle objects are not properly initialised!")
        self.grid_part = xt.Particles(
            p0c=self.twiss.particle_on_co.p0c[0],
            mass0=self.twiss.particle_on_co.mass0,
            x=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.x[0],
            px=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.px[0],
            y=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.y[0],
            py=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.py[0],
            zeta=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.zeta[0],
            ptau=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.ptau[0],
        )
        self.grid_part_norm.phys_to_norm(self.grid_part)

    def find_fp(
        self,
        limits: list[list[float]],
        delta0: float = 0.0,
        co_guess: Union[dict[str, float], None] = None,
        num_pts: int = 100,
        max_num_iter: int = 100,
        nemitt_x: float = 1e-6,
        nemitt_y: float = 1e-6,
        nemitt_z: float = 1.0,
    ) -> Tuple[dict[str, float], dict[str, float]]:
        """
        Function to run fixed point finding algorithm using bisection.

        Input:
            - limits: list of [min, max] values in all dimensions
            - delta0: float, the momentum offset of the reference particle;
              note that delta0 will only be taken into account if the fixed
              point is wanted in transverse plane(s) only, in which case RF
              should be turned off in the line for meaningful results
            - co_guess: dictionary containing the closed orbit guess in case it
              is different from 0, default `None`
            - num_pts: integer, number of points in each dimension to take
              on the initial grid for finding the rough fixed point, default
              `100`
            - max_num_iter: integer, maximum number of iterations at which
              the algorithm should stop, default `100`
            - nemitt_x: float, the normalised emittance in the horizontal plane
              used for converting to and from normalised coordinates, default
              `1e-6`
            - nemitt_y: float, normalised emittance in the vertical plane used
              for converting to and from normalised coordinates, default `1e-6`
            - nemitt_z: float, normalised emittance in the longitudinal plane
              used for converting to and from normalised coordinates, default
              `1.0`

        Output:
            - dictionary with coordinates of fixed point in real space
            - dictionary with coordinates of fixed point in normalised space
        """

        self.delta0 = delta0
        if co_guess is None:
            co_guess = {
                "x": 0.0,
                "px": 0.0,
                "y": 0.0,
                "py": 0.0,
                "zeta": 0.0,
                "ptau": 0.0,
            }
        self.twiss = self.line.twiss(
            continue_on_closed_orbit_error=False,
            delta0=self.delta0,
            co_guess=co_guess,
        )
        self.gbs_part = self.twiss.particle_on_co.copy()
        self.gbs_part_norm = NormedParticles(
            self.twiss,
            nemitt_x,
            nemitt_y,
            nemitt_z,
            part=self.gbs_part,
        )
        self.gbs_part_norm.phys_to_norm(self.gbs_part)
        self.grid_num_pts = num_pts**self.n
        if self.grid_num_pts > self.max_allowed_grid_points:
            for i in range(self.max_allowed_grid_points, 0, -1):
                if self.grid_num_pts % i == 0:
                    self.grid_num_pts = i
                    break
        self.grid_part = xt.Particles(
            p0c=self.twiss.particle_on_co.p0c[0],
            mass0=self.twiss.particle_on_co.mass0,
            x=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.x[0],
            px=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.px[0],
            y=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.y[0],
            py=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.py[0],
            zeta=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.zeta[0],
            ptau=np.ones(self.grid_num_pts) * self.twiss.particle_on_co.ptau[0],
        )
        self.grid_part_norm = NormedParticles(
            self.twiss,
            nemitt_x,
            nemitt_y,
            nemitt_z,
            part=self.grid_part,
        )
        self.grid_part_norm.phys_to_norm(self.grid_part)
        hs_num_pts = 10**self.n
        self.hs_part = xt.Particles(
            p0c=self.twiss.particle_on_co.p0c[0],
            mass0=self.twiss.particle_on_co.mass0,
            x=np.ones(hs_num_pts) * self.twiss.particle_on_co.x[0],
            px=np.ones(hs_num_pts) * self.twiss.particle_on_co.px[0],
            y=np.ones(hs_num_pts) * self.twiss.particle_on_co.y[0],
            py=np.ones(hs_num_pts) * self.twiss.particle_on_co.py[0],
            zeta=np.ones(hs_num_pts) * self.twiss.particle_on_co.zeta[0],
            ptau=np.ones(hs_num_pts) * self.twiss.particle_on_co.ptau[0],
        )
        self.hs_part_norm = NormedParticles(
            self.twiss,
            nemitt_x,
            nemitt_y,
            nemitt_z,
            part=self.hs_part,
        )
        self.hs_part_norm.phys_to_norm(self.hs_part)

        if self.verbose > 0:
            print("###############################################################")
            print("CONSTRUCTING INITIAL PROPER N-POLYGON")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        n_poly = self._construct_proper_n_poly(limits, num_pts)
        if len(n_poly) == 0:
            raise RuntimeError(
                "No panchromatic point in search region, the \
initial proper n-polygon cannot be constructed! Check the limits you provided \
and try again!"
            )
        if self.verbose > 0:
            print("DONE CONSTRUCTING INITIAL PROPER N-POLYGON")
            print("###############################################################")

        if self.verbose > 0:
            print("###############################################################")
            print("STARTING BISECTION")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        accurate, fp_estimate = self._check_accuracy(n_poly)

        num_iter = 0
        while accurate == False and num_iter < max_num_iter:
            _, n_poly = self._apply_generalised_bisection(n_poly)

            accurate, fp_estimate = self._check_accuracy(n_poly)

            num_iter += 1

        if self.verbose > 0:
            print("DONE WITH BISECTION")
            print("Number of iterations:", num_iter)
            print("Fixed point estimate:", fp_estimate)
            print("###############################################################")

        if num_iter == max_num_iter:
            warnings.warn(
                "Maximum number of iterations reached! Attention: desired \
accuracy may not have been achieved!"
            )

        if not isinstance(self.gbs_part, Particles) or not isinstance(
            self.gbs_part_norm, NormedParticles
        ):
            raise ValueError("Particle objects are not properly initialised!")
        self._reset_gbs_parts()
        if self.planes == "H":
            self.gbs_part_norm.x_norm = np.asarray([fp_estimate[0]])
            self.gbs_part_norm.px_norm = np.asarray([fp_estimate[1]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)
        elif self.planes == "V":
            self.gbs_part_norm.y_norm = np.asarray([fp_estimate[0]])
            self.gbs_part_norm.py_norm = np.asarray([fp_estimate[1]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)
        elif self.planes == "L":
            self.gbs_part_norm.zeta_norm = np.asarray([fp_estimate[0]])
            self.gbs_part_norm.pzeta_norm = np.asarray([fp_estimate[1]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)
        elif self.planes == "HV":
            self.gbs_part_norm.x_norm = np.asarray([fp_estimate[0]])
            self.gbs_part_norm.px_norm = np.asarray([fp_estimate[1]])
            self.gbs_part_norm.y_norm = np.asarray([fp_estimate[2]])
            self.gbs_part_norm.py_norm = np.asarray([fp_estimate[3]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)
        else:
            self.gbs_part_norm.x_norm = np.asarray([fp_estimate[0]])
            self.gbs_part_norm.px_norm = np.asarray([fp_estimate[1]])
            self.gbs_part_norm.y_norm = np.asarray([fp_estimate[2]])
            self.gbs_part_norm.py_norm = np.asarray([fp_estimate[3]])
            self.gbs_part_norm.zeta_norm = np.asarray([fp_estimate[4]])
            self.gbs_part_norm.pzeta_norm = np.asarray([fp_estimate[5]])
            self.gbs_part = self.gbs_part_norm.norm_to_phys(self.gbs_part)

        fp_coords_norm = {
            "x_norm": self.gbs_part_norm.x_norm,
            "px_norm": self.gbs_part_norm.px_norm,
            "y_norm": self.gbs_part_norm.y_norm,
            "py_norm": self.gbs_part_norm.py_norm,
            "zeta_norm": self.gbs_part_norm.zeta_norm,
            "pzeta_norm": self.gbs_part_norm.pzeta_norm,
        }

        fp_coords = {
            "x": self.gbs_part.x,
            "px": self.gbs_part.px,
            "y": self.gbs_part.y,
            "py": self.gbs_part.py,
            "zeta": self.gbs_part.zeta,
            "ptau": self.gbs_part.ptau,
            "pzeta": self.gbs_part.pzeta,
            "delta": self.gbs_part.delta,
            "mass0": self.gbs_part.mass0,
            "p0c": self.gbs_part.p0c,
        }

        return fp_coords, fp_coords_norm

    def update_line(
        self,
        line: Line,
        co_guess: Union[dict[str, float], None] = None,
    ) -> None:
        """
        Function to update the line associated with the fixed point finder.

        Input:
            - line: xsuite line
            - co_guess: dictionary of closed orbit guess coordinates in case it
              is different from 0, default `None`

        Output:
            - no output
        """
        if co_guess is None:
            co_guess = {
                "x": 0.0,
                "px": 0.0,
                "y": 0.0,
                "py": 0.0,
                "zeta": 0.0,
                "ptau": 0.0,
            }

        self.line = line.copy()
        self.line.discard_tracker()
        self.line.build_tracker(_context=xo.ContextCpu())
        self.twiss = line.twiss(continue_on_closed_orbit_error=False, co_guess=co_guess)
