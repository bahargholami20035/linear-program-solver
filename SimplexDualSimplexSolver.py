import numpy as np

class LinearProgramSolver:
    """Solves linear programming problems using Simplex and Dual Simplex methods."""
    
    TOLERANCE = 1e-9  # Global tolerance for numerical comparisons

    def __init__(self, objective_coefficients, constraint_matrix, constraint_rhs,
                 initial_basis=None, minimize=True, max_iterations=100):
        """Initialize the solver with problem data."""
        # Input validation
        if not isinstance(objective_coefficients, (list, np.ndarray)):
            raise TypeError("objective_coefficients must be a list or numpy array")
        if not isinstance(constraint_matrix, (list, np.ndarray)):
            raise TypeError("constraint_matrix must be a list or numpy array")
        if not isinstance(constraint_rhs, (list, np.ndarray)):
            raise TypeError("constraint_rhs must be a list or numpy array")
        if initial_basis is not None and not isinstance(initial_basis, list):
            raise TypeError("initial_basis must be a list")
        if not isinstance(minimize, bool):
            raise TypeError("minimize must be a boolean")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise TypeError("max_iterations must be a positive integer")

        self.c = np.array(objective_coefficients, dtype=float)  # Objective coefficients
        self.A = np.array(constraint_matrix, dtype=float)       # Constraint matrix
        self.b = np.array(constraint_rhs, dtype=float)          # Right-hand side
        self.minimize = minimize
        self.max_iterations = max_iterations

        self.m, self.n = self.A.shape  # Number of constraints (m) and variables (n)
        self.original_n = self.n       # Store original number of variables
        if len(self.c) != self.n:
            raise ValueError("Length of objective_coefficients must match number of variables")
        if len(self.b) != self.m:
            raise ValueError("Length of constraint_rhs must match number of constraints")

        # Adjust for maximization
        self.c_original = self.c.copy()
        if not minimize:
            self.c = -self.c

        # Initialize basis
        self.basic_vars = initial_basis if initial_basis is not None else self._find_initial_basis()
        if initial_basis and not self._is_valid_basis(initial_basis):
            raise ValueError("Provided initial_basis is invalid or infeasible")
        self.non_basic_vars = [i for i in range(self.n) if i not in self.basic_vars]
        
        self.tableau = None
        self.iteration_count = 0
        self.is_optimal = False
        self.is_unbounded = False
        self.is_infeasible = False
        self.optimal_solution = None
        self.optimal_value = None

    def _is_valid_basis(self, basis):
        """Check if the basis is valid and non-singular."""
        if len(basis) != self.m:
            return False
        try:
            B = self.A[:, basis]
            np.linalg.inv(B)  # Check for singularity
            return True
        except np.linalg.LinAlgError:
            return False

    def _find_initial_basis(self):
        """Find an initial basis using slack variables or prepare for two-phase/dual method."""
        basis = []
        for i in range(self.m):
            for j in range(self.n):
                col = self.A[:, j]
                if (col[i] == 1 and np.all(col[:i] == 0) and 
                    np.all(col[i+1:] == 0) and j not in basis):
                    basis.append(j)
                    break
        if len(basis) != self.m or np.any(self.b < 0):
            return []  # Indicate need for two-phase or dual method
        return basis

    def _setup_two_phase(self):
        """Set up Phase I by adding auxiliary variables."""
        aux_cols = np.eye(self.m)
        self.A = np.hstack((self.A, aux_cols))
        self.c = np.concatenate((np.zeros(self.n), np.ones(self.m)))
        self.n += self.m
        self.basic_vars = list(range(self.original_n, self.n))
        self.non_basic_vars = list(range(self.original_n))
        self.tableau = self._construct_tableau()

    def _cleanup_two_phase(self):
        """Clean up after Phase I and prepare for Phase II."""
        self.A = self.A[:, :self.original_n]
        self.c = self.c_original.copy()
        self.n = self.original_n
        new_basis = [v for v in self.basic_vars if v < self.n]
        if len(new_basis) < self.m:
            # Add slack variables if needed
            for i in range(self.m):
                if i >= len(new_basis):
                    for j in range(self.n):
                        if (self.A[i, j] == 1 and np.all(self.A[:i, j] == 0) and 
                            np.all(self.A[i+1:, j] == 0) and j not in new_basis):
                            new_basis.append(j)
                            break
        self.basic_vars = new_basis
        self.non_basic_vars = [i for i in range(self.n) if i not in self.basic_vars]
        self.tableau = self._construct_tableau()

    def _setup_dual_simplex(self):
        """Set up for Dual Simplex with slack/surplus and auxiliary variables."""
        extra_cols = []
        basis = []
        for i in range(self.m):
            if self.b[i] < 0:
                surplus = np.zeros(self.m); surplus[i] = -1
                aux = np.zeros(self.m); aux[i] = 1
                extra_cols.extend([surplus, aux])
                basis.append(self.n + 2 * i + 1)  # Auxiliary variable
                self.c = np.concatenate((self.c, [0, 1]))
            else:
                slack = np.zeros(self.m); slack[i] = 1
                extra_cols.append(slack)
                basis.append(self.n + len(extra_cols) - 1)
                self.c = np.concatenate((self.c, [0]))
        if extra_cols:
            self.A = np.hstack((self.A, np.array(extra_cols).T))
            self.n = self.A.shape[1]
        self.basic_vars = basis
        self.non_basic_vars = [i for i in range(self.n) if i not in self.basic_vars]

    def _construct_tableau(self):
        """Construct the Simplex tableau."""
        tableau = np.zeros((self.m + 1, self.n + 1))
        tableau[:-1, :-1] = self.A
        tableau[:-1, -1] = self.b
        try:
            B = self.A[:, self.basic_vars]
            c_B = self.c[self.basic_vars]
            B_inv = np.linalg.inv(B)
            tableau[-1, :-1] = self.c - c_B @ B_inv @ self.A
            tableau[-1, -1] = -c_B @ B_inv @ self.b
        except np.linalg.LinAlgError:
            self.is_infeasible = True
        return tableau

    def _pivot(self, row, col):
        """Perform pivot operation with stability check."""
        pivot = self.tableau[row, col]
        if abs(pivot) < self.TOLERANCE:
            raise ValueError("Pivot element too small, possible degeneracy or instability")
        self.tableau[row] /= pivot
        for i in range(self.m + 1):
            if i != row:
                factor = self.tableau[i, col]
                self.tableau[i] -= factor * self.tableau[row]

    def _simplex_step(self):
        """Perform one iteration of the Simplex method."""
        reduced_costs = self.tableau[-1, :-1]
        if np.all(reduced_costs >= -self.TOLERANCE):
            self.is_optimal = True
            return False
        # Bland's rule: smallest index with negative reduced cost
        entering = min([j for j in range(self.n) if reduced_costs[j] < -self.TOLERANCE])
        ratios = [(self.tableau[i, -1] / self.tableau[i, entering], i) 
                  for i in range(self.m) if self.tableau[i, entering] > self.TOLERANCE]
        if not ratios:
            self.is_unbounded = True
            return False
        _, leaving = min(ratios)
        self._pivot(leaving, entering)
        self.basic_vars[leaving], self.non_basic_vars[entering] = \
            self.non_basic_vars[entering], self.basic_vars[leaving]
        return True

    def _dual_simplex_step(self):
        """Perform one iteration of the Dual Simplex method."""
        rhs = self.tableau[:-1, -1]
        if np.all(rhs >= -self.TOLERANCE):
            self.is_optimal = True
            return False
        leaving = np.argmin(rhs)
        if rhs[leaving] >= -self.TOLERANCE:
            self.is_optimal = True
            return False
        ratios = [(abs(self.tableau[-1, j] / self.tableau[leaving, j]), j) 
                  for j in range(self.n) if self.tableau[leaving, j] < -self.TOLERANCE]
        if not ratios:
            self.is_infeasible = True
            return False
        _, entering = min(ratios)
        self._pivot(leaving, entering)
        self.basic_vars[leaving], self.non_basic_vars[entering] = \
            self.non_basic_vars[entering], self.basic_vars[leaving]
        return True

    def _extract_solution(self):
        """Extract the optimal solution and value."""
        self.optimal_solution = np.zeros(self.n)
        for i, var in enumerate(self.basic_vars):
            if var < self.original_n:  # Only original variables
                self.optimal_solution[var] = self.tableau[i, -1]
        self.optimal_value = -self.tableau[-1, -1]
        if not self.minimize:
            self.optimal_value = -self.optimal_value
        self.optimal_solution = self.optimal_solution[:self.original_n]

    def solve(self):
        """Solve the linear programming problem."""
        if not self.basic_vars:  # No initial basis found
            if np.any(self.b < 0):
                self._setup_dual_simplex()
            else:
                self._setup_two_phase()
                self.tableau = self._construct_tableau()
                while self.iteration_count < self.max_iterations and not (self.is_optimal or self.is_unbounded):
                    self.iteration_count += 1
                    if not self._simplex_step():
                        break
                if self.tableau[-1, -1] > self.TOLERANCE:
                    self.is_infeasible = True
                    return (False, None, None, False, True, self.iteration_count)
                self._cleanup_two_phase()

        self.tableau = self._construct_tableau()
        if self.is_infeasible:
            return (False, None, None, False, True, self.iteration_count)

        step_func = self._dual_simplex_step if np.any(self.b < 0) else self._simplex_step
        while self.iteration_count < self.max_iterations and not (self.is_optimal or self.is_unbounded or self.is_infeasible):
            self.iteration_count += 1
            if not step_func():
                break

        if self.is_optimal:
            self._extract_solution()
        else:
            self.optimal_solution, self.optimal_value = None, None

        return (self.is_optimal, self.optimal_solution, self.optimal_value,
                self.is_unbounded, self.is_infeasible, self.iteration_count)

# Example usage
if __name__ == "__main__":
    solver = LinearProgramSolver(
        objective_coefficients=[-1, -1, 0, 0],
        constraint_matrix=[[1, 1, 1, 0], [2, 1, 0, 1]],
        constraint_rhs=[4, 6],
        minimize=False
    )
    result = solver.solve()
    print(f"Optimal: {result[0]}, Solution: {result[1]}, Value: {result[2]}, "
          f"Unbounded: {result[3]}, Infeasible: {result[4]}, Iterations: {result[5]}")