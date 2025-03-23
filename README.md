# linear-program-solver

A Python implementation of the Simplex and Dual Simplex algorithms for solving linear programming problems.  This project provides a well-documented and easy-to-use solver for both minimization and maximization problems, including handling of infeasibility and unboundedness.

## Features

*   **Simplex Method:** Implements the standard Simplex algorithm.
*   **Dual Simplex Method:** Handles problems with negative right-hand side values (initial infeasibility).
*   **Two-Phase Simplex:** Automatically handles constraints that require auxiliary variables (e.g., equality constraints, greater-than-or-equal-to constraints).
*   **Minimization and Maximization:** Supports both problem types.
*   **Infeasibility and Unboundedness Detection:**  Detects and reports if a problem is infeasible or unbounded.
*   **Clear Error Handling:**  Raises informative exceptions for invalid input.
*   **Well-Documented:**  Includes comprehensive docstrings and comments.
*   **Easy-to-Use:** Provides a simple `solve()` method for solving problems.
*   **Uses NumPy:** Leverages NumPy for efficient numerical computations.

