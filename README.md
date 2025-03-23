# linear-program-solver

A Python implementation of the Simplex and Dual Simplex algorithms for solving linear programming problems.  This project provides a well-documented and easy-to-use solver for both minimization and maximization problems, including handling of infeasibility and unboundedness.

## Features

*   **Simplex and Dual Simplex:** Implements both algorithms for comprehensive LP solving.
*   **Two-Phase Simplex:** Automatically handles constraints requiring auxiliary variables.
*   **Minimization and Maximization:** Supports both objective function types.
*   **Infeasibility and Unboundedness Detection:**  Identifies and reports non-optimal solutions.
*   **Numerical Stability:** Uses a tolerance for floating-point comparisons and checks for small pivot elements.
*   **Bland's Rule:** Implements Bland's rule to prevent cycling in the Simplex method.
*   **Optional Initial Basis:** Allows providing a known initial basis for faster solving.
*   **Clear Error Handling:**  Raises informative exceptions for invalid inputs.
*   **Well-Documented:**  Includes comprehensive docstrings.
*   **NumPy-Based:** Uses NumPy for efficient numerical operations.

