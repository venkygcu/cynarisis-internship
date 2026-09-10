"""NumPy fundamentals for W1D1 ML assignment.

This script demonstrates:
- variables and Python data types
- NumPy 1D, 2D, and 3D arrays
- broadcasting and vectorized operations
- matrix multiplication via @
- mean/std/correlation over a CSV dataset
"""

from __future__ import annotations

import numpy as np


def show_basic_python_types() -> None:
    """Demonstrate variables and primitive Python types."""
    student_name = "Venky"
    age = 21
    gpa = 8.9
    is_active = True

    print("Python basics:")
    print(f"Name: {student_name} | Type: {type(student_name).__name__}")
    print(f"Age: {age} | Type: {type(age).__name__}")
    print(f"GPA: {gpa} | Type: {type(gpa).__name__}")
    print(f"Active: {is_active} | Type: {type(is_active).__name__}")


def create_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 1D, 2D, and 3D arrays and print their shapes."""
    vector = np.array([10, 20, 30, 40], dtype=np.int32)
    matrix = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.float64,
    )
    tensor = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.float64,
    )

    print("\nArray creation:")
    print(f"1D vector shape: {vector.shape} -> {vector}")
    print(f"2D matrix shape: {matrix.shape} -> {matrix}")
    print(f"3D tensor shape: {tensor.shape} -> {tensor}")
    return vector, matrix, tensor


def broadcasting_demo() -> None:
    """Show NumPy broadcasting with a 1D vector and scalar."""
    base = np.array([10, 20, 30], dtype=np.float64)
    scale = 1.5
    boosted = base * scale

    print("\nBroadcasting demo:")
    print(f"Base array: {base}")
    print(f"Scale factor: {scale}")
    print(f"Broadcasted result: {boosted}")


def vectorized_operations() -> None:
    """Compute vectorized stats without Python loops."""
    marks = np.array([55, 67, 89, 91, 72], dtype=np.float64)
    weights = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float64)

    weighted_scores = marks * weights
    normalized = weighted_scores / np.sum(weights)

    print("\nVectorized operations:")
    print(f"Marks: {marks}")
    print(f"Weights: {weights}")
    print(f"Weighted scores: {weighted_scores}")
    print(f"Normalized scores: {normalized}")
    print(f"Mean weighted score: {np.mean(weighted_scores):.3f}")
    print(f"Std dev: {np.std(weighted_scores):.3f}")


def matrix_multiplication_demo() -> None:
    """Demonstrate matrix multiplication using NumPy."""
    matrix_a = np.array([[1, 2], [3, 4]], dtype=np.float64)
    matrix_b = np.array([[5, 6], [7, 8]], dtype=np.float64)
    result = matrix_a @ matrix_b

    print("\nMatrix multiplication:")
    print(f"A =\n{matrix_a}")
    print(f"B =\n{matrix_b}")
    print(f"A @ B =\n{result}")


def dataset_stats() -> None:
    """Load a real CSV dataset and compute mean, std, and correlation."""
    data = np.genfromtxt(
        "data/marketing_metrics.csv",
        delimiter=',',
        names=True,
        dtype=None,
        encoding='utf-8',
    )

    spend = data['ad_spend']
    leads = data['leads']
    roi = data['roi']

    print("\nDataset statistics from CSV:")
    print(f"Mean ad spend: {np.mean(spend):.2f}")
    print(f"Std dev ad spend: {np.std(spend):.2f}")
    print(f"Mean ROI: {np.mean(roi):.2f}")
    print(f"Correlation matrix:\n{np.corrcoef(spend, leads)}")
    print(f"Lead-to-ROI correlation: {np.corrcoef(leads, roi)[0, 1]:.3f}")


def main() -> None:
    show_basic_python_types()
    create_arrays()
    broadcasting_demo()
    vectorized_operations()
    matrix_multiplication_demo()
    dataset_stats()


if __name__ == "__main__":
    main()
