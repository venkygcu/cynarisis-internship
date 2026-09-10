# W1D1: Python for ML — NumPy Fundamentals

This repository contains a Python and NumPy exercise for the AI/ML internship track. It demonstrates array creation, broadcasting, vectorized math, matrix multiplication, and dataset statistics using a real CSV file.

## Project setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/w1d1.py
```

## Included work

- Python variables and core data types
- NumPy 1D, 2D, and 3D arrays with shape inspection
- Broadcasting and vectorized operations without Python loops
- Matrix multiplication using `@`
- CSV-based descriptive stats with mean, standard deviation, and correlation
- Notebook-ready examples in `notebooks/w1d1.ipynb`

## Git branch

- `feat/aiml-W1-venky`

## Commit history

- `feat: initialize ml workspace`
- `feat: numpy fundamentals — array ops and statistics`

## Viva Q&A

### 1. What is broadcasting in NumPy?
Broadcasting is NumPy's automatic behavior for aligning arrays of different shapes during arithmetic. Example: a vector `[10, 20, 30]` multiplied by a scalar `1.5` yields `[15, 30, 45]` without manually expanding the array.

### 2. Why is vectorized NumPy faster than Python loops?
NumPy runs operations in optimized C code, reducing Python interpreter overhead and improving memory throughput for large numeric arrays.

### 3. When would you use a NumPy array vs a Pandas Series?
Use a NumPy array for pure numeric array math and linear algebra. Use a Pandas Series when you need labeled indexing, alignment, or table-based data workflows.

## Self-review checklist

- [x] Python environment created and working
- [x] Jupyter and ML libraries installed
- [x] 1D, 2D, and 3D arrays created and shapes verified
- [x] Broadcasting and vectorized operations tested
- [x] Matrix multiplication executed without loops
- [x] Real CSV dataset processed for mean/std/correlation
- [x] Output captured from script execution
- [x] Git branch prepared for submission

## Evidence

The script prints the actual runtime output for array shapes, broadcasting, weighted operations, matrix multiplication, and CSV statistics from the included `data/marketing_metrics.csv` dataset.
