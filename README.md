# W1D1 to W1D3: Python for ML Exercises

This repository contains the first three weekly exercises for the AI/ML internship track. It covers NumPy fundamentals, Pandas data wrangling, and CSV loading, cleaning, and inspection using real data.

## Project setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src/w1d1.py
```

## W1D1 Included Work

- Python variables and core data types
- NumPy 1D, 2D, and 3D arrays with shape inspection
- Broadcasting and vectorized operations without Python loops
- Matrix multiplication using `@`
- CSV-based descriptive stats with mean, standard deviation, and correlation
- Notebook-ready examples in `notebooks/w1d1.ipynb`

## W1D2 Included Work

- Real CSV loading with a cached India slice from the public Life Expectancy dataset
- Pandas inspection of shape, dtypes, missing values, and duplicates
- Column normalization, numeric coercion, and duplicate removal
- Cleaned CSV and Parquet exports in `data/w1d2_cleaned_india_life_expectancy.csv` and `data/w1d2_cleaned_india_life_expectancy.parquet`
- Notebook-ready workflow in `notebooks/w1d2.ipynb`

## W1D3 Included Work

- Reusable data loading and cleaning workflow in `notebooks/w1d3.ipynb`
- Raw data inspection and validation checks for the India dataset slice
- Cleaned CSV and Parquet exports in `data/w1d3_cleaned_india_life_expectancy.csv` and `data/w1d3_cleaned_india_life_expectancy.parquet`
- Notebook execution that stays runnable with an offline fallback dataset

## Git branch

- `feat/aiml-W1-venky`

## Commit history

- `feat: initialize ml workspace`
- `feat: numpy fundamentals — array ops and statistics`
- `feat: data loading, cleaning, and inspection`
- `feat: add w1d3 data cleaning notebook`

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

The repository now includes validated notebook workflows for day 1 through day 3, with runtime output for the notebook-based data loading and cleaning tasks plus exported CSV and Parquet artifacts.
