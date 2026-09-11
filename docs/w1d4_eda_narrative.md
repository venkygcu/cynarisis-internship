# W1D4 Exploratory Data Analysis

## Five observations

1. The dataset has 11 India records from 2004–2014 and 22 columns: 20 numeric and two categorical.
2. `Country` and `Status` are constant (`India` and `Developing`), so they carry no predictive variation in this slice.
3. `df.isnull().sum()` reports no missing values in this supplied extract.
4. Life expectancy rises from 63.5 to 68.0 across the observed years, while adult mortality generally falls.
5. Several indicators move together strongly, but 11 observations are too few for reliable correlation-based conclusions.

## EDA narrative (200 words)

The dataset has eleven India observations from 2004–2014, with health and socioeconomic indicators. Life expectancy increases from 63.5 years to 68.0 years, while adult mortality declines, suggesting an improving health context. Schooling and income-composition values also trend upward, and the distributions show that several measures change gradually rather than randomly. The category chart is intentionally sparse: Country is always India and Status is always Developing. Those columns have no variation in this single-country extract, so they should be excluded from modelling unless data from other countries is added.

The main concern is sample size, not missingness: `df.isnull().sum()` reports zero nulls in this supplied extract. Eleven rows are insufficient for stable estimates, and one record per year makes time a likely confounder. GDP and population have very large scales and may be skewed, so log transformation and unit validation are sensible before modelling. Before building an MLflow-tracked model or an MLOps pipeline, add more country-year observations, check source definitions and units, deduplicate records, validate plausible ranges, and use a time-aware validation split. If the upstream full dataset contains absent values, document imputation separately rather than assuming this clean extract needs it.
