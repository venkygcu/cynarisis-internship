# W1D4 Exploratory Data Analysis

## Five observations

1. The dataset contains 20 students, seven original columns, and three missing values, all in `ml_score`.
2. Ages are narrowly distributed from 21 to 25, so age is unlikely to explain much score variation in this sample.
3. Median imputation changes only three ML values and preserves all 20 student records for score comparisons.
4. Grades are concentrated in C (11 students) and B (7 students); there are two F grades and no A grades.
5. Attendees average 71.08, below non-attendees at 76.75, but the groups are imbalanced (16 versus 4) and too small for a reliable attendance conclusion.

## EDA narrative (200 words)

This student dataset contains twenty records with age, mathematics, Python, machine-learning, and attendance fields. The initial inspection identifies three missing values, all in `ml_score`, while every other column is complete. Replacing those values with the observed ML-score median retains the full sample and allows an average score to be calculated for each student. The resulting grade distribution is centred on C and B: eleven students receive C, seven receive B, two receive F, and nobody reaches A. The top average score is 84.7, so even the strongest observed result remains below the A threshold. Numeric distributions show a narrow age range of 21 to 25, whereas subject scores span more widely.

The correlation heatmap can suggest score relationships, but it should be read carefully because the sample has only twenty students and the average score is derived from three of the plotted score columns. Attendees average 71.08 versus 76.75 for non-attendees, but groups contain sixteen and four students, so no attendance effect is established. Before modelling, confirm why ML scores were missing, retain an imputation flag, exclude identifier and name fields, collect a larger balanced sample, and evaluate results with held-out data.
