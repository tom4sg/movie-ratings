# Movie Ratings Analysis – PODS

This repo contains data analysis tasks completed for *Principles of Data Science II* at NYU. Using movie ratings from 1,097 participants across 400 films, the tasks apply statistical analysis, regression, and hypothesis testing to real behavioral data.

## Dataset

- File: `movieDataReplicationSet.csv`
- 400 movies rated 0–4 (missing values = unrated)
- Includes participant metadata: income, education, SES

*Note: Dataset not included due to privacy restrictions.*

## Tasks

### D1: Central Tendency
- Mean, median, and mode for each movie
- Overall average rating
- Identify highest/lowest-rated movies

### D2: Dispersion & Correlation
- Standard deviation and mean absolute deviation
- Pearson correlation matrix (400x400)
- Mean/median of dispersion and correlation

### D3: Simple Regression
- Regress *Star Wars I* on *Star Wars II*
- Regress *Titanic* on *Star Wars I*
- Coefficients, residuals, R²

### D4: Statistical Control
- Correlation: education vs. income
- Partial correlation (control for SES)
- Multiple regression: income ~ SES + education

### D6: Parametric Tests
- Independent & paired t-tests between:
  - *Kill Bill Vol. I*, *Vol. II*, *Pulp Fiction*

### D7: Nonparametric Tests
- Mann-Whitney U and KS tests on:
  - *Indiana Jones*, *Ghostbusters (2016)*, *Finding Nemo*, *Interstellar*, *Wolf of Wall Street*

## Tools

- `pandas`, `numpy`, `scikit-learn`, `scipy.stats`

## Author

Tomas Gutierrez  
NYU | Data Science & Recorded Music  
[github.com/tomasgutierrez](https://github.com/tomasgutierrez)
