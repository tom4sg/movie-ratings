# movie-ratings

This repository contains data analysis tasks completed as part of the *Principles of Data Science II* (PODS) course at NYU, using a large-scale movie ratings dataset. The dataset includes ratings from over 1000 participants across 400 movies, with additional demographic and psychological data. Each analysis task builds foundational skills in statistics, regression, and hypothesis testing, using real-world behavioral data.

---

## 📁 Dataset

The dataset used is `movieDataReplicationSet.csv`, which contains:
- Ratings on a 0–4 scale (with `NaN` for unrated movies)
- Participant-level metadata (e.g., income, education, SES)
- 400 movies including titles like *Pulp Fiction*, *Kill Bill*, *Star Wars*, *Interstellar*, and more

> **Note**: The dataset is not included in this repo due to licensing and privacy restrictions.

---

## 🧠 Tasks Breakdown

### D1: Central Tendency
- Compute **mean**, **median**, and **mode** ratings for each movie
- Determine overall average movie rating
- Identify highest/lowest rated films using different metrics

### D2: Dispersion & Correlation
- Calculate **standard deviation** and **mean absolute deviation** for each movie
- Compute pairwise **Pearson correlations** between all movies
- Report summary statistics of dispersion and correlation

### D3: Simple Linear Regression
- Build simple regression models:
  - Predict *Star Wars I* from *Star Wars II*
  - Predict *Titanic* from *Star Wars I*
- Report slope, intercept, residuals, and \( R^2 \)

### D4: Statistical Control
- Correlate **education** and **income**
- Compute **partial correlation** controlling for **SES**
- Build a **multiple regression** predicting income from education and SES

### D6: Parametric Significance Testing
- Conduct **independent** and **paired** t-tests between:
  - *Kill Bill Vol. I*, *Kill Bill Vol. II*, and *Pulp Fiction*
- Report \( t \)-values, \( p \)-values, and degrees of freedom

### D7: Nonparametric Significance Testing
- Use **Mann-Whitney U** and **Kolmogorov-Smirnov** tests to:
  - Compare medians and distributions of:
    - *Indiana Jones* trilogy
    - *Ghostbusters* remake
    - *Finding Nemo*
    - *Interstellar*
    - *Wolf of Wall Street*

---

## 📊 Tools & Libraries

- [`pandas`](https://pandas.pydata.org/) for data manipulation  
- [`numpy`](https://numpy.org/) for numerical analysis  
- [`scikit-learn`](https://scikit-learn.org/) for regression models  
- [`scipy.stats`](https://docs.scipy.org/doc/scipy/) for statistical tests  
- [`matplotlib`](https://matplotlib.org/) (optional) for visualizations

---

## 📌 Highlights

- Clean, modular analysis of real-world, messy data with missing values
- Demonstrates key statistical concepts: central tendency, dispersion, correlation, regression, and significance testing
- Organized to align with weekly PODS assignments for reproducibility

---

## 🧑‍💻 Author

**Tomas Gutierrez**  
NYU Double Major — Data Science & Recorded Music  
[GitHub: @tomasgutierrez](https://github.com/tomasgutierrez)
