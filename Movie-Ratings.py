#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:49:36 2025

@author: tomasgutierrez
"""

#%%
"""
D1: Central tendency

1) Loads the dataset in ‘movieDataReplicationSet.csv’. This dataset contains the movie rating data of 400 movies from 1097 research 
participants on a scale from 0 to 4. Missing values in this dataset represent movies that the participant declined to rate, for whatever 
reason (usually because they had not seen it). 
2) Computes the mean rating of each movie, across all participants (the output here should be a variable that contains 400 means)
3) Computes the median rating of each movie, across all participants (the output here should be a variable that contains 400medians)
4) Computes the modal rating of each movie, across all participants (the output here should be a variable that contains 400 modes)
5) Computes the mean of all the mean ratings from 2) to calculate a “mean of means
”Hint: Whereas you could type all of this out by hand, your life will be much easier if you use loops and/or vector operations. 
You can use the pre-existing mean, median and mode functions in Python. You don’t have to write your own (but you can, if you want to).
Note: Because there is missing data, you’ll need to find a way to handle those. 
There is a code session (CS) on that, which you could check out. It will also be covered in the lab.
"""

import numpy as np
import pandas as pd

# Read csv
df = pd.read_csv("/Users/tomasgutierrez/Desktop/NYU/PODS/Data Analysis Tasks PODS/D1/movieDataReplicationSet.csv")

# Compute Mean rating of each movie (the output here should be a variable that contains 400 means)
movie_means = df[df.columns[:400]].mean()
print(movie_means)
print(movie_means.shape)

# Compute Median rating of each movie (the output here should be a variable that contains 400 medians)
movie_medians = df[df.columns[:400]].median()
print(movie_medians)
print(movie_medians.shape)

# Compute Mode rating of each movie (the output here should be a variable that contains 400 modes)
# noticed that there can be several values that are the mode(both occur the same amount of times), 
# and that NaN can also be considered a mode! So, we can either .dropna() or specify dropna=True within .mode()
movie_modes = df[df.columns[:400]].mode(dropna=True).iloc[0]
print(movie_modes)
print(movie_modes.shape)

# Compute Mean of all Mean Ratings (Mean of Means)
mean_of_means = movie_means.mean()
print(mean_of_means)

# QUIZ PORTION

# Question 1 (1 point) Using the median, the highest rated movie in this dataset is...
# noticed that Dataframe and a Series are different. Dataframe is 2D, Series is 1D. Series
# works alot like a Dictionary, since it has .keys/.keys() and .values. If you want the title associated
# with each value, which is in this case the Movie name, then use .keys(). These are lists, so you can index
# through them as such. 

median_values = movie_medians.values
max_index = 0
for median in median_values:
    if median == movie_medians.max():
        break
    else:
        max_index+=1

print(movie_medians.keys()[max_index])

# OR

print(np.argmax(movie_medians.values))
print(movie_medians.keys()[np.argmax(movie_medians.values)])

# Question 2 (1 point) Using the mean, the lowest rated movie in this dataset is...
# noticed that np.argmin(nparray) returns the index of the minimum value in the numpy array. 
# np.argmax(nparray) returns the index of the maximum value in the numpy array. 
# Just use brackets to access an index in a numpy array

print(np.argmin(movie_means.values))
print(movie_means.keys()[np.argmin(movie_means.values)])

# Question 3 (1 point) Using the mean, the highest rated movie in this dataset is...

print(movie_means.keys()[np.argmax(movie_means.values)])

# Question 4 (1 point) The mean of means is...

print(mean_of_means)

# Question 5 (1 point) Using the median, the lowest rated movie of these choices is

specified_movies = ["Downfall (2004)", "Harry Potter and the Chamber of Secrets (2002)", "Black Swan (2010)", "Halloween (1978)", 
"Battlefield Earth (2000)"]

for movie in specified_movies:
    print(movie, movie_medians[movie])
    
# Question 6 (1 point) The modal rating of the movie Independence Day (1996) is...

print(movie_modes["Independence Day (1996)"])

# Let me just check that I didn't mess up with the .iloc[0] before

mode_test = df[df.columns[:400]].mode(dropna=True)
print(mode_test["Independence Day (1996)"])
#%%
"""
D2: Dispersion and Correlation

1)Builds on D1. So the data from D1 should already be loaded. We’ll reuse it here. 
2)Computes the standard deviation of the ratings of each movie, across all participants (the output here should be a variable that contains 400 standard deviations)
3)Computes the mean absolute deviation of the ratings of each movie, across all participants (the output here should be a variable that contains 400 mean absolute deviations)
4)Computes the mean and median of the standard deviations resulting from 2)
5)Computes the mean and median of the mean absolute deviations resulting from 3)
6)Computes the pairwise Pearson correlation between all ratings of movies. (the output here should be 400x400 correlation matrix –it’s ok to correlate ratings of a given movie with itself, for the sake of simplicity. It will also allow you to check if this worked, as the diagonal should contain 1s). 
7)Computes the mean and median correlation resulting from 6)
"""
movies = df[df.columns[:400]]

# Computes STD of each movie (the output here should be a variable that contains 400 standard deviations)
movie_std = df[df.columns[:400]].std()
print(movie_std)
print(movie_std.shape)

# Computes the mean absolute deviation of each movie (the output here should be a variable that contains 400 mean absolute deviations)
movie_mad = (movies - movies.mean()).abs().mean()
print(movie_mad)
print(movie_mad.shape)

# Computes the mean and median of the standard deviations resulting from 2)
movie_std_mean = movie_std.mean()
print(movie_std_mean)

movie_std_median = movie_std.median()
print(movie_std_median)

# Computes the mean and median of the mean absolute deviations resulting from 3)
movie_mad_mean = movie_mad.mean()
print(movie_mad_mean)

movie_mad_median = movie_mad.median()
print(movie_mad_median)

# Computes the pairwise Pearson correlation between all ratings of movies. (the output here should be 400x400 correlation matrix –it’s ok to correlate ratings of a given movie with itself, for the sake of simplicity. It will also allow you to check if this worked, as the diagonal should contain 1s). 
movies_correlation_matrix = movies.corr()
print(movies_correlation_matrix)
print(movies_correlation_matrix.shape)

# Computes the mean and median correlation resulting from 6)
mean_corr = movies_correlation_matrix.mean()
print(mean_corr)
median_corr = movies_correlation_matrix.median()
print(median_corr)

# QUIZ PORTION

# Question 1 (1 point) What is the mean of the mean absolute deviations of all movies?
print(movie_mad_mean)
# 0.8519316014154186

# Question 2 (1 point) What is the median of the mean absolute deviations of all movies?
print(movie_mad_median)
# 0.8576626878081213

# Question 3 (1 point) What is the mean of the standard deviations of all movies?
print(movie_std_mean)
# 1.0551036136210792

# Question 4 (1 point) What is the average correlation between two movies in this dataset (including self-correlations, as per prompt)?
average_corr = movies_correlation_matrix.values.mean()
print(average_corr)
# 0.39028621339553043

# Question 5 (1 point) What is the median of the standard deviations of all movies?
print(movie_std_median)
# 1.069293158708418
#%%
"""
D3: Simple linear regression
1)Builds on D1 and D2. So the data from D1 should already be loaded. We’ll reuse it here. 
2)Finds the ratings of users who have rated both Star Wars I and Star Wars II. We will only consider these ratings (where the pair of ratings from both movies is jointly present) going forward.
3)Builds a simple regression model –predicting the ratings of Star Wars I from the ratings of Star Wars II (only). We will keep this as simple as possible (not using multiple regression or avoid overfitting –we’ll consider that in a later assignment, do not forget to include the intercept term, however)
4)Returns the betasand residuals of this model. 
5)Finds the ratings of users who have rated both Star Wars I and Titanic. Make sure you are not off by one (!)
6)Builds a simple regression model –predicting the ratings of Titanic from the ratings of Star Wars I (only).
7)Returns the betas and residuals of this model.
"""
from sklearn.linear_model import LinearRegression

print(movies.filter(regex='Star Wars.*').columns)
print(movies.filter(regex='Titanic.*').columns)

titanic = 'Titanic (1997)'
star_wars_1 = 'Star Wars: Episode 1 - The Phantom Menace (1999)'
star_wars_2 = 'Star Wars: Episode II - Attack of the Clones (2002)'
# Finds the ratings of users who have rated both Star Wars I and Star Wars II. We will only consider these ratings (where the pair of ratings from both movies is jointly present) going forward.

star_wars_df = movies.dropna(subset=[star_wars_1, star_wars_2])
star_wars_df = star_wars_df[[star_wars_1, star_wars_2]]
print(star_wars_df.shape)
# Builds a simple regression model –predicting the ratings of Star Wars I from the ratings of Star Wars II (only). We will keep this as simple as possible (not using multiple regression or avoid overfitting –we’ll consider that in a later assignment, do not forget to include the intercept term, however)

X = star_wars_df[[star_wars_2]]
y = star_wars_df[star_wars_1]

model = LinearRegression()
model.fit(X, y)

# Returns the betas and residuals of this model. 

# Betas
intercept = model.intercept_
slope = model.coef_[0]
print(slope)

predictions = model.predict(X)
residuals = y - predictions

rSq = model.score(X, y)


# Finds the ratings of users who have rated both Star Wars I and Titanic. Make sure you are not off by one (!)

wars_titanic_df = movies.dropna(subset=[star_wars_1, titanic])
wars_titanic_df = wars_titanic_df[[star_wars_1, titanic]]
print(wars_titanic_df.shape)

# Builds a simple regression model –predicting the ratings of Titanic from the ratings of Star Wars I (only).

X2 = wars_titanic_df[[star_wars_1]]
y2 = wars_titanic_df[titanic]

model2 = LinearRegression()
model2.fit(X2, y2)

# Returns the betas and residuals of this model.

intercept2 = model2.intercept_
slope2 = model2.coef_[0]
print(slope2)

predictions2 = model2.predict(X2)
residuals2 = y2 - predictions2

rSq2 = model2.score(X2, y2)

# How many users rated Star Wars I and II and Titanic

all_three_rated_df = movies.dropna(subset=[star_wars_1, star_wars_2, titanic])
all_three_rated_df = all_three_rated_df[[star_wars_1, star_wars_2, titanic]]
print(all_three_rated_df.shape)

#%%
"""
D4: Statistical control
1)Write code that builds on D1, D2 and D3. So the data should already be loaded. We’ll reuse it here. 
2)A section of your code should correlate education and income.
3)A section of your code should compute the partial correlation between education and income, controlling for SES.
4)A section of your code should build a multiple regression model that predicts income from both education and SES. 
"""

data = np.genfromtxt('movieDataReplicationSet.csv', delimiter=',')
data = data[1:, :]

#%%
# A section of your code should correlate education and income.

income = data[:, 481]
education = data[:, 479]
SES = data[:, 478]


corrmatrix = np.corrcoef(education, income)
# 0.44986554

#%%
# A section of your code should compute the partial correlation between education and income, controlling for SES.

SES = SES.reshape(-1, 1)
education = education.reshape(-1, 1)
income = income.reshape(-1, 1)

# Algorithm: 
# 1) Simple linear regression predicting education from SES

educationModel = LinearRegression().fit(SES, education)
yHatEducation = educationModel.predict(SES)
educationResiduals = education - yHatEducation

# 2) Simple linear regression predicting income from SES

incomeModel = LinearRegression().fit(SES, income)
yHatIncome = incomeModel.predict(SES)
incomeResiduals = income - yHatIncome

# 3) Correlated the residuals of these two regressions - This is your answer! 

partialCorr = np.corrcoef(incomeResiduals.flatten(), educationResiduals.flatten())
# 0.339187

#%%
# A section of your code should build a multiple regression model that predicts income from both education and SES. 

multiX = data[:, [478, 479]]
# SES is var1, education is var2

incomeMultiModel = LinearRegression().fit(multiX, income)
yHatIncomeMult = incomeMultiModel.predict(multiX)

incomeRsq = incomeMultiModel.score(multiX, income)
# 0.3345762362918535

sesCoef = incomeMultiModel.coef_.flatten()[0]
educationCoef = incomeMultiModel.coef_.flatten()[1]

# R calculation - correlation between yHat and y

R = np.corrcoef(yHatIncomeMult.flatten(), income.flatten())
# 0.57842565

# RMSE

incomeMultiSquareErrors = (yHatIncomeMult - income).flatten()**2
incomeMultiMeanSquaredError = incomeMultiSquareErrors.mean()
incomeMultiRMSE = np.sqrt(incomeMultiMeanSquaredError)
# 31.190367098510183

#%%
"""
D6: Parametric significance testing
1) Builds on the data we have been using since D1. So the same data should already be loaded. We’ll reuse it here.
2) Finds users who have rated “Kill Bill Vol. I (2003)”, “Kill Bill Vol. II (2004)” and “Pulp Fiction (1994)”. We will only consider ratings from users who have seen all 3 movies for the rest of this assignment. Make sure you have no off-by-one errors, or everything else in this assignment will be wrong.
3) Finds the mean and median for all 3 of these movies (using only ratings from users identified in 2) – who have seen and report ratings for all 3 movies, NOT from everyone)
4) Does an independent-samples t-test between all 3 movies (all possibilities), using *only* ratings from users identified in 2).
5) Does a paired-samples t-test between all 3 movies (all possibilities), using using *only* ratings from users identified in 2)
"""

from scipy import stats

# Builds on the data we have been using since D1. So the same data should already be loaded. We’ll reuse it here.
# Find the corresponding column names!
df.filter(like="Kill Bill")
df.filter(like="Pulp")

killbill_2 = "Kill Bill: Vol. 2 (2004)"
killbill_1 = "Kill Bill: Vol. 1 (2003)"
pulp_fiction = "Pulp Fiction (1994)"

# Remove rows of movies that have NaN ratings within these three columns
kill_pulp_df = movies.dropna(subset=[killbill_1, killbill_2, pulp_fiction])

# Remove all columns other than these three specified ones
kill_pulp_df = kill_pulp_df[[killbill_1, killbill_2, pulp_fiction]]

print(kill_pulp_df.shape)
# (238, 3)

#Finds the mean and median for all 3 of these movies (using only ratings from users identified in 2) – who have seen and report ratings for all 3 movies, NOT from everyone
kill_pulp_means = np.mean(kill_pulp_df, axis=0)
# Kill Bill: Vol. 1 (2003) - 3.262605
# Kill Bill: Vol. 2 (2004) - 3.201681
# Pulp Fiction (1994) - 3.275210


kill_pulp_medians = np.median(kill_pulp_df, axis=0)
# Kill Bill: Vol. 1 (2003) - 3.5
# Kill Bill: Vol. 2 (2004) - 3.5
# Pulp Fiction (1994) - 3.5

#Does an independent-samples t-test between all 3 movies (all possibilities), using *only* ratings from users identified in 2

# 1 and 2 (stats.ttest_ind) - Kill Bill 1 vs. Kill Bill 2
t1,p1 = stats.ttest_ind(kill_pulp_df.iloc[:,0], kill_pulp_df.iloc[:,1])
# 0.6712552299230066, 0.502384929498441

# 2 and 3 (stats.ttest_ind) - Kill Bill 2 vs. Pulp Fiction
t2,p2 = stats.ttest_ind(kill_pulp_df.iloc[:,1], kill_pulp_df.iloc[:,2])
# 0.8320119376579401, 0.4058211698544758

# 1 and 3 (stats.ttest_ind) - Kill Bill 1 vs. Pulp Fiction
t3,p3 = stats.ttest_ind(kill_pulp_df.iloc[:,0], kill_pulp_df.iloc[:,2])
# 0.1452904065829127, 0.884543349256629


# Does a paired-samples t-test between all 3 movies (all possibilities), using using *only* ratings from users identified in 2)

# 1 and 2 (stats.ttest_rel) - Kill Bill 1 vs. Kill Bill 2
t1,p1 = stats.ttest_rel(kill_pulp_df.iloc[:,0], kill_pulp_df.iloc[:,1])
# 1.4739760809411917, 0.14181503833524026

# 2 and 3 (stats.ttest_rel) - Kill Bill 2 vs. Pulp Fiction
t2,p2 = stats.ttest_rel(kill_pulp_df.iloc[:,1], kill_pulp_df.iloc[:,2])
# 1.2364948550927515, 0.21749863403452613

# 1 and 3 (stats.ttest_rel) - Kill Bill 1 vs. Pulp Fiction
t3,p3 = stats.ttest_rel(kill_pulp_df.iloc[:,0], kill_pulp_df.iloc[:,2])
# 0.23065120863955996, 0.8177847600417625



# Question 9 (1 point) What are the degrees of freedom for an *independent samples* t-test between Kill Bill 1 and Kill Bill 2 (using the same data of people who saw all 3 movies)?
# n = 238, so degrees of freedom = 238 + 238 - 2
# 474

# Question 10 (1 point) What are the degrees of freedom for a *paired samples* t-test between Kill Bill 1 and Kill Bill 2 (using the same population of people who have rated all 3 movies)?
# n = 238, so degrees of freedom = 238 - 1
# 237

#%%
"""
D7: Nonparametric significance testing
1) Builds on the data we have been using since D1 (and in particular the framework from
parametric significance testing in D6). So the same data should already be loaded. We’ll
reuse it here.
2) Uses nonparametric significance tests throughout this assignment. You can assume an
alpha level of 0.05 throughout.
3) Identify the ratings from users who have seen the movies Indiana Jones and the Raiders
of the lost Ark (1981), Indiana Jones and the last Crusade (1989), Indiana Jones and the
Kingdom of the Crystal Skull (2008), Ghostbusters (2016), Wolf of Wall Street (2013),
Interstellar (2014), and Finding Nemo (2003). Make sure you are not off by one due to
Python indexing. When comparing the ratings of pairs of movies, you will sometimes
want to use all ratings, and sometimes only the ratings of users that have jointly seen
both movies. We recommend that you do both, and note when it matters. Check code
sessions if this point or the previous one are unclear.
4) Tests whether the median ratings of Indiana Jones and the Raiders of the lost Ark (1981)
and Indiana Jones and the last Crusade (1989) are different.
5) Tests whether the median ratings of Indiana Jones and the last Crusade (1989) and
Indiana Jones and the Kingdom of the Crystal Skull (2008) are different.
6) Tests whether the median ratings of Indiana Jones and the Kingdom of the Crystal Skull
(2008) and the Ghostbusters remake from 2016 are different.
7) Tests whether the ratings distribution of the Ghostbusters remake from 2016 and
Finding Nemo (2003) are different.
8) Tests whether the ratings distribution of Finding Nemo (2003) and Interstellar (2014) are
different.
9) Tests whether the ratings distribution of Interstellar (2014) and Wolf of Wall Street
(2013) are different.
10) Tests whether the median ratings of Interstellar (2014) and Wolf of Wall Street (2013)
are different
"""

# Use u1,p1 = stats.mannwhitneyu(combinedData[0],combinedData[1])
# For nonparametric tests with 2 groups

# Use h,pK = stats.kruskal(combinedData[0],combinedData[1],combinedData[2])
# For nonparametric tests with more than 2 groups

# 3) Identify the ratings from users who have seen the movies: 
# Indiana Jones and the Raiders of the lost Ark (1981)
# Indiana Jones and the last Crusade (1989)
# Indiana Jones and the Kingdom of the Crystal Skull (2008)
# Ghostbusters (2016)
# Wolf of Wall Street (2013)
# Interstellar (2014)
# Finding Nemo (2003)

print(movies.filter(regex='Indiana Jones.*').columns)
# ['Indiana Jones and the Last Crusade (1989)','Indiana Jones and the Temple of Doom (1984)','Indiana Jones and the Raiders of the Lost Ark (1981)','Indiana Jones and the Kingdom of the Crystal Skull (2008)']

print(movies.filter(regex='Ghostbusters.*').columns)
# ['Ghostbusters (2016)']

print(movies.filter(regex='Wolf of Wall.*').columns)
# ['The Wolf of Wall Street (2013)']

print(movies.filter(regex='Interstellar.*').columns)
# ['Interstellar (2014)']

print(movies.filter(regex='Finding Nemo.*').columns)
# ['Finding Nemo (2003)']


d7_movies = movies.dropna(subset=['Indiana Jones and the Raiders of the Lost Ark (1981)', 'Indiana Jones and the Last Crusade (1989)', 'Indiana Jones and the Kingdom of the Crystal Skull (2008)', 'Ghostbusters (2016)', 'The Wolf of Wall Street (2013)', 'Interstellar (2014)', 'Finding Nemo (2003)'])
d7_movies = d7_movies[['Indiana Jones and the Raiders of the Lost Ark (1981)', 'Indiana Jones and the Last Crusade (1989)', 'Indiana Jones and the Kingdom of the Crystal Skull (2008)', 'Ghostbusters (2016)', 'The Wolf of Wall Street (2013)', 'Interstellar (2014)', 'Finding Nemo (2003)']]

print(d7_movies.shape)

# NOTICE: The Mann-Whitney U test is the nonparametric test for checking if two samples come from populations with the same median

# BUT REMEMBER: But even if two samples have the same median, they can have a different distribution…

# 4) Tests whether the median ratings of Indiana Jones and the Raiders of the lost Ark (1981) and Indiana Jones and the last Crusade (1989) are different

u1,p1 = stats.mannwhitneyu(d7_movies['Indiana Jones and the Raiders of the Lost Ark (1981)'], d7_movies['Indiana Jones and the Last Crusade (1989)'])
# 5440.5, 0.7476114397374496 (USERS that saw ALL movies)
# 45629.5, 0.03406448669097522 (USERS that saw these two movies)
# 90932.0, 0.0006030788977777768 (USERS that saw either movie)

# 5) Tests whether the median ratings of Indiana Jones and the last Crusade (1989) and Indiana Jones and the Kingdom of the Crystal Skull (2008) are different.

u2,p2 = stats.mannwhitneyu(d7_movies['Indiana Jones and the Kingdom of the Crystal Skull (2008)'], d7_movies['Indiana Jones and the Last Crusade (1989)'])
# 4279.5, 0.015004599824370668 (USERS that saw ALL movies)
# 52622.0, 1.0450496229222022e-06 (USERS that saw these two movies)
# 91121.0, 0.00017791395232999937 (USERS that saw either movie)

# 6) Tests whether the median ratings of Indiana Jones and the Kingdom of the Crystal Skull (2008) and the Ghostbusters remake from 2016 are different.

u3,p3 = stats.mannwhitneyu(d7_movies['Indiana Jones and the Kingdom of the Crystal Skull (2008)'], d7_movies['Ghostbusters (2016)'])
# 6015.5, 0.0930999677994504 (USERS that saw ALL movies)
# 30803.5, 0.40317586212138756 (USERS that saw these two movies)
# 89340.5, 0.5698649555102142 (USERS that saw either movie)


# NOTICE: Kolmogorov-Smirnov test is for testing whether the underlying distributions are the same (whatever they might be)

# 7) Tests whether the ratings distribution of the Ghostbusters remake from 2016 and Finding Nemo (2003) are different
ks,p1 = stats.ks_2samp(d7_movies['Ghostbusters (2016)'], d7_movies['Finding Nemo (2003)'])
# 0.4854368932038835, 2.286784325905413e-11 (USERS that saw ALL movies)
# 0.38860103626943004, 2.1998080152432185e-26 (USERS that saw these two movies)
# 0.3917189423262298, 8.298125201158955e-40 (USERS that saw either movie)

# 8) Tests whether the ratings distribution of Finding Nemo (2003) and Interstellar (2014) are different.
ks,p2 = stats.ks_2samp(d7_movies['Interstellar (2014)'], d7_movies['Finding Nemo (2003)'])
# 0.18446601941747573, 0.059927028813947375 (USERS that saw ALL movies)
# 0.13612565445026178, 4.7806712910118115e-05 (USERS that saw these two movies)
# 0.12415347693862283, 1.4121375263832159e-05 (USERS that saw either movie)

# 9) Tests whether the ratings distribution of Interstellar (2014) and Wolf of Wall Street (2013) are different
ks,p3 = stats.ks_2samp(d7_movies['Interstellar (2014)'], d7_movies['The Wolf of Wall Street (2013)'])
# 0.08737864077669903, 0.8291131528805725 (USERS that saw ALL movies)
# 0.04895104895104895, 0.6835192953486631 (USERS that saw these two movies)
# 0.06389662311701293, 0.14068795556746005 (USERS that saw either movie)

# 10) Tests whether the median ratings of Interstellar (2014) and Wolf of Wall Street (2013) are different.
u4,p4 = stats.mannwhitneyu(d7_movies['Interstellar (2014)'], d7_movies['The Wolf of Wall Street (2013)'])
# 5780.0, 0.2574930954940875 (USERS that saw ALL movies)
# 88891.5, 0.37516616152190096 (USERS that saw these two movies)
# 216256.0, 0.039890796641855915 (USERS that saw either movie)

#%%

# NOW TEST FOR ALL PEOPLE WHO HAVE SEEN THE MOVIES INDIVIDUALLY

def run_tests(data, movie1, movie2):
    # Drop rows with NaN values for the specific pair
    pair_data = data.dropna(subset=[movie1, movie2])
    pair_data = pair_data[[movie1, movie2]]
    
    # Mann-Whitney U test
    mwu_result = stats.mannwhitneyu(
        pair_data[movie1],
        pair_data[movie2],
    )
    
    # Kolmogorov-Smirnov test
    ks_result = stats.ks_2samp(
        pair_data[movie1],
        pair_data[movie2]
    )
    
    return mwu_result, ks_result


def run_individual_tests(data, movie1, movie2):
    # Select non-NaN ratings for each movie individually
    ratings1 = data[movie1].dropna()
    ratings2 = data[movie2].dropna()
    
    # Mann-Whitney U test
    mwu_result = stats.mannwhitneyu(
        ratings1,
        ratings2,
        alternative='two-sided'
    )
    
    # Kolmogorov-Smirnov test
    ks_result = stats.ks_2samp(
        ratings1,
        ratings2
    )
    
    return mwu_result, ks_result




