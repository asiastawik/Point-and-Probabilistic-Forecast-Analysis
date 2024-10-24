# Point and Probabilistic Forecast Analysis with Pandas and Numpy

## Project Overview
This project focuses on analyzing point and probabilistic forecasts using Python libraries such as `pandas`, `numpy`, and `time`. The tasks involve loading and processing datasets, calculating error metrics, comparing performance between `pandas` and `numpy`, and evaluating forecast accuracy using various statistical techniques.

## Task 1: Point Forecast Analysis

### 1.1: Downloading and Examining the Dataset
- Download the dataset containing 3 point forecasts (L02b List 5).
- Examine the dataset to understand its structure and the type of data stored in each column.

### 1.2: Loading Data with Pandas and Numpy
- Import the `pandas` library and load the dataset using `pd.read_csv()` into a `pandas` dataframe (`pdfore`).
- Use `numpy` to load the data into a `numpy` array (`npfore`) using `np.loadtxt(…, skiprows=…, usecols=…)`. Remember, `numpy` does not use column headers and primarily handles numerical data.

### 1.3: Performance Comparison Between Pandas and Numpy
- Measure the time taken to load the dataset into both `pdfore` and `npfore` using the `time()` function from the `time` library.
- Compare the load times of `pandas` vs `numpy` and evaluate the difference in performance.

### 1.4: Error Metrics Calculation
- For the last three columns representing the forecasts of Demand values, calculate the following error metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
- Perform the calculations for both `pandas` and `numpy`.
- Repeat the calculations 100 times to compare the sum of runtimes between `pandas` and `numpy`.
- For `numpy`, compute the RMSE using two approaches:
  - Using `np.power()`
  - Using `array**2` 
- Compare which method is faster and analyze the reasons for the difference in execution time.

### 1.5: Creating an Average Forecast
- Create an average forecast based on the 3 forecasts from the dataset.
- Calculate MAE, RMSE, and MAPE for the average forecast and compare these values with the individual forecasts.

## Task 2: Probabilistic Forecast Analysis

### 2.1: Downloading and Analyzing the Probabilistic Dataset
- Download the probabilistic forecast dataset (L02b List 5 probabilistic forecasts) containing 99 percentiles for all entries from Task 1.
- Use `numpy` to load the data and analyze it.

### 2.2: Coverage Analysis of Prediction Intervals
- Using the Demand values from Task 1 as real observations, compute the coverage of the probabilistic forecast’s prediction intervals:
  - 50% prediction interval
  - 90% prediction interval
  - 98% prediction interval
- Analyze how well the probabilistic forecast covers the actual Demand values across these intervals.

## Libraries Used
- **Pandas** for data loading, manipulation, and analysis.
- **Numpy** for numerical computations and array operations.
- **Time** for performance measurement and runtime comparison.
