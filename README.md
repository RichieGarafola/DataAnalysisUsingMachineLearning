## INSTRUCTIONS:

· Please complete the below exercises using Python in a Jupyter notebook.

· For the SQL exercises (#3), include your query and results. You do not need to query this dataset directly from the notebook.


1. Consider data set 1 (ds1.csv). The data set comprises features (the Five xs) along with three sequences that may or may not be generated from the features (3 ys).

a. Describe the data set in a few sentences. E.g. What are the distributions of each feature? Summary statistics?

b. Try to come up with a predictive model, e.g. y = f(x_1 , … , x_n) for each y sequence. Describe your models and how you came up with them. What (if any) are the predictive variables? How good would you say each of your models is?

2. Consider data set 2 (ds2.csv). The dataset comprises a set of observations that correspond to multiple groups.

a. Describe the data in a few sentences

b. How would you visualize this data set?

c. Can you identify the number of groups in the data and assign each row to its group?

d. Can you create a good visualization of your groupings?

3. Stack Overflow provides a tool at https://data.stackexchange.com/stackoverflow/query/new that allows SQL queries to be run against their data. After reviewing the database schema provided on their site, please answer the questions below by providing both your answer and the query used to derive it.

a. How many posts were created in 2017?

b. What post/question received the most answers?

c. For posts created in 2020, what were the top 10 tags?

d. *BONUS* For the questions created in 2017, what was the average time (in seconds) between when the question was created and when the accepted answer was provided?
    
---
    
**Libraries Used:**

Pandas, Numpy, Pathlib, Matplotlib, Hvplot, Fitter, Sklearn, XGBoost
    
**Machine Learning:**
    - LinearRegression
    - XGBoost
    - Kmeans
    - PCA
    
**Redact warnings**
Warnings    
    
--- 

### Directory

1. [Read in and Wrangle Data](#Prepare-the-Data)
2. [Explore the data](#Data-Exploration-And-Visualization)
3. [Supervised Machine Learning](#Linear-Regression-and-XGBoost)    
4. [Answer to Question 1](#Answer-to-Question-1) 
5. [Read in and Wrangle Data for 'ds2.csv'](#Prepare-the-'ds2.csv'-Data)
6. [Explore the data for 'ds2.csv'](#Data-Exploration-And-Visualization-for-'ds2.csv')
7. [Unsupervised Machine Learning](#K-Means-and-PCA) 
8. [Answer to Question 2](#Answer-to-Question-2)
9. [SQL](#SQL-Questions)
10. [Appendix](#Appendix)

Visualization is essential, that being said a Streamlit Dashboard will be included.
    
[Create a Custom Dashboard](#Create-a-Custom-Dashboard)    



---

## Instructions

**File:** [Skills Assessment](./MachineLearning.ipynb)

---
    
### Prepare the Data
[Directory](#Directory)

First, read and clean our CSV files for analysis. There are 2 CSV file there were introduced. 
    
    - The first file, "ds1.csv", includes one hundred thousand rows of data broken down by the following five features(x) and three sequences(y) 'x1', 'x2', 'x3', 'x5', 'x6', 'ya', 'yb', 'yc'.
    
    - Second file, "ds2.csv", includes two thousand rows of data broken down by 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'

1. Use Pandas to read the 'ds1.csv' file as a DataFrame.

    * `ds1.csv`:Contains the dataset we will be working with.
    
2. Clean the data:
    - Drop the un needed extra index column from the data using the drop() function. 
    - Specify the columns to be dropped using the columns() function 
    - in this instance we are referring to col 0 as it is the first column
    
3. Check for missing data:
    - Use the isna() function to detect for nulls
    - Use the sum() function to count the amount of missing data

---

### Data-Exploration-And-Visualization

[Directory](#Directory)
    
4. Summary Statistics
    - Give a descriptive summary of all features included in the datasets statistics using the describe() function
        - The statistics included in this function are:
    
            - Count
            - Mean
            - Standard Deviation (STD)
            - min/max
            - 25/50/75 percentiles
    
![ds1 Summary](Images/ds1Summary.png)
    
* Note: Since we will be applying the Pandas describe() function to a dataframe, the result is also returned as a dataframe.

5. Visual Depiction of Distributions for each Feature
    - Create a custom function 'OptimalHistogramDistribution()' that will plot the optimal histogram distribution for each feature in the dataset.
        - The 'fitter()' function is used to find the optimal histogram distribution for each isolated feature.
            - Use the 'get_common_distributions()' function to use the top 10 most common histogram distributions. The function will only plot the top 5 distributions for each dataset to eliminate redundent noise. 
    - Create a 'for loop' that iterates through all of the features in the dataset uses our funtion to find a visualizes the best histogram distributions
    
    - Create a Correlaion Matrix of the features and sequences in the dataset 
        - Use 'corr()' function to find the correlation among the columns in the Dataframe using the ‘Pearson’ method. This method ignores non-numerical columns.

---
    
### Linear-Regression-and-XGBoost
[Directory](#Directory)
    
6. Machine Learning
    - Prepare the data for the models
        - train
        - model
        - fit
        - predict
        - Computes the metric 
    
7. Run the model on each isolated Sequence.
    - Sequence ya
    - Sequence yb
    - Sequence yc
    
## Linear Regression

- Load and visualize the ds1.csv data.
- Using the train_test_split() function, split the dataset and train 80%
- Create a model with scikit-learn using the LinearRegression() function
- Fit the data into the model
- Display the slope
- Display the y-intercept
- Display the model's best fit line formula
- Make predictions using the X set
- Create a copy of the original data and Add a column with the predicted values
- Create a plot of the predicted values
- Compute the metrics for the linear regression model:
        - score 
        - r2
        - mse 
        - rmse 
        - std 

## XGBoost
Important note! matplotlib needs to be forced when using XGBoost, otherwise plots render weird.
Must be lack of version compatability

- Load and visualize the ds1.csv data.
- convert the dataset into an optimized data structure called Dmatrix using the DMatrix() function. XGBoost supports this structure and gives it acclaimed performance and efficiency gains.
- Using the train_test_split() function, split the dataset and train 80%
- Instantiation of the XBBoost Regressor model by calling the XGBRegressor() function
- Fit the regressor model to the training set using the fit() method
- Predict the model of the test set using the predict() method
- Use a k-fold cross validation to build a more robust model
- Create a hyper-parameter dictionary 'params' which holds all the hyper-parameters and their values as key-values
    params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,
                    'max_depth': 5, 'alpha': 10}
- Visualize feature importance using the plot_importance() method
- Visualize the Boosting Tree using the plot_tree() function

---
### Answer-to-Question-1   

[Directory](#Directory)

---
    
1. Consider data set 1 (ds1.csv). The data set comprises features (the Five xs) along with three sequences that may or may not be generated from the features (3 ys).
    a. Describe the data set in a few sentences. E.g. What are the distributions of each feature? Summary statistics?

    b. Try to come up with a predictive model, e.g. y = f(x_1 , … , x_n) for each y sequence. Describe your models and how you came up with them. What (if any) are the predictive variables? How good would you say each of your models is?
    
---
**1a Answer:**
    
![ds1 Summary](Images/ds1Summary.png)
![feature distribution](Images/FeatureDistribution.png)    
![sequence distribution](Images/SequenceDistribution.png)
![ds1_df - x1](Images/ds1HistogramDistributionx1a.png)
![ds1_df - x1](Images/ds1HistogramDistributionx1b.png)
![ds1_df - x2](Images/ds1HistogramDistributionx2a.png) 
![ds1_df - x2](Images/ds1HistogramDistributionx2b.png)   
![ds1_df - x3](Images/ds1HistogramDistributionx3a.png)
![ds1_df - x3](Images/ds1HistogramDistributionx3b.png)
![ds1_df - x5](Images/ds1HistogramDistributionx5a.png)   
![ds1_df - x5](Images/ds1HistogramDistributionx5b.png)    
![ds1_df - x6](Images/ds1HistogramDistributionx6a.png)
![ds1_df - x6](Images/ds1HistogramDistributionx6b.png)    
![ds1_df - ya](Images/ds1HistogramDistributionya1.png)    
![ds1_df - ya](Images/ds1HistogramDistributionya2.png)    
![ds1_df - yb](Images/ds1HistogramDistributionyb1.png)    
![ds1_df - yb](Images/ds1HistogramDistributionyb2.png)
![ds1_df - yc](Images/ds1HistogramDistributionyc1.png)
![ds1_df - yc](Images/ds1HistogramDistributionyc2.png)   

    
    
**1b. Answer:**

Considering that the question includes a formula of Linear Regression, and we want to predict the value of a variable based on the value of another variable, this is the model we will work with. A good indication of when to use a Linear Regression model is when the histograms look approximately normal. Although there are much more accurate models, a Linear Regression is highly interpretable as it describes the relationship between a dependent variable, y, and one or more independent variables, X. The dependent variable is also called the response variable. Independent variables are called predictor variables.

XGBoost will also be used as it is a scalable and highly accurate implementation of gradient boosting. It has both linear model solver and tree learning algorithms. So, what makes it fast is its capacity to do parallel computation on a single machine. It also has additional features for doing cross-validation and finding important variables which makes it essential to be used as a Feature Selection technique.

It is helpful to visualize the correlations to have a better understanding of the relationships within the dataset. This will show which variables are predictive by nature, a heatmap will be used to achieve this. After exploring the heatmaps, the following conclusions can be drawn:

-    yb is closely correlated to x1 and x3
-    ya has the most correlations, x1, x2, and x3, but less strong of a relationship
-    yc does not seem to be correlated to any of the feature sets

![ds1_df Correlation Matrix](Images/ds1_df_correlation_matrix.png)
    
![Linear Regression Model Assessment ya](Images/LinearRegressionModelAssessment_ya.png)
![Linear Regression Model Assessment PredictedValues ya](Images/LinearRegressionModelAssessmentPredictedValues_ya.png)    
![Linear Regression Histogram ya](Images/LinearRegressionModelHistogram_ya.png)    
    
![Linear Regression Model Assessment yb](Images/LinearRegressionModelAssessment_yb.png)
![Linear Regression Model Assessment PredictedValues yb](Images/LinearRegressionModelAssessmentPredictedValues_yb.png)    
![Linear Regression Histogram yb](Images/LinearRegressionModelHistogram_yb.png)
    
![Linear Regression Model Assessment yc](Images/LinearRegressionModelAssessment_yc.png)
![Linear Regression Model Assessment PredictedValues yc](Images/LinearRegressionModelAssessmentPredictedValues_yc.png)
![Linear Regression Histogram yc](Images/LinearRegressionModelHistogram_yc.png)
    
![XGBoost Model Feature Importance ya](Images/XGBoostModelFeatureImportance_ya.png)
![XGBoost Model Tree ya](Images/XGBoostModelTree_ya.png)
    
![XGBoost Model Feature Importance ya](Images/XGBoostModelFeatureImportance_yb.png)
![XGBoost Model Tree ya](Images/XGBoostModelTree_yb.png)

![XGBoost Model Feature Importance ya](Images/XGBoostModelFeatureImportance_yc.png)
![XGBoost Model Tree ya](Images/XGBoostModelTree_yc.png)
    
    
---
# 2
---
### Prepare the 'ds2.csv' Data 

[Directory](#Directory)
    
    
    - The file, "ds2.csv", includes two thousand rows of data broken down by 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10'

1. Use Pandas to read the 'ds2.csv' file as a DataFrame.

    * `ds2.csv`:Contains the dataset we will be working with.
        - Drop the un needed extra index column
2. Clean the data:

    - Give a descriptive summary of all features included in the datasets statistics using the describe() function
        - The statistics included in this function are:

            Count
            Mean
            Standard Deviation (STD)
            min/max
            25/50/75 percentiles
* Note: Since we will be applying the Pandas describe() function to a dataframe, the result is also returned as a dataframe.e data using the drop() function. 
    - Specify the columns to be dropped using the columns() function 
    - in this instance we are referring to col 0 as it is the first column

---
    
### Data-Exploration-And-Visualization-for-'ds2.csv'

[Directory](#Directory)

3. Summary Statistics
    - Give a descriptive summary of all features included in the datasets statistics using the describe() function
        - The statistics included in this function are:

            Count
            Mean
            Standard Deviation (STD)
            min/max
            25/50/75 percentiles
* Note: Since we will be applying the Pandas describe() function to a dataframe, the result is also returned as a dataframe.
    
4. Visual Depiction of Distributions for each Feature
    - Use the function 'OptimalHistogramDistribution()' introduced earlier to visualize the data.
    
---

### K Means and PCA    
    
5. Identify the number of groups in the data
    - Use the elbow method to determine the optimal number of clusters (groups), k, in the 'ds2' dataset.
    - Once the elbow curve has been established, evaluate the two most likely values for k using the K-means algorithm and a scatter plot.
        - Scale the data by using the StandardScaler() module to normalize the DataFrame values.
        - Create two lists: one to hold the list of inertia scores and another for the range of k values (from 1 to 11) to analyze.
        - Define a K-means model by using a 'for-loop' to evaluate each instance of k
            - fit the K-means model based on the scaled DataFrame
            - Append the model’s inertia to the empty inertia list.
    
        - Define a K-means model using k to define the clusters
            - fit the model
            - make predictions
            - add the prediction values to a copy of the scaled DataFrame
        - Store the values for k and the inertia in a Dictionary called elbow_data. 
            Use elbow_data to create a Pandas DataFrame called df_elbow.
    
        - Plot the clusters
        -  Instantiate the PCA instance
            - Use the fit_transform() function from PCA
                - fit the PCA model to the DataFrame. 
                - Review the first 5 rows of list data.
        - Calculate the percentage of the total variance that is captured by the four PCA variables using the explained_variance_ratio_ function from PCA
        - Create a scatterplot matrix to visualize the PCA
        - Create a single figure scatter plot with the PCA overlayed to further visualize classification. 

---
### Answer-to-Question-2 

[Directory](#Directory)
    
**2a. Answer:**   
![ds2 summary](Images/ds2Summary.png)  

**2b. Answer:**  
![Histogram Distribution x1](Images/ds2HistogramDistributionx1a.png)
![Histogram Distribution x1](Images/ds2HistogramDistributionx1b.png)
![Histogram Distribution x2](Images/ds2HistogramDistributionx2a.png)
![Histogram Distribution x2](Images/ds2HistogramDistributionx2b.png)
![Histogram Distribution x3](Images/ds2HistogramDistributionx3a.png)
![Histogram Distribution x3](Images/ds2HistogramDistributionx3b.png)
![Histogram Distribution x4](Images/ds2HistogramDistributionx4a.png)
![Histogram Distribution x4](Images/ds2HistogramDistributionx4b.png)
![Histogram Distribution x5](Images/ds2HistogramDistributionx5a.png)
![Histogram Distribution x5](Images/ds2HistogramDistributionx5b.png)
![Histogram Distribution x6](Images/ds2HistogramDistributionx6a.png)
![Histogram Distribution x6](Images/ds2HistogramDistributionx6b.png)
![Histogram Distribution x7](Images/ds2HistogramDistributionx7a.png)
![Histogram Distribution x7](Images/ds2HistogramDistributionx7b.png)
![Histogram Distribution x8](Images/ds2HistogramDistributionx8a.png)
![Histogram Distribution x8](Images/ds2HistogramDistributionx8b.png)
![Histogram Distribution x9](Images/ds2HistogramDistributionx9a.png)
![Histogram Distribution x9](Images/ds2HistogramDistributionx9b.png)
![Histogram Distribution x10](Images/ds2HistogramDistributionx10a.png) 
![Histogram Distribution x10](Images/ds2HistogramDistributionx10b.png)
   
**2c. Answer:**     
![pca scatter](Images/pca_scatter.png) 
![k means cluster](Images/KMeansCluster.png)
![scatter plot matrix](Images/ScatterplotMatrix.png)    
    
### SQL-Questions  

[Directory](#Directory)

---
#3 SQL
---
    
Stack Overflow provides a tool at https://data.stackexchange.com/stackoverflow/query/new that allows SQL queries to be run against their data. After reviewing the database schema provided on their site, please answer the questions below by providing both your answer and the query used to derive it.
a. How many posts were created in 2017?

b. What post/question received the most answers?

c. For posts created in 2020, what were the top 10 tags?

d. BONUS For the questions created in 2017, what was the average time (in seconds) between when the question was created and when the accepted answer was provided?

- 3a. How many posts were created in 2017?

    SELECT 
    
    count(Id) AS PostsCreated2017
    
    FROM Posts
    
    WHERE year(CreationDate) = '2017';
    
**File:** [Skills Assessment 3a](./Resources/PostsCreated2017.csv)
    
- 3b. What post/question received the most answers?

    SELECT Top 1
    
    Id, Title, AnswerCount
    
    from Posts
    
    order by AnswerCount desc;

**File:** [Skills Assessment 3b](./Resources/TopPost.csv)


    
- 3c. For posts created in 2020, what were the top 10 tags?

    Select Top 10
    
    Tags, count(Id) AS Quantity
    
    from Posts
    
    where DATEPART(year, CreationDate)=2020
    
    and Tags is not null
    
    group by Tags
   
    ORDER BY count(Id) DESC;   
    
**File:** [Skills Assessment 3c](./Resources/Top10Tags.csv)

### Create a Custom Dashboard

[Directory](#Directory)

Streamlit Dashboard:
**File:** [Dashboard](./Dashboard/Dashboard.py)


**View the Streamlit application**
Run the Streamlit application to view a Dashboard for easier visualization and navigation. To do so, complete the following steps:

1. In the terminal, navigate to the projects folder.
2. In the terminal, run the Streamlit application by using streamlit run "Dashboard.py"

### Appendix
## Appendix

[Directory](#Directory)

* ![ds1_df Correlation Matrix](Images/ds1_df_correlation_matrix.png)
* ![ds1_df - x1](Images/ds1HistogramDistributionx1a.png)
* ![ds1_df - x1](Images/ds1HistogramDistributionx1b.png)
* ![ds1_df - x2](Images/ds1HistogramDistributionx2a.png) 
* ![ds1_df - x2](Images/ds1HistogramDistributionx2b.png)   
* ![ds1_df - x3](Images/ds1HistogramDistributionx3a.png)
* ![ds1_df - x3](Images/ds1HistogramDistributionx3b.png)
* ![ds1_df - x5](Images/ds1HistogramDistributionx5a.png)   
* ![ds1_df - x5](Images/ds1HistogramDistributionx5b.png)    
* ![ds1_df - x6](Images/ds1HistogramDistributionx6a.png)
* ![ds1_df - x6](Images/ds1HistogramDistributionx6b.png)    
* ![ds1_df - ya](Images/ds1HistogramDistributionya1.png)    
* ![ds1_df - ya](Images/ds1HistogramDistributionya2.png)    
* ![ds1_df - yb](Images/ds1HistogramDistributionyb1.png)    
* ![ds1_df - yb](Images/ds1HistogramDistributionyb2.png)
* ![ds1_df - yc](Images/ds1HistogramDistributionyc1.png)
* ![ds1_df - yc](Images/ds1HistogramDistributionyc2.png)    
* ![ds1 Summary](Images/ds1Summary.png)
* ![Histogram Distribution x1](Images/ds2HistogramDistributionx1a.png)
* ![Histogram Distribution x1](Images/ds2HistogramDistributionx1b.png)
* ![Histogram Distribution x2](Images/ds2HistogramDistributionx2a.png)
* ![Histogram Distribution x2](Images/ds2HistogramDistributionx2b.png)
* ![Histogram Distribution x3](Images/ds2HistogramDistributionx3a.png)
* ![Histogram Distribution x3](Images/ds2HistogramDistributionx3b.png)
* ![Histogram Distribution x4](Images/ds2HistogramDistributionx4a.png)
* ![Histogram Distribution x4](Images/ds2HistogramDistributionx4b.png)
* ![Histogram Distribution x5](Images/ds2HistogramDistributionx5a.png)
* ![Histogram Distribution x5](Images/ds2HistogramDistributionx5b.png)
* ![Histogram Distribution x6](Images/ds2HistogramDistributionx6a.png)
* ![Histogram Distribution x6](Images/ds2HistogramDistributionx6b.png)
* ![Histogram Distribution x7](Images/ds2HistogramDistributionx7a.png)
* ![Histogram Distribution x7](Images/ds2HistogramDistributionx7b.png)
* ![Histogram Distribution x8](Images/ds2HistogramDistributionx8a.png)
* ![Histogram Distribution x8](Images/ds2HistogramDistributionx8b.png)
* ![Histogram Distribution x9](Images/ds2HistogramDistributionx9a.png)
* ![Histogram Distribution x9](Images/ds2HistogramDistributionx9b.png)
* ![Histogram Distribution x10](Images/ds2HistogramDistributionx10a.png) 
* ![Histogram Distribution x10](Images/ds2HistogramDistributionx10b.png)
* ![ds2 summary](Images/ds2Summary.png)
* ![feature distribution](Images/FeatureDistribution.png)    
* ![k means cluster](Images/KMeansCluster.png)
* ![Linear Regression Model Assessment ya](Images/LinearRegressionModelAssessment_ya.png)
* ![Linear Regression Model Assessment yb](Images/LinearRegressionModelAssessment_yb.png)
* ![Linear Regression Model Assessment yc](Images/LinearRegressionModelAssessment_yc.png)
* ![Linear Regression Model Assessment PredictedValues ya](Images/LinearRegressionModelAssessmentPredictedValues_ya.png)
* ![Linear Regression Model Assessment PredictedValues yb](Images/LinearRegressionModelAssessmentPredictedValues_yb.png)
* ![Linear Regression Model Assessment PredictedValues yc](Images/LinearRegressionModelAssessmentPredictedValues_yc.png)
* ![Linear Regression Histogram ya](Images/LinearRegressionModelHistogram_ya.png)
* ![Linear Regression Histogram yb](Images/LinearRegressionModelHistogram_yb.png)
* ![Linear Regression Histogram yc](Images/LinearRegressionModelHistogram_yc.png)
* ![pca scatter](Images/pca_scatter.png)
* ![ds2 pair plot](Images/scaled_ds2_pair_plot.png)
* ![scatter cluster](Images/ScatterCluster.png)
* ![scatter plot matrix](Images/ScatterplotMatrix.png)
* ![sequence distribution](Images/SequenceDistribution.png)
* ![sequence distribution](Resources/PostsCreated2017.csv)
* ![sequence distribution](Resources/TopPose.csv)
* ![sequence distribution](Resources/Top10Tags.csv)

    
