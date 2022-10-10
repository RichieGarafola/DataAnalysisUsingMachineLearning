import streamlit as st 
import pandas as pd
import numpy as np
from pathlib import Path

csv_path1 = Path('../Resources/ds1.csv')
csv_path2 = Path('../Resources/ds2.csv')
ds1_df = pd.read_csv(csv_path1)
ds2_df = pd.read_csv(csv_path2)
ds1_df = ds1_df.drop(ds1_df.columns[0], axis=1)
ds2_df = ds2_df.drop(ds2_df.columns[0], axis=1)
pca_path = Path('./pca.csv')
pca_df = pd.read_csv(pca_path)
pca_df = pca_df.drop(pca_df.columns[0], axis=1)
predictions_path = Path('./predictions_df.csv')
predictions_df = pd.read_csv(predictions_path)
predictions_df = predictions_df.drop(predictions_df.columns[0], axis=1)



q1_path = Path('./PostsCreated2017.csv')
q1_df = pd.read_csv(q1_path)

q2_path = Path('./TopPost.csv')
q2_df = pd.read_csv(q2_path)

q3_path = Path('./Top10Tags.csv')
q3_df = pd.read_csv(q3_path)



############
# Dashboard
############
st.title("Assessment")

st.header("1.")
st.markdown("""
Consider data set 1 (ds1.csv). The data set comprises features (the Five xs) along with three sequences that may or may not be generated from the features (3 ys).
""") 
st.markdown("""
a. Describe the data set in a few sentences. E.g. What are the distributions of each feature? Summary statistics?
""")
st.write(ds1_df.describe())

tab1, tab2, tab3, tab4, tab5 = st.tabs(["x1", "x2", "x3", "x5", "x6"])

with tab1:
    st.header("Feature x1")
    st.image("./Images/ds1HistogramDistributionx1a.png")
    st.image("./Images/ds1HistogramDistributionx1b.png")
    st.subheader("x1 - Uniform Distribution (Equal spread, no peaks)")

with tab2:
    st.header("Feature x2")
    st.image("./Images/ds1HistogramDistributionx2a.png")
    st.image("./Images/ds1HistogramDistributionx2b.png")
    st.subheader("x2 - Normal Distribution (Unimodal, symmetric, Bell Curve)")

with tab3:
    st.header("Feature x3")
    st.image("./Images/ds1HistogramDistributionx3a.png")
    st.image("./Images/ds1HistogramDistributionx3b.png")
    st.subheader("x3 - Normal Distribution (Unimodal, symmetric, Bell Curve)")
    
with tab4:
    st.header("Feature x5")
    st.image("./Images/ds1HistogramDistributionx5a.png")
    st.image("./Images/ds1HistogramDistributionx5b.png")
    st.subheader("x5 - Right Skewed Distribution (Positively-skewed)")
    
with tab5:
    st.header("Feature x6")
    st.image("./Images/ds1HistogramDistributionx6a.png")
    st.image("./Images/ds1HistogramDistributionx6b.png")
    st.subheader("x6 - Normal Distribution (Unimodal, symmetric, Bell Curve, wide range")

st.markdown("""

---
            
**b.** Try to come up with a predictive model, e.g. y = f(x_1 , … , x_n) for each y sequence. Describe your models and how you came up with them. What (if any) are the predictive variables? How good would you say each of your models is?

---

""")

st.markdown("""

Considering that the question includes a formula of Linear Regression, and we want to predict the value of a variable based on the value of another variable, this is the model we will work with. A good indication of when to use a Linear Regression model is when the histograms look approximately normal. Although there are much more accurate models, a Linear Regression is highly interpretable as it describes the relationship between a dependent variable, y, and one or more independent variables, X. The dependent variable is also called the response variable. Independent variables are called predictor variables.

XGBoost will also be used as it is a scalable and highly accurate implementation of gradient boosting. It has both linear model solver and tree learning algorithms. So, what makes it fast is its capacity to do parallel computation on a single machine. It also has additional features for doing cross-validation and finding important variables which makes it essential to be used as a Feature Selection technique.

It is helpful to visualize the correlations to have a better understanding of the relationships within the dataset. This will show which variables are predictive by nature, a heatmap will be used to achieve this. After exploring the heatmap, the following conclusions can be drawn: 
""")

st.image('./Images/ds1_df_correlation_matrix.png')

st.markdown("""
- yb is closely correlated to x1 and x3

- ya has the most correlations, x1, x2, and x3, but less strong of a relationship

- yc does not seem to be correlated to any of the feature sets
""")


st.subheader("XGBoost Model Assessment")

tab1, tab2, tab3 = st.tabs(["ya", "yb", "yc"])

with tab1:
    st.header("ya")
    st.image("./Images/XGBoostModelFeatureImportance_ya.png") 
    st.image("./Images/XGBoostModelTree_ya.png")

    
with tab2:
    st.header("yb")
    st.image("./Images/XGBoostModelFeatureImportance_yb.png") 
    st.image("./Images/XGBoostModelTree_yb.png")


with tab3:
    st.header("yc")
    st.image("./Images/XGBoostModelFeatureImportance_yc.png") 
    st.image("./Images/XGBoostModelTree_yc.png")
    
st.subheader("Linear Regression Model Assessment")

tab1, tab2, tab3 = st.tabs(["ya", "yb", "yc"])

with tab1:
    st.header("ya")
    st.image("./Images/LinearRegressionModelHistogram_ya.png")    
    st.image("./Images/LinearRegressionModelAssessment_ya.png")

    st.image("./Images/LinearRegressionModelAssessmentPredictedValues_ya.png")
    
with tab2:
    st.header("yb")
    st.image("./Images/LinearRegressionModelHistogram_yb.png")    
    st.image("./Images/LinearRegressionModelAssessment_yb.png")

    st.image("./Images/LinearRegressionModelAssessmentPredictedValues_yb.png")

with tab3:
    st.header("yc")
    st.image("./Images/LinearRegressionModelHistogram_yc.png")    
    st.image("./Images/LinearRegressionModelAssessment_yc.png")

    st.image("./Images/LinearRegressionModelAssessmentPredictedValues_yc.png")

st.header("2.")
st.markdown("""
Consider data set 2 (ds2.csv). The dataset comprises a set of observations that correspond to multiple groups.
""") 
st.markdown("""
a. Describe the data in a few sentences
""")
st.write(ds2_df.describe())

st.markdown("""
b. How would you visualize this data set?
""")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"])

with tab1:
    st.header("X1")
    st.image("./Images/ds2HistogramDistributionx1a.png")
    st.image("./Images/ds2HistogramDistributionx1b.png")

with tab2:
    st.header("X2")
    st.image("./Images/ds2HistogramDistributionx2a.png")
    st.image("./Images/ds2HistogramDistributionx2b.png")

with tab3:
    st.header("X3")
    st.image("./Images/ds2HistogramDistributionx3a.png")
    st.image("./Images/ds2HistogramDistributionx3b.png")

with tab4:
    st.header("X4")
    st.image("./Images/ds2HistogramDistributionx4a.png")
    st.image("./Images/ds2HistogramDistributionx4b.png")

with tab5:
    st.header("X5")
    st.image("./Images/ds2HistogramDistributionx5a.png")
    st.image("./Images/ds2HistogramDistributionx5b.png")

with tab6:
    st.header("X6")
    st.image("./Images/ds2HistogramDistributionx6a.png")
    st.image("./Images/ds2HistogramDistributionx6b.png")
    
with tab7:
    st.header("X7")
    st.image("./Images/ds2HistogramDistributionx7a.png")
    st.image("./Images/ds2HistogramDistributionx7b.png")

with tab8:
    st.header("X8")
    st.image("./Images/ds2HistogramDistributionx8a.png")
    st.image("./Images/ds2HistogramDistributionx8b.png")

with tab9:
    st.header("X9")
    st.image("./Images/ds2HistogramDistributionx9a.png")
    st.image("./Images/ds2HistogramDistributionx9b.png")

with tab10:
    st.header("X10")
    st.image("./Images/ds2HistogramDistributionx10a.png")
    st.image("./Images/ds2HistogramDistributionx10b.png")
    

st.markdown("""
c. Can you identify the number of groups in the data and assign each row to its group?""")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("K Means")
    st.image("./Images/KMeansCluster.png")

with col2:
    st.header("PCA")
    st.dataframe(pca_df)
with col3:
    st.header("Predictions")
    st.dataframe(predictions_df["Clusters Predictions"])

st.markdown("""
d. Can you create a good visualization of your groupings?
""")

st.write("Scatter Plot Matrix")
st.image("./Images/ScatterplotMatrix.png")


st.write("PCA Cluster")
st.image("../Images/pca_cluster.png")

st.write("Predictions")
st.image("../Images/pca_scatter.png")


st.header("3.")
st.markdown("""
Stack Overflow provides a tool at https://data.stackexchange.com/stackoverflow/query/new that allows SQL queries to be run against their data. After reviewing the database schema provided on their site, please answer the questions below by providing both your answer and the query used to derive it.

---
""")
st.markdown("""
a. How many posts were created in 2017?
---
"""
)
st.markdown("""
        SELECT 

        count(Id) AS PostsCreated2017

        FROM Posts

        WHERE year(CreationDate) = '2017';
""")
st.write(q1_df)



st.markdown("""
b. What post/question received the most answers?
---
"""
)
st.markdown("""
        SELECT Top 1

        Id, Title, AnswerCount

        from Posts

        order by AnswerCount desc;
""")
st.write(q2_df)


st.markdown("""
c. For posts created in 2020, what were the top 10 tags?
---
"""
)

st.markdown("""
        Select Top 10

        Tags, count(Id) AS Quantity

        from Posts

        where DATEPART(year, CreationDate)=2020

        and Tags is not null

        group by Tags

        ORDER BY count(Id) DESC;
""")
st.write(q3_df)


st.markdown("""
d. *BONUS* For the questions created in 2017, what was the average time (in seconds) between when the question was created and when the accepted answer was provided?
""") 