
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500)

######################################################################################################
# TASK 1: Calculate Average Rating Based on Current Comments and Compare with Existing Average Rating.
######################################################################################################

# In the shared data set, users gave points and comments to a product.
# Our aim in this task is to evaluate the scores given by weighting them by date.
# It is necessary to compare the first average score with the weighted score according to the date to be obtained.

######################################################################################################
# Step 1: Read the Data Set and Calculate the Average Score of the Product.
######################################################################################################

df = pd.read_csv('Measurement_Problems/amazon_review.csv')


def check_dataframe(df, row_num=5):
    print("*************** Dataset Shape ***************")
    print("No. of Rows:", df.shape[0], "\nNo. of Columns:", df.shape[1])
    print("*************** Dataset Information ***************")
    print(df.info())
    print("*************** Types of Columns ***************")
    print(df.dtypes)
    print(f"*************** First {row_num} Rows ***************")
    print(df.head(row_num))
    print(f"*************** Last {row_num} Rows ***************")
    print(df.tail(row_num))
    print("*************** Summary Statistics of The Dataset ***************")
    print(df.describe().T)
    print("*************** Dataset Missing Values Analysis ***************")
    print(missing_values_analysis(df))


def missing_values_analysis(df):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=True)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df


check_dataframe(df)

"""
*************** Dataset Shape ***************
No. of Rows: 4915 
No. of Columns: 12
*************** Dataset Information ***************
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4915 entries, 0 to 4914
Data columns (total 12 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   reviewerID      4915 non-null   object 
 1   asin            4915 non-null   object 
 2   reviewerName    4914 non-null   object 
 3   helpful         4915 non-null   object 
 4   reviewText      4914 non-null   object 
 5   overall         4915 non-null   float64
 6   summary         4915 non-null   object 
 7   unixReviewTime  4915 non-null   int64  
 8   reviewTime      4915 non-null   object 
 9   day_diff        4915 non-null   int64  
 10  helpful_yes     4915 non-null   int64  
 11  total_vote      4915 non-null   int64  
dtypes: float64(1), int64(4), object(7)
memory usage: 460.9+ KB
None
*************** Types of Columns ***************
reviewerID         object
asin               object
reviewerName       object
helpful            object
reviewText         object
overall           float64
summary            object
unixReviewTime      int64
reviewTime         object
day_diff            int64
helpful_yes         int64
total_vote          int64
dtype: object
*************** First 5 Rows ***************
       reviewerID        asin  reviewerName helpful                                         reviewText  overall                                 summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
0  A3SBTW3WS4IQSN  B007WTAJTO           NaN  [0, 0]                                         No issues.    4.000                              Four Stars      1406073600  2014-07-23       138            0           0
1  A18K1ODH1I2MVB  B007WTAJTO          0mie  [0, 0]  Purchased this for my device, it worked as adv...    5.000                           MOAR SPACE!!!      1382659200  2013-10-25       409            0           0
2  A2FII3I2MBMUIA  B007WTAJTO           1K3  [0, 0]  it works as expected. I should have sprung for...    4.000               nothing to really say....      1356220800  2012-12-23       715            0           0
3   A3H99DFEG68SR  B007WTAJTO           1m2  [0, 0]  This think has worked out great.Had a diff. br...    5.000  Great buy at this price!!!  *** UPDATE      1384992000  2013-11-21       382            0           0
4  A375ZM4U047O79  B007WTAJTO  2&amp;1/2Men  [0, 0]  Bought it with Retail Packaging, arrived legit...    5.000                        best deal around      1373673600  2013-07-13       513            0           0
*************** Last 5 Rows ***************
          reviewerID        asin reviewerName helpful                                         reviewText  overall                        summary  unixReviewTime  reviewTime  day_diff  helpful_yes  total_vote
4910  A2LBMKXRM5H2W9  B007WTAJTO       ZM "J"  [0, 0]  I bought this Sandisk 16GB Class 10 to use wit...    1.000       Do not waste your money.      1374537600  2013-07-23       503            0           0
4911   ALGDLRUI1ZPCS  B007WTAJTO           Zo  [0, 0]  Used this for extending the capabilities of my...    5.000                    Great item!      1377129600  2013-08-22       473            0           0
4912  A2MR1NI0ENW2AD  B007WTAJTO    Z S Liske  [0, 0]  Great card that is very fast and reliable. It ...    5.000  Fast and reliable memory card      1396224000  2014-03-31       252            0           0
4913  A37E6P3DSO9QJD  B007WTAJTO     Z Taylor  [0, 0]  Good amount of space for the stuff I want to d...    5.000              Great little card      1379289600  2013-09-16       448            0           0
4914   A8KGFTFQ86IBR  B007WTAJTO          Zza  [0, 0]  I've heard bad things about this 64gb Micro SD...    5.000                So far so good.      1388620800  2014-02-01       310            0           0
*************** Summary Statistics of The Dataset ***************
                  count           mean          std            min            25%            50%            75%            max
overall        4915.000          4.588        0.997          1.000          5.000          5.000          5.000          5.000
unixReviewTime 4915.000 1379465001.668 15818574.323 1339200000.000 1365897600.000 1381276800.000 1392163200.000 1406073600.000
day_diff       4915.000        437.367      209.440          1.000        281.000        431.000        601.000       1064.000
helpful_yes    4915.000          1.311       41.619          0.000          0.000          0.000          0.000       1952.000
total_vote     4915.000          1.521       44.123          0.000          0.000          0.000          0.000       2020.000
*************** Dataset Missing Values Analysis ***************
              Total Missing Values  Ratio
reviewerName                     1  0.020
reviewText                       1  0.020

"""

product_rating = df['overall'].mean()
# product_rating = 4.587589013224822

######################################################################################################
# Step 2: Calculate the Weighted Average of Score by Date.
######################################################################################################

df['reviewTime'] = pd.to_datetime(df['reviewTime'], format='%Y-%m-%d')
# df['reviewTime'][0]: Timestamp('2014-07-23 00:00:00')


def weighted_rating(df, m, C):
    """

    This code is used to calculate a weighted score for a product. This score is based on both the product's
    popularity and users' average rating for the product.

    The function takes three variables:

    df: A DataFrame containing the properties of the products
    m: Minimum number of votes
    C: Average score

    The function calculates a weighted score for each product in the df DataFrame. For each product,
    the function determines the total number of votes (v) and the average score (R). It then calculates
    the weighted score using the formula.

    The functioning of the function is as follows:

    df['total_vote']: Gets the column containing the total number of votes for each product.

    df['overall']: Gets the column containing the average score for each product.

    m = 1: Determines the minimum number of votes. The function necessarily takes this value, but a specific
    value is assigned within the function.

    if v == 0: The function returns 0 if the total number of votes is zero.

    else: If the total number of votes is non-zero, calculate the weighted score using the following formula:

    (v / (v + m) * R): A score based on the popularity of the product
    (m / (m + v) * C): A score based on the average rating of the product

    These two scores are added together and the weighted score is calculated.

    Finally, the calculated weighted score is returned by the function.
    This function calculates a weighted score based on the popularity and average score of the product,
    and using these scores you can rank the products.
    """
    v = df['total_vote']
    R = df['overall']
    m = 1
    if v == 0:
        return 0
    else:
        return (v / (v + m) * R) + (m / (m + v) * C)


df['weighted_rating'] = df.apply(lambda x: weighted_rating(x, 1, product_rating), axis=1)
df['weighted_rating'].mean()
# df['weighted_rating'].mean(): 0.4608903403691315

# Calculation of all weighted average ratings on a year-month basis with matplotlib
grouped = df.groupby(pd.Grouper(key='reviewTime', freq='M')).agg({'weighted_rating': np.mean})
plt.plot(grouped.index, grouped['weighted_rating'])
plt.title('Monthly Weighted Average Ratings')
plt.xlabel('Time')
plt.ylabel('Weighted Average Rating')
plt.show()

######################################################################################################
# Task 2: Specify 20 Reviews for the Product to be Displayed on the Product Detail Page.
######################################################################################################

######################################################################################################
# Step 1. Generate the helpful_no variable
######################################################################################################

df['helpful_no'] = df['total_vote'] - df['helpful_yes']


# Note:
# total_vote is the total number of up-downs given to a comment.
# up means helpful.
# There is no helpful_no variable in the data set, it must be generated over existing variables.

######################################################################################################
# Step 2. Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound Scores and add to dataframe
######################################################################################################


def score_pos_neg_diff(positive, negative):
    return positive - negative


def score_average_rating(up_vote, down_vote):
    total_vote = up_vote + down_vote
    if total_vote == 0:
        return 0
    else:
        score = up_vote / (up_vote + down_vote)
        return score


def wilson_lower_bound(up_vote, down_vote, confidence=0.95):
    total_vote = up_vote + down_vote
    if total_vote == 0:
        return 0
    else:
        z = norm.ppf(1 - (1 - confidence) / 2)
        phat = 1.0 * up_vote / total_vote
        score = (phat + z * z / (2 * total_vote) - z * np.sqrt((phat * (1 - phat) + z * z / (4 * total_vote))
                                                               / total_vote)) / (1 + z * z / total_vote)
        return score


df['score_pos_neg_diff'] = df.apply(lambda x: score_pos_neg_diff(x['helpful_yes'], x['helpful_no']), axis=1)
df['score_average_rating'] = df.apply(lambda x: score_average_rating(x['helpful_yes'], x['helpful_no']), axis=1)
df['wilson_lower_bound'] = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']), axis=1)

######################################################################################################
# Step 3. Identify 20 Comments and Interpret Results.
######################################################################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
df.describe().T