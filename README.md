
# Rating Product & Sorting Reviews in Amazon

![img](https://cdn.searchenginejournal.com/wp-content/uploads/2021/12/google-reviews-map-61b9ed75818cc-sej.png)

## Why Rating Product & Sorting Reviews is Important?

Rating products and sorting reviews are both important because they help consumers make informed purchase decisions.

Product ratings provide a quick and easy way for consumers to evaluate the overall quality of a product based on the experiences of others. A high rating indicates that the majority of users found the product to be satisfactory, while a low rating may suggest that the product has some significant flaws or shortcomings. By considering the product rating, consumers can quickly determine whether a product is worth their time and money, and they can use it as a first step in the decision-making process.

Sorting reviews based on various criteria such as most recent, highest rated, or most helpful allows consumers to find the most relevant and informative reviews. Sorting reviews by date can help consumers see if any recent changes have affected the product's quality, while sorting by rating can help identify the best and worst aspects of the product. Sorting reviews by the most helpful can highlight reviews that are most relevant and informative to consumers, as determined by other users.

Together, rating products and sorting reviews can help consumers make more informed decisions by providing a quick overview of a product's overall quality and identifying the most relevant and informative reviews. By using these tools, consumers can save time and money, make more confident purchase decisions, and minimize the risk of buyer's remorse.

## Business Problem

One of the most important problems in e-commerce is the correct calculation of the points given to the products after sales. The solution to this problem means providing greater customer satisfaction for the e-commerce site, prominence of the product for the sellers and a seamless shopping experience for the buyers. Another problem is the correct ordering of the comments given to the products. Since misleading comments will directly affect the sale of the product, it will cause both financial loss and loss of customers. In the solution of these 2 basic problems, while the e-commerce site and the sellers will increase their sales, the customers will complete the purchasing journey without any problems.

## Features of Dataset

- Total Variables : 12
- Total Row : 4915
- CSV File Size : 71.9 MB

## The story of the dataset

This dataset, which includes Amazon product data, includes product categories and various metadata. The product with the most reviews in the electronics category has user ratings and reviews.

| Variable | Description  | 
| --- | ---| 
| reviewerID | User ID   | 
| asin | Product ID   | 
| reviewerName | User name   | 
| helpful | Useful  rating   | 
| reviewText | Evaluation   | 
| overall | Product rating   | 
| summary | Rating summary   | 
| unixReviewTime | Evaluation time  | 
| reviewTime | Number of days since evaluation | 
| day_diff | The number of times the evaluation was found useful | 
| helpful_yes | Number of votes given to the evaluation| 
| total_vote | Evaluation time Raw | 

## Methods and libraries used in the project

- pandas, numpy, datetime, matplotlib.pyplot, scipy.stats
- wilson_lower_bound

## Requirements.txt

- Please review the 'requirements.txt' file for required libraries.


