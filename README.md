# Sales Forecasting Project using Olist E-commerce Dataset

This project aims to predict monthly sales for the next 12 months for the Olist E-commerce platform using historical sales data.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Steps](#project-steps)
  - [Step 1: Define the Problem Statement](#step-1-define-the-problem-statement)
  - [Step 2: Gather and Understand the Data](#step-2-gather-and-understand-the-data)
  - [Step 3: Data Preprocessing](#step-3-data-preprocessing)
  - [Step 4: Exploratory Data Analysis (EDA)](#step-4-exploratory-data-analysis-eda)
  - [Step 5: Feature Engineering](#step-5-feature-engineering)
  - [Step 6: Model Building](#step-6-model-building)
  - [Step 7: Model Evaluation](#step-7-model-evaluation)
  - [Step 8: Forecasting Future Sales](#step-8-forecasting-future-sales)
  - [Step 9: Visualization](#step-9-visualization)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

Sales forecasting is crucial for businesses to manage inventory, plan marketing strategies, and make informed financial decisions. This project demonstrates a step-by-step approach to forecasting monthly sales using the Olist E-commerce dataset.

## Dataset Description

The dataset consists of various files related to orders, customers, payments, reviews, and more. Key files used in this project include:

- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_products_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_customers_dataset.csv`

## Project Steps

### Step 1: Define the Problem Statement

Predict monthly sales for the next 12 months for the Olist E-commerce platform.

### Step 2: Gather and Understand the Data

Load the necessary data files:

python
import pandas as pd

path = 'C:/Users/moham/Downloads/E-Commerce Predictive modelling/'

orders = pd.read_csv(path + 'olist_orders_dataset.csv')
order_items = pd.read_csv(path + 'olist_order_items_dataset.csv')
payments = pd.read_csv(path + 'olist_order_payments_dataset.csv')
products = pd.read_csv(path + 'olist_products_dataset.csv')
reviews = pd.read_csv(path + 'olist_order_reviews_dataset.csv')
customers = pd.read_csv(path + 'olist_customers_dataset.csv')


### Step 3: Data Preprocessing

- Handle missing values
- Convert data types
- Merge datasets to create a consolidated dataframe

python
# merging datasets
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
order_data = orders.merge(order_items, on='order_id').merge(payments, on='order_id')
```

### Step 4: Exploratory Data Analysis (EDA)

- Analyze sales trends
- Identify seasonality
- Visualize sales data

```python
import matplotlib.pyplot as plt

#  Monthly sales plot
monthly_sales = order_data.resample('M', on='order_purchase_timestamp').sum()
plt.figure(figsize=(10,6))
plt.plot(monthly_sales.index, monthly_sales['payment_value'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.show()


### Step 5: Feature Engineering

- Create features such as month, day, year, etc.
- Aggregate data to monthly level

python
order_data['order_purchase_month'] = order_data['order_purchase_timestamp'].dt.to_period('M')
monthly_sales = order_data.groupby('order_purchase_month').sum()['payment_value'].reset_index()

### Step 6: Model Building

- Split data into training and testing sets
- Build and train the model (e.g., ARIMA, Prophet)

python
from sklearn.model_selection import train_test_split

train, test = train_test_split(monthly_sales, test_size=0.2, shuffle=False)

# Example: Using ARIMA
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train['payment_value'], order=(5,1,0))
model_fit = model.fit()


### Step 7: Model Evaluation

- Evaluate model performance using metrics such as RMSE, MAE

python
from sklearn.metrics import mean_squared_error

predictions = model_fit.forecast(steps=len(test))
rmse = mean_squared_error(test['payment_value'], predictions, squared=False)
print(f'RMSE: {rmse}')


### Step 8: Forecasting Future Sales

- Use the model to predict future sales

python
forecast = model_fit.forecast(steps=12)


### Step 9: Visualization

- Visualize the actual vs predicted sales

 python
plt.figure(figsize=(10,6))
plt.plot(monthly_sales['order_purchase_month'], monthly_sales['payment_value'], label='Actual Sales', marker='o')
plt.plot(test['order_purchase_month'], predictions, label='Predicted Sales', linestyle='--', marker='x')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Monthly Sales with Data Labels')
plt.legend()

for i in range(len(monthly_sales)):
    plt.text(monthly_sales['order_purchase_month'][i], monthly_sales['payment_value'][i], str(round(monthly_sales['payment_value'][i], 2)), fontsize=8, color='blue')

for i in range(len(test)):
    plt.text(test['order_purchase_month'][i], predictions[i], str(round(predictions[i], 2)), fontsize=8, color='red')

plt.show()


## Conclusion

This project demonstrates how to use historical sales data to forecast future sales. Accurate sales forecasting helps businesses make informed decisions regarding inventory management, marketing strategies, and financial planning.

## References

- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Statsmodels Documentation](https://www.statsmodels.org/stable/index.html)
## Output ![image](https://github.com/user-attachments/assets/558fb310-ad64-48f3-ab69-2b0243019805)

