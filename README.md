# Predicting stock behaviour

This project is a small mix of a hobby project and a school project. The goal is to predict the behaviour of a stock in the next hour.

## Introduction
In this project we model the problem of predicting stock price fluctuations by simplifying the problem to classification problem. It is enough to know whether the price of a stock goes up or down in the next hour to be able to **optimally exploit** the stock for profit. Instead of binary classification, we opt to make the problem a multi-class classification problem by dividing the price change into 3 classes: **up**, **down** and **no change**. This is done to hopefully reduce the noise in the data and make the problem easier to solve.

If the price of a stock changes less than 0.2% in the next hour, we consider that the stock price has not changed. We chose 0.2%, because it balances the class labels.

## Data
As data we use data pulled from the Yahoo Finance API for Helsinki Stock Exchange (OMX Helsinki). We only take certain stocks filling our quality criteria, which leads to a total of 14 stocks. As input for our model, we give the model a timeseries of M last price changes of each stock, and we predict the class of the next price change for each stock, so a total of 14*3 outputs. Since we difference the data for the model we guide the model to keep track of the price changes, and we squeeze the data into a smaller range which should improve convergence.

We chose the following stocks, because they had no missing values, splits, and their average price was between 20 - 100 euros in the last 5 years:

- Revenio Group
- Neste
- Orion Class B
- Kone
- Olvi
- Huhtamäki
- Detection Technology
- Orion Class A
- Cargotec Corp
- Vaisala
- Ålandsbanken Class B
- Valmet
- eQ
- Ponsse


## Model
As models, we try RNN, LSTM and Transformer models. For each model type we perform hyperparameter tuning using the Hyperband algorithm in KerasTuner library. 