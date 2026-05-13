# 1. Introduction
The main goal of this project is to build Machine Learning models that can predict the daily closing price of gold. Since gold prices are so important in the financial market, predicting future prices helps in making better decisions. We used different regression methods to learn how gold prices moved in the past, and then we checked how well our models did using common testing methods (like 5-Fold Cross-Validation, R2 Score, and Mean Absolute Error).


# 2. Project Objectives
To make sure we cover everything required, we focused on these main steps:
* **Data Preprocessing:** We cleaned the data, removed any missing values, and split the 'Date' column into numbers so the models could understand it.
* **Feature Scaling:** We adjusted the scale of our data (like Volume and Year) so that distance-based models (like SVR) wouldn't get confused by the large numbers.
* **Training Models:** We trained three different models (Linear Regression, Random Forest, and Support Vector Regressor).
* **Fair Evaluation:** We used a standard split (70% Training and 30% Testing), plus an extra check (5-Fold Cross-Validation) to make sure our results were actually reliable.
* **Future Predictions:** We saved the very last 15 days of data completely aside, just to test our best model on real "future" data at the end.

  
# 3. Dataset Description
The dataset we worked with contains historical daily gold prices from 2004 to 2026. When we first loaded the file, the data was separated by semicolons (;) instead of normal commas, so we had to specify that in our code (using pandas) to read it correctly.

## 3.1. Features (The Inputs)
After cleaning the data, these are the columns we used to train our models:
* **Year, Month, Day:** We extracted these numbers from the original text-based 'Date' column.
* **Open:** The gold price when the market opened that day.
* **High:** The highest price of the day.
* **Low:** The lowest price of the day.
* **Volume:** The total amount of gold traded that day.

## 3.2. Target (The Output)
* **Close:** The final price of gold when the market closed. This is the exact number our models are trying to predict.

> *Note: The screenshot below shows what our data looked like after we cleaned it and extracted the date numbers.*

<img width="481" height="148" alt="image" src="https://github.com/user-attachments/assets/58309d7f-2e7a-4a68-b336-ddb0d81162e1" />

# 4. Data Preprocessing 

Before feeding the data to our models, we had to clean and prepare it. This step is super important to make the models work right and to avoid common mistakes like "data leakage" (which happens if the model accidentally peeks at the test data during training).

## 4.1. Data Cleaning and Transformation
The original dataset had a few issues that we fixed in our code:
* **Fixing the Dates:** Models only understand numbers, not text like "2023-10-05". So, we converted these text dates into actual datetime objects (using `pandas.to_datetime`), and then split them into separate `Year`, `Month`, and `Day` columns.
* **Removing Empty Rows:** We simply deleted any rows that had missing data `(NaN values)` so they wouldn't break the training process `(using dropna)`.

## 4.2. Data Splitting Strategy
We divided our data carefully to follow the project requirements:
* **Future Data Isolation:** We took the very last 15 rows of the dataset and hid them completely before doing any training or scaling. This is our "unseen" future data for the final test.
* **Train/Test Split:** The rest of the data was divided (using `train_test_split`). We used 70% of the data for training and 30% for testing.

## 4.3. Feature Scaling
* **Why we did it:** Some numbers in our dataset are huge (like `Volume`), while others are very small (like `Month`). Distance-based models (like SVR) get confused by this, so we needed to put all features on a similar scale (using `StandardScaler`).
* **Preventing Data Leakage:** To be completely fair, we only "fitted" the scaler on the training data `(X_train)`. Then, we applied that exact same scale to the testing and future data `(using transform)`. Note that we didn't scale the target variable `('Close' Price)` so our final predictions stay in real dollars.

> *Below is a snapshot from our code showing how we handled the scaling to prevent data leakage:*
<img width="558" height="273" alt="image" src="https://github.com/user-attachments/assets/22b97206-cad5-4dbe-a4a2-c946bb224293" />


# 5. Model Selection
We chose three different models to see which one works best for our data:

* **Linear Regression:** This is our basic starting model. It simply assumes a straight-line relationship between our inputs (like Open, High, and Low) and the final output (the Close price).
* **Random Forest Regressor:** This model uses 100 decision trees working together (using `n_estimators=100`). It is really good at finding complex patterns in the data without needing much manual adjustment.
* **Support Vector Regressor (SVR):** We used this model with its default settings (using `kernel='rbf'`). It is a powerful algorithm, but as we noticed during the project, it only performs well after we properly scale the features (using `StandardScaler`).
  
  
# 6. Model Evaluation and Results

To truly understand how good our models are, we didn't just look at one simple test score. We used two methods to make sure our evaluation was fair and reliable:

* **5-Fold Cross-Validation:** We ran this on our 70% training data. It basically chops the data into 5 pieces, uses 4 pieces to train the model, and tests it on the 1 piece left out. It repeats this 5 times. This is a great way to prove that our models are consistently good and didn't just get a "lucky" easy part of the data.
* **Test Set Evaluation:** Finally, we tested the models on the 30% of data they had never seen before. We looked at the **R2 Score** to see the overall accuracy (a score close to 1.0 is perfect) and the **Mean Absolute Error (MAE)** to see exactly how many dollars we were off by on average.

## 6.1. Performance Comparison

Here is what we noticed after running all three models:

* **Linear Regression:** This model did an amazing job, with an R2 score very close to 1.0. Honestly, this makes sense. Since we are trying to guess the 'Close' price, and we already know the 'Open', 'High', and 'Low' prices for that same day, there is a very obvious straight-line relationship that this model picked up easily.
* **Random Forest:** This model also gave us very high accuracy. It’s a strong algorithm that figured out the patterns in the data right away without us needing to tweak its settings too much.
* **Support Vector Regressor (SVR):** This one was interesting. Before we did the "Feature Scaling" step, it performed terribly. But after we scaled the data, it became much better and competitive. This proved how sensitive SVR is to large numbers like our 'Volume' column.

## 6.2. Visualizing the Results

**1. Accuracy and Error Comparison**
The bar charts below compare our three models side-by-side. You can clearly see that Linear Regression and Random Forest are the winners here, as they have the highest R2 scores and the lowest error (MAE), making them very reliable for this specific dataset.

<img width="956" height="483" alt="Accuracy and Error Charts" src="https://github.com/user-attachments/assets/605e63cb-e77d-4802-81ea-57bfb9d87e2e" />


**2. Actual vs. Predicted Prices (100-Day Sample)**
If we tried to plot all the thousands of test days on one chart, it would just look like a messy, unreadable blob. So, we decided to plot just a 100-day sample from the test set. As you can see in the chart below, the red dashed line (our prediction) perfectly hugs the blue solid line (the actual price). This shows that our model successfully learned to follow the daily ups and downs of the gold market.

<img width="837" height="433" alt="Actual vs Predicted Chart" src="https://github.com/user-attachments/assets/db71c49a-eb22-4a4a-8f5f-d572ec9265df" />


# 7. Future Predictions (The 15-Day Challenge)

Just like the project instructions required, we kept the very last 15 rows of our dataset completely hidden from the models during the whole training and testing phases. Once we figured out which model performed the best, we put it to the ultimate test: guessing the gold prices for these 15 purely "future" days.

The results were actually really impressive. This test proved that our model can handle completely new, unseen data and still predict the closing price accurately. Below is a comparison showing what the model predicted versus the actual market prices for those 15 days:

<img width="447" height="403" alt="image" src="https://github.com/user-attachments/assets/5e2f1fce-2bb8-472e-bdd7-d5b73b2ab097" />


> *(Note: As you can see, the model successfully kept a very small error margin across these future records. This proves it actually learned the market trends instead of just memorizing the training data).*

# 8. Conclusion

In the end, this project was a great way to see how Machine Learning can actually be used to predict real things in the financial market. By taking the time to properly clean the raw data, carefully using feature scaling to avoid data leakage, and testing out three different algorithms (Linear Regression, Random Forest, and SVR), we managed to build a really solid prediction pipeline. 

It showed us that with the right data preparation and model selection, predicting complex things like gold prices with high accuracy is completely possible.

It showed us that with the right data preparation and model selection, predicting complex things like gold prices with high accuracy is completely possible.

The Linear Regression and Random Forest models proved to be exceptionally accurate for this specific dataset structure. Furthermore, the project met all rigorous academic requirements, from K-Fold cross-validation to the strict isolation of future data, proving the practical and analytical viability of these Machine Learning techniques.
