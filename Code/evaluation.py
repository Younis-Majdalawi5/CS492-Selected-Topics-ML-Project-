from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def evaluateModels(models, X_train, y_train, X_test, y_test):
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()

        predictions = model.predict(X_test)

        test_r2 = r2_score(y_test, predictions)
        test_mae = mean_absolute_error(y_test, predictions)

        results[name] = {
            'K-Fold R2 Mean': cv_mean,
            'Test R2 Score': test_r2,
            'Test MAE': test_mae
        }

    return results


def plotResults(evaluation_results):
    models = []
    r2_scores = []
    mae_values = []

    for name, metrics in evaluation_results.items():
        models.append(name)
        r2_scores.append(metrics['Test R2 Score'])
        mae_values.append(metrics['Test MAE'])

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(models, r2_scores, color=['#2c3e50', '#27ae60', '#e74c3c'])
    plt.title('Accuracy Comparison (R2 Score)')
    plt.ylabel('Score (0 to 1)')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=15)

    plt.subplot(1, 2, 2)
    plt.bar(models, mae_values, color=['#3498db', '#9b59b6', '#f39c12'])
    plt.title('Error Comparison (MAE)')
    plt.ylabel('Error Value ($)')
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()


def plotPredictionSample(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_test_values = y_test.values

    plt.figure(figsize=(14, 7))

    plt.plot(y_test_values[0:100], label='Actual Gold Price', color='#2980b9', linewidth=2, alpha=0.8)

    plt.plot(y_pred[0:100], label='Predicted Gold Price', color='#c0392b', linestyle='--', linewidth=1.5)

    min_price = min(y_test_values[0:100].min(), y_pred[0:100].min())
    max_price = max(y_test_values[0:100].max(), y_pred[0:100].max())
    plt.ylim(min_price * 0.9, max_price * 1.1)

    plt.title('Gold Price Prediction: Actual vs Model (100-Day Sample)', fontsize=14)
    plt.xlabel('Timeline (Days)', fontsize=12)
    plt.ylabel('Price in USD ($)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.show()