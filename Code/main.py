import preprocessing
import models as md
import evaluation as evo

file_path = '../data/goldPriceData1Day.csv'
raw_data = preprocessing.loadAndCleanData(file_path)

model_data, future_data = preprocessing.splitFutureData(raw_data)

X_train_raw, X_test_raw, y_train, y_test = md.splitData(model_data)

X_future_raw = future_data.drop(columns=['Close'])
y_future_actual = future_data['Close']

X_train, X_test, X_future, scaler = preprocessing.applyScaling(X_train_raw, X_test_raw, X_future_raw)


models = md.getRegressionModels()


print("Processing Models... Please wait.")
evaluation_results = evo.evaluateModels(models, X_train, y_train, X_test, y_test)

print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("="*50)

for model_name in evaluation_results:
    metrics = evaluation_results[model_name]
    print(f"\nModel: {model_name}")
    print(f"  K-Fold R2 Score (Mean): {metrics['K-Fold R2 Mean']:.4f}")
    print(f"  Test Set R2 Score:      {metrics['Test R2 Score']:.4f}")
    print(f"  Test Set MAE ($):       {metrics['Test MAE']:.2f}")

evo.plotResults(evaluation_results)

chosen_model = models['Linear Regression']
evo.plotPredictionSample(chosen_model, X_test, y_test)

print("\n" + "="*50)
print("FUTURE PREDICTION (LAST 15 RECORDS)")
print("="*50)
future_preds = chosen_model.predict(X_future)
for i in range(len(future_preds)):
    print(f"Day {i+1}: Predicted=${future_preds[i]:.2f} | Actual=${y_future_actual.values[i]:.2f}")



print(raw_data.head())

