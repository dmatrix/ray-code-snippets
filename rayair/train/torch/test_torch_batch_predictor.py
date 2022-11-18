import pandas as pd
import ray
from ray.train.batch_predictor import BatchPredictor

def calculate_accuracy(df):
    return pd.DataFrame({"correct": df["preds"] == df["label"]})

# Create a batch predictor that returns identity as the predictions.
batch_pred = BatchPredictor.from_pandas_udf(
    lambda data: pd.DataFrame({"preds": data["feature_1"]}))
    
 # Create a dummy dataset.
ds = ray.data.from_pandas(pd.DataFrame({
    "feature_1": [1, 2, 3], "label": [1, 2, 3]}))

# Execute batch prediction using this predictor.
predictions = batch_pred.predict(ds,
    feature_columns=["feature_1"], keep_columns=["label"])
                
# print predictions and calculate final accuracy
print(predictions)
correct = predictions.map_batches(calculate_accuracy)
print(f"Final accuracy: {correct.sum(on='correct') / correct.count()}")