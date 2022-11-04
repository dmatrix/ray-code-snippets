import torch
import numpy as np
from ray.train.torch import TorchPredictor

# List outputs are not supported by default TorchPredictor. So
# let's a custom TorchPredictor
class MyModel(torch.nn.Module):
    def forward(self, input_tensor):
        return [input_tensor, input_tensor]

# Use a custom predictor to format model output as a dict.
class CustomPredictor(TorchPredictor):
    def call_model(self, tensor):
        model_output = super().call_model(tensor)
        return {
            str(i): model_output[i] for i in range(len(model_output))
        }

if __name__ == "__main__":
    data_batch = np.array([1, 2])
    predictor = CustomPredictor(model=MyModel())
    predictions = predictor.predict(data_batch, dtype=torch.float)
    print(f"Custom results: {predictions}")
