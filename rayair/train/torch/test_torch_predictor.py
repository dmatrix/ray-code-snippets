import numpy as np
import pandas as pd
import torch
from ray.train.torch import TorchPredictor
import pandas as pd
import torch
from ray.train.torch import TorchPredictor

class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 1)

    def forward(self, input_dict: dict):
        out1 = self.linear1(input_dict["A"].unsqueeze(1))
        out2 = self.linear2(input_dict["B"].unsqueeze(1))
        return out1 + out2


if __name__ == "__main__":
    
    # set manul seed so we get consistent output
    torch.manual_seed(42)
    # Use Standard PyTorch model
    model = torch.nn.Linear(2, 1)
    predictor = TorchPredictor(model=model)

    data = np.array([[1, 2], [3, 4]])
    predictions = predictor.predict(data, dtype=torch.float)
    print(f"Standard model predictions: {predictions}")

    print("---")
	
    # Use Custom PyTorch model 
    predictor = TorchPredictor(model=CustomModule())
	# Pandas dataframe.
    data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
    predictions = predictor.predict(data, dtype=torch.float)
    print(f"Custom model predictions: {predictions}")
