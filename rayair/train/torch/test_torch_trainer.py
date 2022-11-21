import torch
import torch.nn as nn

import ray
from ray import train
from ray.air import session, Checkpoint
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig

# Define NN layers archicture, epochs, and number of workers
input_size = 1
layer_size = 32
output_size = 1
num_epochs = 180
num_workers = 3

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, layer_size)
        self.layer2 = nn.Linear(layer_size, output_size)
        
    def forward(self, input):
        return self.layer2(self.layer1(input))

def train_loop_per_worker():
    dataset_shard = session.get_dataset_shard("train")
    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    model = train.torch.prepare_model(model)

    # Iterate over epochs and batches
    for epoch in range(num_epochs):
        for batches in dataset_shard.iter_torch_batches(
            batch_size=32, dtypes=torch.float):

            # Add batch as additional dimension [32, x]
            inputs, labels = torch.unsqueeze(batches["x"], 1), batches["y"]
            output = model(inputs)

            # Get outputs as the same dimension as labels
            loss = loss_fn(output.squeeze(), labels)
            
            # Zero out grads, do backward, and update optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print what's happening with loss per 30 epochs
            if epoch % 30 == 0:
                print(f"epoch: {epoch}/{num_epochs}, loss: {loss:.3f}")

        # Report and record metrics, checkpoint model at end of each 
        # epoch
        session.report({"loss": loss.item(), "epoch": epoch}, checkpoint=Checkpoint.from_dict(
                            dict(epoch=epoch, model=model.state_dict()))
        )

torch.manual_seed(42)
train_dataset = ray.data.from_items(
        [{"x": x, "y": 2 * x + 1} for x in range(2000)]
)

# Define scaling and run configs
# If using GPUs, use the below scaling config instead.
# scaling_config = ScalingConfig(num_workers=3, use_gpu=True)
scaling_config = ScalingConfig(num_workers=num_workers)
run_config = RunConfig(checkpoint_config=CheckpointConfig(num_to_keep=1))

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    scaling_config=scaling_config,
    run_config=run_config,
    datasets={"train": train_dataset})

result = trainer.fit()
best_checkpoint_loss = result.metrics['loss']
# print(f"best loss: {best_checkpoint_loss:.4f}")

# Assert loss is less 0.09
assert best_checkpoint_loss <= 0.09