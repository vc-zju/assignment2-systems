import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("first feed forward layer output dtype: ", x.dtype)
        x = self.ln(x)
        print("layer norm output dtype: ", x.dtype)
        x = self.fc2(x)
        print("final layer output dtype: ", x.dtype)
        return x

def main():
    model = ToyModel(10, 10).cuda()
    x = torch.randn(10, 10, device="cuda")
    with torch.autocast(device_type="cuda"):
        output = model(x)
        print("first feed forward layer weight dtype: ", model.fc1.weight.dtype)
        print("second feed forward layer weight dtype: ", model.fc2.weight.dtype)
        print("layer norm weight dtype: ", model.ln.weight.dtype)
        print("layer norm bias dtype: ", model.ln.bias.dtype)
        loss = output.sum()
        print("loss dtype: ", loss.dtype)
        loss.backward()
        print("gradient dtype: ", model.fc1.weight.grad.dtype)
        print("gradient dtype: ", model.fc2.weight.grad.dtype)
        print("gradient dtype: ", model.ln.weight.grad.dtype)
        print("gradient dtype: ", model.ln.bias.grad.dtype)

if __name__ == "__main__":
    main()