import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4x4
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x).squeeze()

def run_classical():
    data = torch.load('shared_digits_data.pt', weights_only=False)
    tx, ty = data['train_x'], torch.tensor(data['train_y'], dtype=torch.float32)
    
    model = SimpleCNN()
    print(f"CNN参数量: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练
    for _ in range(80):
        opt.zero_grad()
        input_x = torch.tensor(tx.reshape(-1, 1, 8, 8) / 16.0, dtype=torch.float32)
        preds = model(input_x)
        loss = nn.BCEWithLogitsLoss()(preds, ty)
        loss.backward(); opt.step()

    # 测试
    angles, accs = [], []
    for ang, (test_x, test_y) in data['test_dict'].items():
        with torch.no_grad():
            input_x = torch.tensor(test_x.reshape(-1, 1, 8, 8) / 16.0, dtype=torch.float32)
            preds = torch.sigmoid(model(input_x))
            correct = ((preds > 0.5) == (torch.tensor(test_y) == 1)).float().mean()
            accs.append(correct.item() * 100)
            angles.append(ang)

    plt.plot(angles, accs, 'o--', color='gray', label='Standard CNN (~3000 params)')
    plt.ylim(0, 105); plt.ylabel('Accuracy (%)'); plt.xlabel('Angle'); plt.legend(); plt.grid()
    plt.title("Classical CNN Generalization Result")
    plt.savefig('classical_accuracy.png')
    plt.show()

if __name__ == "__main__":
    run_classical()