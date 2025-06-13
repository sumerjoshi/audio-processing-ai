import torch
from torch.utils.data import Dataset, DataLoader
from model.pretrained.dual_head_cnn14 import DualHeadCnn14
from train import train_loop

class MockAudioDataset(Dataset):
    def __init__(self, num_samples=10):
        self.data = [torch.randn(32000) for _ in range(num_samples)]  # Raw audio
        self.labels = [torch.tensor(1.0 if i % 2 == 0.0 else 0.0) for i in range(num_samples)]
            
    
    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    

# tmp_path deleted after use in pytest
def test_train_model(tmp_path):
    dataset = MockAudioDataset()
    loader = DataLoader(dataset, batch_size=2)
    model = DualHeadCnn14(
        16000,
        1024,
        320,
        64,
        50,
        8000,
        527,
        False
    )
    model.fc_binary = torch.nn.Linear(2048, 1)

    save_path = tmp_path / "test_model.pth"
    train_loop(model, loader, 1, save_path, optimizer=torch.optim.Adam(model.parameters()), loss_fn=torch.nn.BCEWithLogitsLoss())
    assert save_path.exists()