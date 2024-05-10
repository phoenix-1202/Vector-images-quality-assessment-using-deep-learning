import torch
import time
import copy
import cv2
import os
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from numpy import double
import albumentations as albu
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2

BATCH_SIZE = 64

augs = albu.Compose([
    albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225], ),
    ToTensorV2(),
])


class QualityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, folder: str, train: bool = True, transforms=None):
        self.df = df
        self.folder = folder
        self.train = train
        self.transforms = transforms

    def __getitem__(self, index):
        im_path = os.path.join(self.folder, self.df.iloc[index]['image_name'])
        x = cv2.imread(im_path, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        if self.transforms:
            x = self.transforms(image=x)['image']

        if self.train:
            y = self.df.iloc[index]['label']
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.df)


class VitPredictor:
    def __init__(self, num_features: int, num_epochs: int):
        self.model = vit_b_16(pretrained=True)
        self.num_features = num_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=1e-4,
                                           weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=2,
                                                         gamma=0.1)
        self.criterion = nn.BCELoss()

    def save_model(self):
        """
        Function to save the trained model to disk.
        """
        torch.save({
            'epoch': self.num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }, 'outputs_vit/model.pth')

    def train(self, datasets: dict[str, QualityDataset], dataloaders: dict[str, DataLoader], device: torch.device):
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0.0

                train_bar = tqdm(dataloaders[phase], desc=f"Training")
                for _, (inputs, labels) in enumerate(train_bar):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero out the grads
                    self.optimizer.zero_grad()

                    # Forward
                    # Track history in train mode
                    with torch.set_grad_enabled(phase == 'train'):
                        self.model = self.model.to(device)
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(datasets[phase])
                epoch_acc = double(running_corrects) / len(datasets[phase])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        self.model.load_state_dict(best_model_wts)
        return model


if __name__ == '__main__':
    train_df = None
    train, valid = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42,
        stratify=train_df.label.values
    )

    train_dataset = QualityDataset(
        df=train,
        folder="",  # папка к файлам трейна
        train=True,
        transforms=augs
    )

    valid_dataset = QualityDataset(
        df=valid,
        folder="",  # папка к файлам валидации
        train=False,
        transforms=augs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
    )

    datasets = {'train': train_dataset, 'valid': valid_dataset}
    dataloaders = {'train': train_loader, 'valid': valid_loader}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VitPredictor(num_features=5, num_epochs=5)
    model.train(datasets, dataloaders, device)
