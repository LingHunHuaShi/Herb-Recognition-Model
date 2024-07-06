from torch.utils.tensorboard import SummaryWriter

import network.ResNet50
from herb_dataset import HerbDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 lr: float,
                 weight_decay: float,
                 freeze_epoch: int,
                 epochs: int,
                 batch_size: int,
                 eval_batch_size: int,
                 transform: transforms.Compose,
                 num_workers,
                 save_dir: str,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):
        super().__init__()
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

        self.epochs = epochs
        self.freeze_epoch = freeze_epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.transform = transform

        self.device = device
        self.save_dir = save_dir
        self.num_workers = num_workers
        self.logger = SummaryWriter()
        self.best_acc = 0.0

    def write(self, epoch: int, metrics: dict):
        self.logger.add_scalar('loss/train', metrics['train_loss'], epoch)
        self.logger.add_scalar('loss/val', metrics['val_loss'], epoch)
        self.logger.add_scalar('accuracy/train', metrics['train_acc'], epoch)
        self.logger.add_scalar('accuracy/val', metrics['val_acc'], epoch)

    def calculate_acc(self, outputs: torch.Tensor, targets: torch.Tensor):
        outputs, targets = outputs.detach().cpu(), targets.detach().cpu()
        pred = torch.argmax(outputs, dim=-1)
        acc = torch.sum(pred == targets).item() / len(targets)
        return acc

    def single_iteration(self, epoch, train_loader, val_loader, metric: dict):
        self.model.train()
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        if epoch + 1 < self.freeze_epoch:
            self.model.freeze()

        for (x, y) in tqdm(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)

            loss = self.criterion(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            train_acc.append(self.calculate_acc(output, y))
        self.model.eval()
        with torch.no_grad():
            for (x, y) in tqdm(val_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)
                val_loss.append(loss.item())
                val_acc.append(self.calculate_acc(output, y))

        train_loss = sum(train_loss) / len(train_loss)
        val_loss = sum(val_loss) / len(val_loss)
        train_acc = sum(train_acc) / len(train_acc)
        val_acc = sum(val_acc) / len(val_acc)
        metric['train_loss'] = train_loss
        metric['val_loss'] = val_loss
        metric['train_acc'] = train_acc
        metric['val_acc'] = val_acc

        if val_acc > self.best_acc:
            self.best_acc = val_acc
            dict = {
                'epoch': epoch + 1,
                'weight': self.model.state_dict(),
                'model': self.model,
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(dict, self.save_dir + f'/best_checkpoint.pt')

        self.write(epoch=epoch + 1, metrics=metric)
        tqdm.write(f'Epochs:[{epoch + 1}/{self.epochs}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}\n'
                   f'train_acc: {train_acc:.4f}, val_acc: {val_acc:4f}')

    def run(self):
        herb_dataset = HerbDataset("./image_list.txt", transform=self.transform)
        train_size = int(0.8 * len(herb_dataset))
        val_size = len(herb_dataset) - train_size
        train_set, val_set = random_split(herb_dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)

        metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'train_acc': 0.0,
            'val_acc': 0.0
        }
        for e in range(self.epochs):
            self.single_iteration(e, train_loader, val_loader, metrics)
            self.save_checkpoint(e)
            self.model.unfreeze()

    def save_checkpoint(self, epoch):
        dict = {
            'epoch': epoch + 1,
            'weight': self.model.state_dict(),
            'model': self.model,
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(dict, self.save_dir + f'/checkpoint_{epoch + 1}.pt')


if __name__ == '__main__':
    train_batch = 32
    val_batch = 64
    epochs = 30
    freeze_epoch = 5
    lr = 1e-5
    weight_path = None
    save_dir = './save/'

    train_transform = transforms.Compose(
        [
            transforms.Resize(236, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    trainer = Trainer(
        model=network.ResNet50.ResNet(pretrained=True, num_classes=20),
        criterion=criterion,
        optimizer=torch.optim.Adam,
        lr=lr,
        weight_decay=1e-4,
        freeze_epoch=freeze_epoch,
        epochs=epochs,
        batch_size=train_batch,
        eval_batch_size=val_batch,
        save_dir=save_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        transform=train_transform,
        num_workers=4,
    )
    trainer.run()
