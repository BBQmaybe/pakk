import pandas as pd
import lightning as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F
import torchmetrics
import timm
import toml


class ModelUtils:
    @staticmethod
    def compute_class_weights(primary_labels):
        unique_labels = primary_labels.unique()
        total_samples = primary_labels.shape[0]
        class_weights = {}
        for label in unique_labels:
            count = (primary_labels == label).sum() 
            class_weight = count / total_samples
            class_weights[label] = class_weight
        return torch.tensor(list(class_weights.values())).pow(-0.5)
    
class BirdCLEFModel(L.LightningModule):
    def __init__(self, current_config):
        super(BirdCLEFModel, self).__init__()
        self.current_config = current_config
        self.model = self._create_model()
        self.metadata = pd.read_csv(self.current_config['meta_parameters']['metadata'])
        self.weights = ModelUtils.compute_class_weights(self.metadata['primary_label'])
        self.criterion = nn.CrossEntropyLoss(self.weights)

        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.current_config['model_parameters']['num_classes'], average='macro')
        self.precision = torchmetrics.Precision(task='multiclass', num_classes=self.current_config['model_parameters']['num_classes'], average='macro')
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=self.current_config['model_parameters']['num_classes'], average='macro')

        torch.set_float32_matmul_precision('high')
        self.save_hyperparameters()   

    def _create_model(self):
        model = timm.create_model(self.current_config['model_parameters']['model_name'], 
                                  pretrained=True, 
                                  num_classes=self.current_config['model_parameters']['num_classes'])
        return model 

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage: str):
        x, y = batch
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)

        predict = self.model(x)
        loss = self.criterion(predict, y)

        predicted_classes = torch.argmax(F.softmax(predict, dim=1), dim=1)
        ground_truth = torch.argmax(y, dim=1)

        f1 = self.f1(predicted_classes, ground_truth)
   
        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_f1', f1, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def test_step(self, batch, batch_idx):
        self.step(batch, 'test')

    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val')

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.current_config['model_parameters']['learning_rate'], weight_decay=self.current_config['optimizer_parameters']['weight_decay'])
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.current_config['model_parameters']['epochs'], eta_min=self.current_config['optimizer_parameters']["eta_min"], last_epoch=-1)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch', 
                'monitor': 'val_loss', 
                'frequency': 1
            }
        }