import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.01, path='checkpoint.pt'):
        """
        Args:
            patience (int): Cuántas épocas esperar después de la última mejora.
            verbose (bool): Si True, imprime un mensaje cuando se produce una mejora.
            delta (float): Mínima mejora significativa que debe haber entre las épocas.
            path (str): Donde guardar el mejor modelo.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.delta:
            print(f'EarlyStopping counter: validation loss is under the value of delta ({self.delta}).')
            self.save_checkpoint(val_loss, model)
            self.early_stop = True
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Guarda el modelo cuando la pérdida de validación disminuye.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss