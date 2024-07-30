import os
import torch
import numpy as np
from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

##################### TRAIN & VAILIDATION #####################

def training_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim, 
                loss_fn: torch.nn, scheduler: torch.optim, device: torch.cuda, log_path:str, epoch: int, image_size:int=224) -> tuple:
    
    loop = tqdm(dataloader, leave=True)

    model.train()

    total_loss = 0.0
    total_correct_predictions = 0
    total_sample = 0

    for batch_idx, (rgb_frames, kp_frames, labels) in enumerate(loop):

        rgb_frames = torch.nn.functional.interpolate(rgb_frames, size=(32, image_size, image_size), mode='trilinear', align_corners=False)
        kp_frames = torch.nn.functional.interpolate(kp_frames, size=(32, image_size, image_size), mode='trilinear', align_corners=False)

        rgb_frames = rgb_frames.to(device)  
        kp_frames = kp_frames.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(rgb_frames, kp_frames)
        probs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            scheduler.step()

        total_loss += loss.item()
        total_correct_predictions += (preds == labels).sum().item()
        loop.set_postfix(loss=loss.item(), accuracy=((preds == labels).sum().item() / rgb_frames.size(0)))
        dict_tmp = {"loss": loss.item(), "accuracy": (preds == labels).sum().item() / rgb_frames.size(0)}
        save_log(log_path, dict_tmp, epoch, len(dataloader), f"End {epoch} epoch")
        total_sample += rgb_frames.size(0)

    res_dict = {"loss": total_loss /  len(dataloader), "accuracy":total_correct_predictions /  total_sample}
    save_log(log_path, res_dict, epoch, len(dataloader), f"End {epoch} epoch")
    
    return res_dict

def validation_loop(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn, image_size: int, device: torch.cuda) -> dict:


    loop = tqdm(dataloader, leave=True)

    model.eval()

    total_loss = 0.0
    total_correct_predictions = 0
    total_sample = 0

    with torch.no_grad():
        for batch_idx, (rgb_frames, kp_frames, labels) in enumerate(loop):

            rgb_frames = torch.nn.functional.interpolate(rgb_frames, size=(32, image_size, image_size), mode='trilinear', align_corners=False)
            kp_frames = torch.nn.functional.interpolate(kp_frames, size=(32, image_size, image_size), mode='trilinear', align_corners=False)

            rgb_frames = rgb_frames.to(device)  
            kp_frames = kp_frames.to(device)
            labels = labels.to(device)
            
            outputs = model(rgb_frames, kp_frames)
            probs = torch.nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            total_correct_predictions += (preds == labels).sum().item()
            total_sample  += rgb_frames.size(0)
            loop.set_postfix(loss=loss.item(), accuracy=((preds == labels).sum().item() / rgb_frames.size(0)))
        
        res_dict = {"loss": total_loss /  len(dataloader), "accuracy":total_correct_predictions /  total_sample}
        return res_dict
            

##################### EarlyStopping #####################

class EarlyStopping:
    def __init__(self, patience:int=10, delta=0, verbose=False, path:str='checkpoint.pth'):

        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change in monitored quantity to qualify as an improvement.
            verbose (bool): If True, it prints a message for each improvement.
            path (str): Path for the checkpoint to be saved.
        """

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

##################### SAVE LOG #####################
def save_log(folder_log: str, loss_dict, epoch: int, iter: int, mess: str = None) -> None:
    log_name = "log" + str(epoch) + ".txt"
    path_log = os.path.join(folder_log, log_name)

    with open(path_log, 'a') as file:
        if mess != None:
            file.write(f"END {epoch} EPOCH")
        file.write(f"ITERATION {iter} OF EPOCH {epoch} \n")
        file.write("\n")
        file.write(str(loss_dict))