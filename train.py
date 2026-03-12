import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('..')

import config
from model import MobileNetV2
from dataset import get_dataloaders

def train():
    print('Training is started')
    config.Config.print_gpu_info()
    config.Config.setup_dirs()

    # Check gpu's memory
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f'Have {total_mem:.2f} GB of memory')
        if total_mem < 8:
            print('Change your batch_size to 1-2 or change image size to 128')

    train_loader, val_loader = get_dataloaders(
        config.Config.INPUT_DIR,
        config.Config.GT_DIR,
        config.Config.VAL_INPUT_PATH,
        config.Config.VAL_GT_PATH,
        config.Config.DEFECTS_NAME,
        batch_size=config.Config.BATCH_SIZE,
        img_size=config.Config.IMG_SIZE,
        num_workers=config.Config.NUM_WORKERS,
        pin_memory=config.Config.PIN_MEMORY
    )

    #model
    model = MobileNetV2(
        in_channel=config.Config.IN_CHANNEL,
        out_channel=config.Config.OUT_CHANNEL,
        pretrained=True
    ).to(config.Config.DEVICE)

    #Loss and optimizer
    crit = nn.L1Loss()
    optimizer = Adam(model.parameters(), lr=config.Config.LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    #Mixed Precision Scaler
    scaler = GradScaler() if config.Config.USE_MIXED_PRECISION else None

    # Best model
    best_loss = float('inf')

    print(f'\n Train batches: {len(train_loader)}')
    print(f'\n Val batches: {len(val_loader)}')
    print(f'\n Epochs: {config.Config.NUM_EPOCHS_FREEZE} (freeze) {config.Config.NUM_EPOCHS_UNFREEZE} (fine-tuning)')
    print(f'\n Batch size: {config.Config.BATCH_SIZE}')
    print(f'\n Mixed Precision: {config.Config.USE_MIXED_PRECISION}\n')

    print('Training decoder')
    for epoch in range(config.Config.NUM_EPOCHS_FREEZE):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.Config.NUM_EPOCHS_FREEZE}')
        for inputs, targets in pbar:
            inputs = inputs.to(config.Config.DEVICE, non_blocking=True)
            targets = targets.to(config.Config.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            #Mixed Precision
            if config.Config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(inputs)
                    loss = crit(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = crit(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        val_loss = validate(model, val_loader, crit, config.Config.DEVICE,
                                config.Config.USE_MIXED_PRECISION)
        scheduler.step(val_loss)

        print(f'\n Epoch {epoch+1}: Train loss {train_loss:.4f}, Validation loss {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                               config.Config.CHECKPOINT_DIR / 'best_decoder.pth')
            print('Model has been saved')

    print('\n Fine-tuning is started')
    model.unfreeze_encoder()

    optimizer = Adam(model.parameters(), lr=config.Config.LR_UF)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = float('inf')

    for epoch in range(config.Config.NUM_EPOCHS_UNFREEZE):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.Config.NUM_EPOCHS_UNFREEZE}')
        for inputs, targets in pbar:
            inputs = inputs.to(config.Config.DEVICE, non_blocking=True)
            targets = targets.to(config.Config.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            #Mixed Precision
            if config.Config.USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(inputs)
                    loss = crit(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = crit(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)

        val_loss = validate(model, val_loader, crit,
                            config.Config.DEVICE,
                            config.Config.USE_MIXED_PRECISION)

        scheduler.step(val_loss)

        print(f'\n Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                            config.Config.CHECKPOINT_DIR / 'best_full_model.pth')
            print('MOdel saved')
        print(f'\n Training is ended! Best val loss: {best_loss:.4f}')
    return model

def validate(model, val_loader, criterion, device, use_mixed_precision=False):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_mixed_precision:
                with autocast:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    return val_loss

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f'Checkpoints is saved to {path}')

if __name__ == '__main__':
    train()