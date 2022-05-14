import copy
import os

from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader
import torchvision

from utils import get_matadata
from custom_dataset import GiMriDataset
import config


metadata_df = get_matadata(os.path.join(config.DATA_DIR, 'train'))

# Random split from metadata
metadata_df = shuffle(metadata_df, random_state=config.SEED)
idx = int(len(metadata_df) * 0.8)
train_df = metadata_df.iloc[:idx]
val_df = metadata_df.iloc[idx:]

# Create custom training and validation datasets
train_dataset = GiMriDataset(train_df,
                             csv_path=os.path.join(config.DATA_DIR, 'train.csv'),
                             transform=config.TRAINING_TRANSFORM,
                             load_labels=True)

val_dataset = GiMriDataset(val_df,
                           csv_path=os.path.join(config.DATA_DIR, 'train.csv'),
                           transform=config.TRAINING_TRANSFORM,
                           load_labels=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS)

val_loader = DataLoader(val_dataset,
                        batch_size=config.BATCH_SIZE,
                        shuffle=True,
                        num_workers=config.NUM_WORKERS)


# Model
model = torchvision.models.segmentation.fcn_resnet50(pretrained=False,
                                                     num_classes=4,
                                                     pretrained_backbone=False)

# Only one channel input
model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.to(config.DEVICE)

# Optimizer and Loss
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Weighting
# weights = [0.01, 1.0, 1.0, 1.0]
# class_weights = torch.FloatTensor(weights).cuda()
# loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)


def evaluate(model):
    running_loss = 0.
    model.eval()
    for i, (images, targets, _) in enumerate(val_loader):
        images = images.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        with torch.no_grad():
            output = model(images)['out']

            loss = loss_fn(output, targets)
            running_loss += loss

    val_loss = running_loss / (len(val_loader))
    print(f"Validation loss: {val_loss}")

    return val_loss


# 6) Training and validation loop
best_val_loss = 1000
for epoch in range(config.NUM_EPOCHS):
    # Training set
    running_loss = 0.
    model.train()
    for i, (images, targets, _) in enumerate(train_loader):
        images = images.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        output = model(images)['out']

        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

        if i % 5 == 0:
            print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}, batch {i + 1}/{len(train_loader)}:\t Loss: {loss.item():.5f}")

    print(f"Training loss: {running_loss / (len(train_loader))}")

    print("\t *** End of epoch ***")
    val_loss = evaluate(model)

    if val_loss < best_val_loss:
        best_model = copy.deepcopy(model)
        print(f"Best validation loss: {val_loss}. Saving the model...")
        best_val_loss = val_loss
