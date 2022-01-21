from dataset import TextImageGenerator
from utils_fit import train_fn, eval_fn
from net import OCR
import pandas as pd
from collections import Counter
import torch, argparse, sys
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from callbacks import EarlyStopping
from torchsummary import summary



def get_letters(dt):
    CHAR_VECTOR = sorted(list(Counter(
        [name for file in dt.class_name.values for name in file.split('.')[0].strip().replace(' ', '')]).keys()))
    letters = [letter for letter in CHAR_VECTOR]
    return letters


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('images/annotations.csv')
    df['image_name'] = df['image_name'].apply(lambda f: 'images/' + f)
    letters = get_letters(df)
    num_classes = len(letters) + 1
    batch_size = 32
    num_workers = 0
    patience = 25
    max_text_len = df.len.max()
    model_path = f"snapshots/ocr.pth"


    train_data, valid_data = train_test_split(df, test_size=0.2, random_state=17) 

    print(train_data.shape, valid_data.shape)
    try:
        print(f"{num_classes=}, {patience=}, {batch_size=}, {max_text_len=}, {letters=}") # python 3.9
    except:
        print(f"num_classes={num_classes}, patience={patience}, batch_size={batch_size}, max_text_len={max_text_len}, letters={letters}")




    #model
    args = [64, "M", 128, "M", 256, "M", 512, "M", 512]
    model = OCR(features=args, in_channels=1, num_classes=num_classes, out_features=32, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.94, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)
    

    print(summary(model, (1, 64, 128)))

    
    # data
    train_generator = TextImageGenerator(data=train_data, letters=letters, max_len=max_text_len, transform=True)
    val_generator = TextImageGenerator(data=valid_data, letters=letters, max_len=max_text_len, transform=False)
    train_loader = DataLoader(dataset=train_generator, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)
    valid_loader = DataLoader(dataset=val_generator, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              drop_last=True)

    EPOCHS = 100000
    for epoch in range(1, EPOCHS):
        loss = train_fn(model, train_loader, optimizer, device, epoch, EPOCHS)
        val_loss = eval_fn(model, valid_loader, device)
        print("loss: {:.3f} - val_loss: {:.3f}".format(loss, val_loss))
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if loss <= 0 or val_loss <= 0: break
        if early_stopping.early_stop: break

