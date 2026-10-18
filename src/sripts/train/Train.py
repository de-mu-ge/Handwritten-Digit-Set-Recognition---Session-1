# from src.sripts.dataloader.Data_loader import train_dataloader
from src.sripts import Model2828
from src.sripts import Config

import torch
import torch.nn as nn

model = Model2828()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def train()->None:

    from src.sripts.dataloader.Data_loader import train_dataloader
    print("train data load end")

    for epoch in range(Config.epochs):
        index = 0
        loss_item =0

        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            index += 1
            loss_item += loss.item()

        print(f"epoch{epoch + 1}")
        print(f"loss: {loss_item / index}")


    torch.save(model.state_dict(), Config.model_save_path)


def exam()->None:

    # Exam Dataset:
    index = 0
    true = 0

    from src.sripts.dataloader.Data_loader import exam_dataloader, train_dataloader
    print("exam data load end")

    mod = Model2828()
    mod.load_state_dict(torch.load(Config.model_save_path))
    mod.eval()

    for images, labels in exam_dataloader:

        index += 1

        outputs = mod(images)
        output = torch.max(outputs, 1)[1]

        if output == labels[0]:
            true += 1

    print("exam dataset")
    print(f"true_rate: {true / index}")



    # model = Model2828()
    # model.load_state_dict(torch.load(Config.model_save_path))
    # model.eval()

    # Train Dataset:
    index = 0
    true = 0

    for images, labels in train_dataloader:

        index += 1

        outputs = mod(images)
        output = torch.max(outputs, 1)[1]

        # print(type(output))
        # print(type(labels))
        #
        # print(output)
        # print(labels)

        if output == labels[0]:
            true += 1

    print("train dataset")
    print(f"true_rate: {true / index}")



