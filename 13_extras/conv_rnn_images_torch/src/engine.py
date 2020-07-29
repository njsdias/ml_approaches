from tqdm import tqdm
import torch
import config


def train_fn(model, data_loader, optimizer):

    # put the model in train model
    model.train()
    fin_loss = 0
    tk = tqdm(data_loader, total=len(data_loader))
    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)

        # for every data we need to zero grad
        optimizer.zero_grad()

        # we are not interested in predictions. Only in loss
        _, loss = model(**data)

        # to make possible model learn use backwards
        # backwards calculate the gradients
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

    return fin_loss / len(data_loader)


def eval_fn(model, data_loader):

    # put the model in train model
    model.eval()
    fin_loss = 0
    fin_preds = []
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader))
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            batch_preds, loss = model(**data)
            fin_loss += loss.item()

            fin_preds.append(batch_preds)

        return fin_preds, fin_loss / len(data_loader)
