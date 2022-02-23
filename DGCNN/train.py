import numpy as np
import torch as t
from torch.utils.data import DataLoader
from models import DGCNN, calculate_loss
from data.dataloader import ModelNet40
from utils import *
from torch import optim
import sklearn.metrics as metrics
import datetime


def train(args: dict) -> None:

    #################################
    # train/test set and dataloader #
    #################################

    train_set = ModelNet40(args["dataset_dir"],
                           args["num_points"], partition="train")
    test_set = ModelNet40(args["dataset_dir"],
                          args["num_points"], partition="test")
    train_loader = DataLoader(
        train_set, batch_size=args["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=args["batch_size"], shuffle=True, drop_last=False)

    ###############################
    # device, model and optimizer #
    ###############################

    device = t.device("cuda" if args["cuda"] else "cpu")

    model = DGCNN(args, 40).to(device)

    if args["use_sgd"]:
        optimizer = optim.SGD(model.parameters(
        ), lr=args["lr"]*100, momentum=args["momentum"], weight_decay=1e-4)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args["lr"], weight_decay=1e-4)

    if args["scheduler"] == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args["epochs"], eta_min=1e-3)
    elif args["scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.7)
    else:
        raise NotImplementedError("暂时没有提供cos和step以外的选项")

    if args["model_dir"] is not None:
        file = t.load(args["model_dir"])
        model.load_state_dict(file["model"])
        optimizer.load_state_dict(file["optimizer"])

    set_seed(args["seed"])
    ####################
    # training process #
    ####################

    best_test_acc = 0

    for epoch in range(args["epochs"]):
        train_loss = 0.
        count = 0.
        # call model.train because of the dropuout layer
        model.train()

        train_pred = []
        train_true = []

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            optimizer.zero_grad()
            logits = model.forward(data)

            loss = calculate_loss(logits, label)
            loss.backward()
            optimizer.step()

            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item()*batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

        if args["scheduler"] == "cos":
            scheduler.step()
        else:
            if optimizer.param_groups[0]["lr"] > 1e-5:
                scheduler.step()
            if optimizer.param_groups[0]['lr'] < 1e-5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        print(outstr)
        ##################
        # Test Procedure #
        ##################
        test_loss = 0.
        count = 0.
        # call model.eval
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model.forward(data)
            loss = calculate_loss(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item()*batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(
            test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        print(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            t.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                   f"./DGCNN/saved_models/model-{time}.pth")


if __name__ == "__main__":
    args = load_settings(r"C:\Users\ssk\Desktop\GNN\Code\DGCNN\settings.yaml")
    train(args)
