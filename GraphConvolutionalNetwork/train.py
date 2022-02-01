import yaml
from models import GCN
from utils import *
from dataloader import load_data
import torch.optim as optim
import time
import torch.nn.functional as F
import tensorboardX
if __name__ == "__main__":
    # load settings and print them
    with open("GraphConvolutionalNetwork\settings.yaml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("↓↓↓ your settings ↓↓↓\n---------------------------------")
    for key in args.keys():
        print(f"|{key}:".ljust(20), str(args[key]).ljust(10), "|")

    print("---------------------------------\nTraining...")
    # summary writer
    writer = tensorboardX.SummaryWriter(args["log_dir"])
    # load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    # model and optimizer
    model = GCN(n_features=features.shape[1],
                n_hiddens=args["n_hiddens"],
                n_class=labels.max().item()+1,
                dropout_rate=args["dropout_rate"])

    optimizer = optim.Adam(
        model.parameters(),
        lr=args["lr"],
        weight_decay=args["weight_decay"])
    # to cuda
    if args["cuda"]:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    t_total = time.time()

    for epoch in range(args["epochs"]):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()
        output = model.forward(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = calculate_accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args["fastmode"]:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model.forward(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = calculate_accuracy(output[idx_val], labels[idx_val])

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - start_time))
        writer.add_scalar("train_loss", loss_train.item(), epoch)
        writer.add_scalar("val_loss", loss_val.item(), epoch)
        writer.add_scalar("train_acc", acc_train.item(), epoch)
        writer.add_scalar("val_acc", acc_val.item(), epoch)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # test
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = calculate_accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
