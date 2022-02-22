
import numpy as np
import torch as t
import os
from torch.utils.data import DataLoader
from models import DGCNN
from data.dataloader import ModelNet40
from utils import *
import sklearn.metrics as metrics









def test(args:dict)->None:
    test_set = ModelNet40(args["dataset_dir"],
                          args["num_points"], partition="test")
    test_loader = DataLoader(
        test_set, batch_size=args["batch_size"], shuffle=True, drop_last=False)

    device = t.device("cuda" if args["cuda"] else "cpu")

    model = DGCNN(args, 40).to(device)

    if not os.path.exists(args["model_path"]):
        raise OSError("路径不存在，请检查")
    model.load_state_dict(t.load(args["model_path"]))
    model = model.eval()
    test_acc = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (
        test_acc, avg_per_class_acc)
    print(outstr)


if __name__ == "__main__":
    args = load_settings(r"C:\Users\ssk\Desktop\GNN\Code\DGCNN\settings.yaml")
    test(args)
