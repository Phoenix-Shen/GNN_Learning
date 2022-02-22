# %%
import yaml


class IOStream():
    def __init__(self, log_path) -> None:
        self.f = open(log_path, "a")

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def load_settings(settings_dir: str) -> dict:
    """
    从yaml中读取文件，并返回一个字典
    -----
    returns dictionary of the settings
    """
    with open(settings_dir, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    print("↓↓↓ your settings ↓↓↓\n---------------------------------")
    for key in args.keys():
        print(f"{key}:".ljust(20), str(args[key]).ljust(10))
    return args


# # %% test
# settings = load_settings(r"C:\Users\ssk\Desktop\GNN\Code\DGCNN\settings.yaml")
# # %%
