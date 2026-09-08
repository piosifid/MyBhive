import os
import yaml


class ConfigLoader(object):
    @staticmethod
    def load_config(config_name):
        with open("{}/config/default.yml".format(os.getenv("B_HIVE_DIR")), mode="r+") as yml:
            default = yaml.safe_load(yml)

        path = "{}/config/{}.yml".format(os.getenv("B_HIVE_DIR"), config_name)
        if not (os.path.exists(path)):
            print(f"{path} does not exist")
            print("Falling back to default config!")
            return default
        else:
            with open(path, mode="r+") as yml:
                content = yaml.safe_load(yml)
                if content == None:
                    raise ValueError(
                        "Your config is empty. Please fill in infos or use another one!"
                    )
            # add all items that are in default to config
            return dict(list(default.items()) + list(content.items()))
