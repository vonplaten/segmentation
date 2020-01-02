import sys
import getopt

import src.util.config

import importlib


class Main:
    def __init__(self, action):
        config = src.util.config.Config()

        # Dataloaderfactory
        packagename = f"{config.dataloader_name}".split(".")[0]
        classname = f"{config.dataloader_name}".split(".")[1]
        dataloader_class_ = getattr(importlib.import_module(f"src.dataloaders.{packagename}"), classname)
        dataloaderfactory = dataloader_class_()

        # Model
        packagename = f"{config.model_name}".split(".")[0]
        classname = f"{config.model_name}".split(".")[1]
        model_class_ = getattr(importlib.import_module(f"src.models.{packagename}"), classname)
        model = model_class_(config.model_nclass)

        if action == "train":
            packagename = f"{config.trainer_name}".split(".")[0]
            classname = f"{config.trainer_name}".split(".")[1]
            trainer_class_ = getattr(importlib.import_module(f"src.trainers.{packagename}"), classname)
            self.trainer = trainer_class_(dataloaderfactory, model, config.model_statepath)
            self.trainer.run()

        if action == "inf":
            packagename = f"{config.inferencer_name}".split(".")[0]
            classname = f"{config.inferencer_name}".split(".")[1]
            inferencer_class_ = getattr(importlib.import_module(f"src.inferencers.{packagename}"), classname)
            self.inferencer = inferencer_class_(dataloaderfactory, model, config.model_statepath)
            self.inferencer.run()


# example notebook: %run main -a train
if __name__ == "__main__":
    action = ''
    argv = sys.argv[1:]
    try:
        opts, _ = getopt.getopt(argv, "a:")
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-a':
            action = arg

    main = Main(action)
    del action
    del argv
    del opts
    del opt
    del arg
