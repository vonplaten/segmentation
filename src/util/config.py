import os
import yaml
import datetime
import ast


class Config():
    cfg = None

    # Trainer
    CATEGORY_TRAINER = "trainer"
    TRAINER_NAME = "name"
    TRAINER_NUMEPOCHS = "num_epochs"
    TRAINER_LEARNINGRATE = "lr"

    # Input
    CATEGORY_INPUT = "input"
    INPUT_NAME = "name"

    INPUT_TRAINSIZE = "train_size"
    INPUT_VALSIZE = "val_size"
    INPUT_TESTSIZE = "test_size"

    INPUT_TRAINBATCHSIZE = "train_batch_size"
    INPUT_VALBATCHSIZE = "val_batch_size"
    INPUT_TESTBATCHSIZE = "test_batch_size"

    # Model
    CATEGORY_MODEL = "model"
    MODEL_NAME = "name"
    MODEL_NCLASS = "n_class"
    MODEL_STATEPATH = "state"

    # Inferencer
    CATEGORY_INFERENCER = "inferencer"
    INFERENCER_NAME = "name"
    INFERENCER_PRINT_NUM = "print_inf_num"

    # Output
    CATEGORY_OUTPUT = "output"
    OUTPUT_PRINTFOLDER = "print_folder"
    OUTPUT_MODELFOLDER = "model_folder"

    def __init__(self):
        if Config.cfg is None:
            with open("config.yaml") as f:
                Config.cfg = yaml.load(f, Loader=yaml.BaseLoader)

    # Input
    @property
    def dataloader_name(self):
        return Config.cfg[Config.CATEGORY_INPUT][Config.INPUT_NAME]

    @property
    def dataloader_trainsize(self):
        return int(Config.cfg[Config.CATEGORY_INPUT][Config.INPUT_TRAINSIZE])

    @property
    def dataloader_trainbatchsize(self):
        return int(Config.cfg[Config.CATEGORY_INPUT][Config.INPUT_TRAINBATCHSIZE])

    @property
    def dataloader_valsize(self):
        return int(Config.cfg[Config.CATEGORY_INPUT][Config.INPUT_VALSIZE])

    @property
    def dataloader_valbatchsize(self):
        return int(Config.cfg[Config.CATEGORY_INPUT][Config.INPUT_VALBATCHSIZE])

    @property
    def dataloader_testsize(self):
        return int(Config.cfg[Config.CATEGORY_INPUT][Config.INPUT_TESTSIZE])

    @property
    def dataloader_testbatchsize(self):
        return int(Config.cfg[Config.CATEGORY_INPUT][Config.INPUT_TESTBATCHSIZE])

    # Trainer
    @property
    def trainer_name(self):
        return Config.cfg[Config.CATEGORY_TRAINER][Config.TRAINER_NAME]

    @property
    def trainer_numepochs(self):
        return int(Config.cfg[Config.CATEGORY_TRAINER][Config.TRAINER_NUMEPOCHS])

    @property
    def trainer_lr(self):
        return ast.literal_eval(Config.cfg[Config.CATEGORY_TRAINER][Config.TRAINER_LEARNINGRATE])

    # Model
    @property
    def model_name(self):
        return Config.cfg[Config.CATEGORY_MODEL][Config.MODEL_NAME]

    @property
    def model_nclass(self):
        return int(Config.cfg[Config.CATEGORY_MODEL][Config.MODEL_NCLASS])

    @property
    def model_statepath(self):
        r = Config.cfg[Config.CATEGORY_MODEL][Config.MODEL_STATEPATH]
        if r in ["none", "", "None"]:
            return None
        return r

    # Inferencer
    @property
    def inferencer_name(self):
        return Config.cfg[Config.CATEGORY_INFERENCER][Config.INFERENCER_NAME]

    @property
    def dataloader_print_inferencer_num(self):
        return int(Config.cfg[Config.CATEGORY_INFERENCER][Config.INFERENCER_PRINT_NUM])

    # Output
    @property
    def output_printfolder(self):
        return Config.cfg[Config.CATEGORY_OUTPUT][Config.OUTPUT_PRINTFOLDER]

    @property
    def output_modelfolder(self):
        return Config.cfg[Config.CATEGORY_OUTPUT][Config.OUTPUT_MODELFOLDER]
