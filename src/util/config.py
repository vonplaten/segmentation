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

    # Dataloader
    CATEGORY_DATALOADER = "dataloader"
    DATALOADER_NAME = "name"

    DATALOADER_TRAINSIZE = "train_size"
    DATALOADER_VALSIZE = "val_size"
    DATALOADER_TESTSIZE = "test_size"

    DATALOADER_TRAINBATCHSIZE = "train_batch_size"
    DATALOADER_VALBATCHSIZE = "val_batch_size"
    DATALOADER_TESTBATCHSIZE = "test_batch_size"

    DATALOADER_TRAINNUMWORKERS = "train_num_workers"
    DATALOADER_VALNUMWORKERS = "val_num_workers"
    DATALOADER_TESTNUMWORKERS = "test_num_workers"

    DATALOADER_TRAINSHUFFLE = "train_shuffle"
    DATALOADER_VALSHUFFLE = "val_shuffle"
    DATALOADER_TESTSHUFFLE = "test_shuffle"

    DATALOADER_PRINTFOLDER = "print_folder"
    DATALOADER_PRINT_INPUT_NUM = "print_input_num"

    # Model
    CATEGORY_MODEL = "model"
    MODEL_NAME = "name"
    MODEL_PRINTFOLDER = "print_folder"
    MODEL_NCLASS = "n_class"
    MODEL_STATEPATH = "state"

    # Inferencer
    CATEGORY_INFERENCER = "inferencer"
    INFERENCER_NAME = "name"
    INFERENCER_PRINT_NUM = "print_inf_num"

    def __init__(self):
        if Config.cfg is None:
            with open("config.yaml") as f:
                Config.cfg = yaml.load(f, Loader=yaml.BaseLoader)

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

    # Dataloader
    @property
    def dataloader_name(self):
        return Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_NAME]

    @property
    def dataloader_trainsize(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TRAINSIZE])

    @property
    def dataloader_valsize(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_VALSIZE])

    @property
    def dataloader_testsize(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TESTSIZE])

    @property
    def dataloader_trainbatchsize(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TRAINBATCHSIZE])

    @property
    def dataloader_valbatchsize(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_VALBATCHSIZE])

    @property
    def dataloader_testbatchsize(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TESTBATCHSIZE])

    @property
    def dataloader_trainnumworkers(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TRAINNUMWORKERS])

    @property
    def dataloader_valnumworkers(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_VALNUMWORKERS])

    @property
    def dataloader_testnumworkers(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TESTNUMWORKERS])

    @property
    def dataloader_trainshuffle(self):
        return ast.literal_eval(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TRAINSHUFFLE])

    @property
    def dataloader_valshuffle(self):
        return ast.literal_eval(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_VALSHUFFLE])

    @property
    def dataloader_testshuffle(self):
        return ast.literal_eval(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_TESTSHUFFLE])

    @property
    def dataloader_printfolder(self):
        return Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_PRINTFOLDER]

    @property
    def dataloader_print_input_num(self):
        return int(Config.cfg[Config.CATEGORY_DATALOADER][Config.DATALOADER_PRINT_INPUT_NUM])

    # Model
    @property
    def model_name(self):
        return Config.cfg[Config.CATEGORY_MODEL][Config.MODEL_NAME]

    @property
    def model_printfolder(self):
        return Config.cfg[Config.CATEGORY_MODEL][Config.MODEL_PRINTFOLDER]

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

    # # Hyperparams
    # HYPERPARAMS = "hyperparams"
    # LEARNINGRATE = "learning_rate"
    # BATCHSIZE = "batch_size"
    # NUMEPOCHS = "num_epochs"
    # MINIBATCHGRADIENTSIZE = "mini_batch_gradient_size"
    # TRAININPUTBALANCE = "train_input_balance"

    # # Settings
    # SETTINGS = "settings"
    # NUMCLASSES = "num_classes"
    # ISDEBUGRUN = "debug_run"
    # MODELSTATEPATH = "state_path"
    # OUTPATH =  "out_folder_path"
    # TENSORBOARDPATH = "tensorboard_path"
    # NOISETHRESHOLD = "noise_threshold"
    # PRINTEPOCHS = "print_epochs"
    # NUMPRINTS = "num_prints"

    # # Hyperparams
    # @property
    # def learningrate(self):
    #     return float(Config.cfg[Config.HYPERPARAMS][Config.LEARNINGRATE])

    # @property
    # def batchsize(self):
    #     return int(Config.cfg[Config.HYPERPARAMS][Config.BATCHSIZE])

    # @property
    # def numepochs(self):
    #     return int(Config.cfg[Config.HYPERPARAMS][Config.NUMEPOCHS])

    # @property
    # def minibatchgradientsize(self):
    #     return int(Config.cfg[Config.HYPERPARAMS][Config.MINIBATCHGRADIENTSIZE])

    # @property
    # def accumulationsteps(self):
    #     return self.minibatchgradientsize // self.batchsize

    # # @property
    # # def schedulerstepsize(self):
    # #     return int(Config.cfg[Config.HYPERPARAMS][Config.SCHEDULERSTEPSIZE])
    # @property
    # def traininputbalance(self):
    #     return ast.literal_eval(Config.cfg[Config.HYPERPARAMS][Config.TRAININPUTBALANCE])

    # # Settings
    # @property
    # def isdebugrun(self):
    #     s = Config.cfg[Config.SETTINGS][Config.ISDEBUGRUN]
    #     if s == 'True':
    #         return True
    #     elif s == 'False':
    #         return False
    #     else:
    #         raise ValueError

    # @property
    # def modelstatepath(self):
    #     p = Config.cfg[Config.SETTINGS][Config.MODELSTATEPATH]
    #     if p == "none":
    #         return None
    #     return p

    # @property
    # def outpath(self):
    #     if (self.isdebugrun):
    #         return os.path.join(Config.cfg[Config.SETTINGS][Config.OUTPATH], "runs", "debug")
    #     return os.path.join(Config.cfg[Config.SETTINGS][Config.OUTPATH],
    # "runs", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(' ','_'))

    # @property
    # def numclasses(self):
    #     return int(Config.cfg[Config.SETTINGS][Config.NUMCLASSES])

    # @property
    # def tensorboardpath(self):
    #     p = Config.cfg[Config.SETTINGS][Config.TENSORBOARDPATH]
    #     if p == "none":
    #         return None
    #     return p

    # @property
    # def noisethreshold(self):
    #     return int(Config.cfg[Config.SETTINGS][Config.NOISETHRESHOLD])

    # @property
    # def printepochnrlist(self):
    #     return ast.literal_eval(Config.cfg[Config.SETTINGS][Config.PRINTEPOCHS])

    # @property
    # def numprints(self):
    #     return int(Config.cfg[Config.SETTINGS][Config.NUMPRINTS])
