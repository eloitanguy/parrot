class TrainConfig(object):
    def __init__(self):
        self.epochs = 10
        self.batch_size = 2
        self.workers = 4
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        self.experiment_name = 'test'
        self.train_labels = 'annotations/train.json'
        self.val_labels = 'annotations//val.json'
        self.mp3_folder = '/media/eloi/WindowsDrive/data/mozilla_speech/clips'
        self.checkpoint_every = 100
