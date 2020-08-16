class TrainConfig(object):
    def __init__(self):
        self.epochs = 10
        self.batch_size = 4
        self.workers = 4
        self.initial_lr = 5e-4
        self.weight_decay = 1e-2
        self.experiment_name = 'run1'
        self.train_labels = 'annotations/train.json'
        self.val_labels = 'annotations//val.json'
        self.mp3_folder = '/media/eloi/WindowsDrive/data/mozilla_speech/clips'
        self.checkpoint_every = 1000
        self.test_mp3 = '/media/eloi/WindowsDrive/data/mozilla_speech/clips/common_voice_en_1342.mp3'
        self.max_time = 2500  # excludes approximately 0.33% of the dataset and caps the VRAM usage
        self.val_batches = 400 // self.batch_size
        self.val_every = 1000
