import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import transpose
import json
import csv
from utils import printProgressBar
import torchaudio
import random
import os

from torch.utils.data import DataLoader
from model import ParrotModel

HARD_TSV_PATHS = {'test': '/media/eloi/WindowsDrive/data/mozilla_speech/test.tsv',
                  'other': '/media/eloi/WindowsDrive/data/mozilla_speech/other.tsv',
                  'dev': '/media/eloi/WindowsDrive/data/mozilla_speech/dev.tsv',
                  'train': '/media/eloi/WindowsDrive/data/mozilla_speech/train.tsv',
                  'validated': '/media/eloi/WindowsDrive/data/mozilla_speech/validated.tsv'}

TSV_LENGTH = {'test': 16030,
              'other': 175085,
              'dev': 16030,
              'train': 435948,
              'validated': 1085495}

C_TO_INDEX = {'epsilon': 0,
              'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12,
              'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23,
              'x': 24, 'y': 25, 'z': 26, ' ': 27, "'": 28
              }

INDEX_TO_C = {0: 'epsilon',
              1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l',
              13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w',
              24: 'x', 25: 'y', 26: 'z', 27: ' ', 28: "'"
              }


def train_collate_function(data):
    spectrograms = [transpose(data[idx]['spectrogram'], 1, 0) for idx in range(len(data))]  # temp shapes (time, 128)
    labels = [data[idx]['label'] for idx in range(len(data))]
    spectrograms = transpose(pad_sequence(spectrograms, batch_first=True), 1, 2)  # final shape (batch_size, 128, time)
    return {'spectrogram': spectrograms.type(torch.half), 'label': labels}


class ParrotTrainDataset(Dataset):
    def __init__(self, train_labels, mp3_folder):
        with open(train_labels, 'r') as f:
            self.labels = json.load(f)
        self.mp3_folder = mp3_folder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ann = self.labels[idx]
        waveform, _ = torchaudio.load(os.path.join(self.mp3_folder, ann['file_name']))
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform).squeeze(0)
        return {'spectrogram': mel_spectrogram, 'label': ann}


def split_annotations(ann_file, val_percent=0.02, test_percent=0.02):
    with open(ann_file, 'r') as f:
        all_annotations = json.load(f)

    random.shuffle(all_annotations)
    n = len(all_annotations)
    idx_val = int(val_percent*n)
    idx_test = idx_val + int(test_percent*n)

    with open('val.json', 'w') as f:
        json.dump(all_annotations[:idx_val], f, indent=4)
    with open('test.json', 'w') as f:
        json.dump(all_annotations[idx_val:idx_test], f, indent=4)
    with open('train.json', 'w') as f:
        json.dump(all_annotations[idx_test:], f, indent=4)


def prepare_sentence(sentence):
    prep_sent = sentence.lower().replace('.', '').replace(',', '').replace('!', '').replace('"', '').replace('?', '').\
        replace(':', '').replace('\u2019', "'").replace('\u2018', "'").replace('\u2014', "-").replace('\u201c', '').\
        replace('\u201d', '').replace('\u00e2', "'").replace(':', '').split(' ')
    out = []
    for word in prep_sent:
        out.extend(list(word)+[' '])
    return out


def dump_full_annotation_json():
    output_by_file_name = {}
    for path_label, path in HARD_TSV_PATHS.items():
        with open(path, 'r') as f:
            read_tsv = csv.reader(f, delimiter="\t")
            tsv_length = TSV_LENGTH[path_label]
            for line_idx, line in enumerate(read_tsv):
                printProgressBar(line_idx, tsv_length, prefix=path_label)
                file_name = line[1]
                sentence = line[2]
                if line_idx != 0 and sentence != '':
                    output_by_file_name[file_name] = prepare_sentence(sentence)

    output = [{'file_name': key, 'sentence': item} for key, item in output_by_file_name.items()]
    with open('all_annotations.json', 'w') as out:  # dumping several times for sanity checking
        json.dump(output, out)


if __name__ == '__main__':
    # dump_full_annotation_json()
    # split_annotations('all_annotations.json')
    dev_set = ParrotTrainDataset('annotations/val.json', '/media/eloi/WindowsDrive/data/mozilla_speech/clips/')
    dev_loader = DataLoader(dataset=dev_set, num_workers=6, batch_size=6, collate_fn=train_collate_function)
