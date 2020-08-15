from dataset import INDEX_TO_C, C_TO_INDEX
import torch
import gtts
from torch import transpose
import torchaudio


def greedy_decoder(softmax_output):
    """ Decodes a softmax output of shape (classes, time) into a sentence"""
    ctc = torch.argmax(softmax_output, dim=0)
    decoded_classes = []
    for idx in range(ctc.shape[0]):
        if ctc[idx] == 0:  # epsilon
            continue
        if idx == 0:  # the first character can't have redundancy with the previous character
            decoded_classes.append(ctc[idx].item())
        else:
            if ctc[idx - 1] != ctc[idx]:  # change -> new character
                decoded_classes.append(ctc[idx].item())
            else:  # redundancy: pass
                continue

    decoded_sentence = ''
    for decoded_class in decoded_classes:
        decoded_sentence += INDEX_TO_C[decoded_class]

    return {'sentence': decoded_sentence, 'classes': decoded_classes}


def one_hot_from_character(character):
    out = torch.zeros((len(INDEX_TO_C), 1))
    out[C_TO_INDEX[character]] = 1
    return out


# predict:(batch_size, time, number of classes), target: (batch_size, max target length)
def ctc_loss(predict, target, predict_lengths, target_lengths):
    predict = torch.transpose(predict, 0, 1)  # to (time, batch_size, classes)
    ctc = torch.nn.CTCLoss(blank=0, reduction='mean')
    loss = ctc(predict, target, predict_lengths, target_lengths)
    return loss


def test_mp3_file(input_path, model, out_path):
    """ Reads the mp3 file <input_path> with the <model>
    and outputs its inference into a text-to-speech mp3 at <output_path>"""

    waveform, _ = torchaudio.load(input_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform).cuda()  # shapes (1, 128, time)
    out_init = model(mel_spectrogram).squeeze(0)  # shape (time, 29)
    out_trans = transpose(out_init, 1, 0)  # shape (29, time)
    decode_out = greedy_decoder(out_trans)['sentence']
    tts = gtts.gTTS(decode_out)
    tts.save(out_path)
