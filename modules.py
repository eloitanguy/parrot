from dataset import INDEX_TO_C, C_TO_INDEX
import torch


def greedy_decoder(softmax_output):
    """ Decodes a softmax output of shape (classes, time) into a sentence"""
    ctc = torch.argmax(softmax_output, dim=0)
    print(ctc)
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
