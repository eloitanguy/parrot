from dataset import C_TO_INDEX, INDEX_TO_C
import torch


def greedy_decoder(softmax_output):
    """ Decodes a softmax output of shape (classes, time) into a sentence"""
    chosen_characters_classes = torch.argmax(softmax_output, dim=1)
    decoded_classes = []
    num_classes = len(C_TO_INDEX)
    prev_character_class = 0
    for i, character_class in range(chosen_characters_classes.shape[0]):
        return None # TODO
