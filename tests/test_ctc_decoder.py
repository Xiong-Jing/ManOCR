import torch

from manchu_ocr.models.recognition.decoders.ctc_decoder import CTCGreedyDecoder


def test_ctc_greedy_decoder_collapses_repeats_and_blanks():
    decoder = CTCGreedyDecoder({0: "", 1: "a", 2: "b"}, blank_idx=0)
    indices = torch.tensor([[0, 1, 1, 0, 2, 2, 0]])

    assert decoder.decode_indices(indices) == ["ab"]
