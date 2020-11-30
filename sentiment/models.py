import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNClassifier(nn.Module):
    def __init__(self, config, vocab_size, label_size):
        super(RNNClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, config["nembedding"], padding_idx=0)
        self.gru = nn.GRU(
            input_size=config["nembedding"],
            hidden_size=config["nhidden"],
            num_layers=config["nlayers"],
            dropout=config["dropout"],
            bidirectional=False,
        )
        self.dense = nn.Linear(in_features=config["nhidden"], out_features=label_size)

    def forward(self, entity_ids, seq_len):
        embedding = self.embedding(entity_ids)
        embedding = pack_padded_sequence(embedding, seq_len, batch_first=False)
        out, _ = self.gru(embedding)
        out, lengths = pad_packed_sequence(out, batch_first=False)
        # Since we are doing classification, we only need the last
        # output from RNN
        lengths = [length - 1 for length in lengths]
        last_output = out[lengths, range(len(lengths))]
        logits = self.dense(last_output)
        return logits


class StackedCRNNClassifier(nn.Module):
    def __init__(self, config, vocab_size, label_size):
        super(StackedCRNNClassifier, self).__init__()
        cnn_config = config["cnn"]
        rnn_config = config["rnn"]

        self.embedding = nn.Embedding(vocab_size, config["nembedding"], padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, Nk, Ks)
                for Ks, Nk in zip(cnn_config["kernel_sizes"], cnn_config["nkernels"])
            ]
        )
        self.lstm = nn.LSTM(
            input_size=cnn_config["nkernels"][-1],
            hidden_size=rnn_config["nhidden"],
            num_layers=rnn_config["nlayers"],
            dropout=rnn_config["dropout"],
            bidirectional=False,
        )
        self.dense = nn.Linear(
            in_features=rnn_config["nhidden"], out_features=label_size
        )

    def forward(self, entity_ids, seq_len):
        x = self.embedding(entity_ids)
        x = x.transpose(0, 1)

        for i, conv in enumerate(self.convs):
            # Since we are using conv2d, we need to add extra outer dimension
            x = x.unsqueeze(1)
            x = F.relu(conv(x)).squeeze(3)
            x = x.transpose(1, 2)

        out, _ = self.lstm(x.transpose(0, 1))
        last_output = out[-1, :, :]
        logits = self.dense(last_output)

        return logits