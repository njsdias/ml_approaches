import config
import transformers
import torch.nn as nn


class BERTBasedUncased(nn.Module):
    def __init__(self):
        super(BERTBasedUncased, self).__init__()

        # load the model from BERT_PATH
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)

        # add a dropout for regularization
        self.bert_drop = nn.Dropout(0.3)

        # a simple linear layer for output
        # yes, there is only one output
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # BERT in its default settings returns two outputs
        # last hidden state and output of bert pooler layer
        # we use the output of the pooler which is of the size
        # (batch_size, hidden_size)
        # hidden size can be 768 or 1024 depending on
        # if we are using bert base or large respectively
        # in our case, it is 768
        # note that this model is pretty simple
        # you might want to use last hidden state
        # or several hidden states
        _, o2 = self.bert(ids, attention_mask=mask,
                          token_type_ids=token_type_ids)

        # pass through dropout layer
        bo = self.bert_drop(o2)

        # pass through linear layer
        output = self.out(bo)

        # return output
        return output


