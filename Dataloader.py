from random import shuffle 
import torch 

class AmazonReviews(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, vectors, sentence_length, device):
        self.sentences = sentences
        self.labels = labels
        self.vectors = vectors
        self.sentence_limit = sentence_length
        self.device = device

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index][0]

        ids = self.tok_to_ind(sentence)
        no_ids = len(ids)

        return {"ids" : torch.tensor(ids).to(self.device), "label" : torch.tensor(label).to(self.device), "length" : no_ids}
    
    def tok_to_ind(self, tokens):
        sos_token = "<SOS>"
        eos_token = "<EOS>"
        pad_token = "<PAD>"
        unk_token = "<UNK>"

        token_ids = [self.vectors.key_to_index.get(token, self.vectors.key_to_index[unk_token]) for token in tokens]
        sentence_length = len(token_ids)
        if sentence_length < self.sentence_limit - 2:
            token_ids = token_ids[:sentence_length]
            token_ids.append(self.vectors.key_to_index[eos_token])
        else:
            token_ids = token_ids[:self.sentence_limit - 2]
            token_ids.append(self.vectors.key_to_index[eos_token])

        token_ids.insert(0, self.vectors.key_to_index[sos_token])
        token_ids = token_ids + [self.vectors.key_to_index[pad_token]] * (self.sentence_limit - len(token_ids))


        return token_ids
    
def dataloader(sentences, labels, vectors, sentence_length, batch_size, device):
    data = AmazonReviews(sentences, labels, vectors, sentence_length, device)
    return torch.utils.data.DataLoader(data, batch_size, shuffle=True)
