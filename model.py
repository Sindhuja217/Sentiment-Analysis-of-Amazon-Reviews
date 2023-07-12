import torch
import torch.nn as nn

class Classifier_model(nn.Module):
        def __init__(self, vocab_size, embedded_matrix, embedded_dim, activation_func, dropout_rate, num_classes, vectors, sentence_length):
                super(Classifier_model, self).__init__()
                self.embedded = nn.Embedding(vocab_size, embedded_dim)
                self.embedded.weight = nn.Parameter(torch.tensor(embedded_matrix))
                embedded_dim = vectors.vector_size
                hidden_size = int(3 * embedded_dim)
                self.fc1 = nn.Linear(embedded_dim * sentence_length, hidden_size)
                self.activation = activation_func
                self.dropout = nn.Dropout(dropout_rate)
                self.fc2 = nn.Linear(hidden_size, num_classes)
                self.output = nn.Softmax(dim = 1)

        def forward(self, x):
                x = self.embedded(x)
                x = torch.flatten(x, start_dim = 1)
                x = self.fc1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.output(x)

                return x