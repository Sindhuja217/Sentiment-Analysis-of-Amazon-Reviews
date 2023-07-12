import csv
import sys
import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Dataloader import dataloader
from tqdm import tqdm
from model import Classifier_model
from pathlib import Path
import itertools


torch.manual_seed(54)

word2vec_model = r'../data/word2vec.pkl'
with open(word2vec_model, 'rb') as file:
        word2vec = pickle.load(file)


list = [["<SOS>", "<EOS>", "<PAD>", "<UNK>"] * 10]
word2vec.build_vocab(list, update=True)

vectors = word2vec.wv
embedded_matrix = word2vec.wv.vectors
vocab_size, embedded_dim = embedded_matrix.shape

folder_path = sys.argv[1]

train_d = folder_path + r"/train.csv"
trainlabel = folder_path + r"/trainlabel.csv"
val_d = folder_path + r"/val.csv"
vallabel = folder_path + r"/vallabel.csv"
test_d = folder_path + r"/test.csv"
testlabel = folder_path + r"/testlabel.csv"

def read_files(file_path):
    comments = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data = []
            for word in row:
                word = word.replace("[", "").replace("]", "").replace("'", "")
                word = word.strip()
            #print(row[0])
                data.append(word)
            comments.append(data)

    return comments


def read_label(label_path):
    labels = []
    with open(label_path, 'r') as file_l:
        reader_l = csv.reader(file_l)
        for i in reader_l:
            i_int = [int(value) for value in i]
            labels.append(i_int)

    return labels

def train(dataloader, model, criterion, optimizer):
    model.train()
    losses, accuracy = [], []
    for batch in tqdm(dataloader):
        y = batch["label"]
        logits = model(batch['ids'])
        loss = criterion(logits, y) #Diff
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        accuracy.append((preds == y).float().mean().item())
    print('Train Loss:', round(np.array(losses).mean() , 2), "Train Accuracy:", round(np.array(accuracy).mean() * 100 , 2))
    return np.array(accuracy).mean()     
     
@torch.no_grad()
def test(dataloader, model, criterion):
    model.eval()
    losses, accuracy = [], []
    for batch in dataloader:
        y = batch['label']
        logits = model(batch['ids'])
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds = torch.argmax(logits, -1)
        accuracy.append((preds == y).float().mean().item())
    print('Loss', round(np.array(losses).mean() , 2), 'Accuracy', round(np.array(accuracy).mean() * 100, 2))
    return np.array(accuracy).mean(), np.array(losses).mean()
    
def pred(dataloader, model):
    prediction = []
    for batch in dataloader:
        logits = model(batch['ids'])
        preds = torch.argmax(logits, -1)
        prediction.append(preds)
    predict = torch.cat(prediction, dim=0)
        
    return predict
    

def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    train_data = read_files(train_d)
    train_label = read_label(trainlabel)
    val_data = read_files(val_d)
    val_label = read_label(vallabel)
    test_data = read_files(test_d)
    test_label = read_label(testlabel)

    train_dataloader = dataloader(train_data, train_label, vectors, sentence_length = 30, batch_size = 128, device = device)
    val_dataloader = dataloader(val_data, val_label, vectors, sentence_length = 30, batch_size = 128, device = device)
    test_dataloader = dataloader(test_data, test_label, vectors, sentence_length = 30, batch_size = 128, device = device)
    
    hidden_layers = [torch.nn.ReLU(), torch.nn.Sigmoid(), torch.nn.Tanh()]
    hidden_layer_files = ['nn_relu.model', 'nn_sigmoid.model', 'nn_tanh.model']
    names = ['relu', 'sigmoid', 'tanh']
    dropouts = [0.2, 0.4, 0.6]
    learning_rate = [0.05, 0.01, 0.001]
    result_header = ['Epochs', 'Activation Fn', 'Dropout', 'Learn Rate', 'Train Acc', 'Val Acc']
    results = []

    #results = ''
    for i, hidden_layer in enumerate(hidden_layers):
        best_accuracy = None
        parameter_combinations = itertools.product(dropouts, learning_rate)
        for parameter in parameter_combinations:
            print('dropout' ,parameter[0], 'learning_rate' , parameter[1])
            model = Classifier_model(vocab_size, embedded_matrix, embedded_dim, hidden_layer, parameter[0], 2, vectors, 30)
            model.to(device)
            print(model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), parameter[1], momentum= 0.9, weight_decay= 1e-5)

            for epoch in range(3):
                print("Epoch {}".format(epoch))
                training_accuracy = train(train_dataloader, model, criterion, optimizer)
                print('Validating')
                validation_accuracy, validation_loss = test(val_dataloader, model, criterion)
                print("Testing")
                testing_accuracy = test(test_dataloader, model, criterion)
                results.append([(epoch+1), hidden_layer, parameter[0], parameter[1], round(training_accuracy*100,2), round(validation_accuracy,2)])

            if best_accuracy is not None and best_accuracy > validation_accuracy:
                print("Model is  not saved")

            else:
                best_accuracy, loss  = validation_accuracy, validation_loss
                best_dropout = parameter[0]
                best_learningrate = parameter[1]
                for hidden_file in hidden_layer_files:
                    model_file = r'data/' + hidden_file
                    model_file = Path(model_file).__str__()
                    # torch.save(model.state_dict(), model_file)
                    torch.save(model, 'data/'+'nn_'+ names[i]+'.model')
        #results += 'Activation Fn: {}, Dropout: {}, Learning Rate: {}, Accuracy: {}'.format(names[i], best_dropout, best_learningrate, best_accuracy)
        #print(results)

        print('Activation_funct : ', names[i])
        print('Best_Accuracy : {:.2f}'.format(best_accuracy * 100))
        print('loss : {:.2f}'.format(loss))
        print('Dropout : ', best_dropout)
        print('Learning rate :', best_learningrate)

    test_accuracy = []
    loss = []
    criterion = torch.nn.CrossEntropyLoss()
    for i, hidden_layer in enumerate(hidden_layers):
        model = torch.load('data/'+'nn_'+ names[i]+'.model')
        accuracy, losses= test(test_dataloader, model, criterion)
        test_accuracy.append("{:.2f}".format(accuracy*100))
        loss.append("{:.2f}".format(losses))
    print('Test Accuracy - ', test_accuracy)
    print('Test Loss - ', loss)

    #Saving all results
    with open('data/train_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
    
    # Write headers
        writer.writerow(result_header)
    
    # Write data rows
        writer.writerows(results)



if __name__ == '__main__':
    main()

