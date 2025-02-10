import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="cnn_emo_classification")


vocab = set()
PAD = '<pad>'
ATTENTION = '<attention>'

vocab.add(PAD)
vocab.add(ATTENTION)

def vocab2id(vocab):
    word2id = {}
    for i, word in enumerate(vocab):
        word2id[word] = i
    return word2id

class AtepcDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            setences = []
            aspects = []
            labels = []

            setence = []
            aspect = []
            label = []
            for line in f:
                if line == '\n':
                    setences.append(setence)
                    aspects.append(aspect)
                    labels.append(label)
                    setence = []
                    aspect = []
                    label = []
                    continue
                word = line.split(' ')
                assert len(word) == 3
                setence.append(word[0])
                vocab.add(word[0])
                aspect.append(word[1])
                label.append(int(word[2]))

            self.data = self.load_data(setences, aspects, labels)

    def load_data(self, setences, aspects, labels):
        data = []
        for sentence, aspect, label in zip(setences, aspects, labels):
            for index, asp in enumerate(aspect):
                if asp == 'O':
                    continue
                if label[index] == -1:
                    continue
                data.append((sentence, index, label[index]))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = AtepcDataset('datasets/mixed/mixed.atepc.train.dat')
test_dataset = AtepcDataset('datasets/mixed/mixed.atepc.test.dat')

print('---- Loading dataset ----')

print('train_dataset size:', len(train_dataset))
print('test_dataset size:', len(test_dataset))

print('---- Finished loading dataset ----\n')

vocab_index = vocab2id(vocab)

MIN_PADDING_LENGTH = 16

def collate_fn(batch):
    sentence_lengths = [len(sentence) for sentence, _, _ in batch]
    max_sentence_length = max(max(sentence_lengths), MIN_PADDING_LENGTH)
    sentence_tensor = torch.zeros((len(batch), max_sentence_length), dtype=torch.long)
    aspect_tensor = torch.zeros((len(batch)), dtype=torch.long)
    label_tensor = torch.zeros((len(batch)), dtype=torch.long)

    for i, (sentence, index, label) in enumerate(batch):
        sentence_tensor[i, :len(sentence)] = torch.tensor([vocab_index[word] for word in sentence], dtype=torch.long)
        sentence_tensor[i, len(sentence):] = torch.tensor([vocab_index[PAD]] * (max_sentence_length - len(sentence)), dtype=torch.long)
        aspect_tensor[i] = index
        label_tensor[i] = label

    return sentence_tensor, aspect_tensor, label_tensor


filter_sizes = [3, 5, 7, 9]
out_channels = [128, 64, 32, 16]
assert len(filter_sizes) == len(out_channels)
EMBEDDING_DIM = 256
BATCH_SIZE = 32
EPOCHS = 100

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
ATTE_TERSOR = torch.tensor([vocab_index[ATTENTION]], dtype=torch.long).to(device)


class TextCNN(nn.Module):
    def __init__(self, vocab_size):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, EMBEDDING_DIM)
        self.convs = nn.ModuleList([nn.Conv2d(1, c, (k, EMBEDDING_DIM)) for c, k in zip(out_channels, filter_sizes)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(sum(out_channels), 3)

    def forward(self, x, aspect):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        atte_embedding = self.embedding(ATTE_TERSOR)  # (1, embedding_dim)
        atte_embedding = 10 * atte_embedding.squeeze(0)  # (embedding_dim)
        for batch, aspect_index in enumerate(aspect):
            x[batch, aspect_index, :] += atte_embedding
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(batch_size, out_channel, seq_len - filter_size + 1)] * len(filter_sizes)
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x] # [(batch_size, out_channel)] * len(filter_sizes)
        x = torch.cat(x, 1)  # (batch_size, len(filter_sizes) * out_channel)
        x = self.dropout(x) # (batch_size, len(filter_sizes) * out_channel)
        x = self.fc(x)  # (batch_size, 3)
        # x = F.softmax(x, dim=1)  # (batch_size, 3)
        return x


net = TextCNN(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train(epoch):
    net.train()
    for i, (sentence, aspect, label) in enumerate(train_loader):
        optimizer.zero_grad()
        sentence = sentence.to(device)
        aspect = aspect.to(device)
        output = net(sentence, aspect)
        output = output.to(torch.device('cpu'))
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}'.format(epoch+1, EPOCHS, i+1, len(train_loader), loss.item()))
            wandb.log({'train_loss': loss.item(), 'epoch': epoch+1})

def evaluate(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence, aspect, label in test_loader:
            sentence = sentence.to(device)
            aspect = aspect.to(device)
            output = net(sentence, aspect)
            output = output.to('cpu')
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test set: {:.4f} %'.format(accuracy))
    wandb.log({'test_accuracy': accuracy, 'epoch': epoch+1})
    return accuracy

def save_model(path):
    torch.save(net.state_dict(), path)

def load_model(path):
    net.load_state_dict(torch.load(path, weights_only=True))

max_accuracy = 0
for epoch in range(EPOCHS):
    train(epoch)
    accuracy = evaluate(epoch)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        # save_model('models/%d.pth' % (epoch+1))
save_model('models/final.pth')
print('Max accuracy:', max_accuracy)

# load_model('models/1.pth')
# evaluate(1)

wandb.finish()

