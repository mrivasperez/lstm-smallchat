import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

# Hyperparameters
vocab_size = 10000
embedding_dim = 512
hidden_dim = 1024
batch_size = 32
num_epochs = 30
learning_rate = 0.001

# Load and preprocess the data


def load_data(file_path):
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            q, a = line.strip().split('\t')
            questions.append(q)
            answers.append(a)
    return questions, answers


questions, answers = load_data('data.txt')

# Create vocabulary
word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
idx2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
word_index = 4
for sentence in questions + answers:
    for word in sentence.lower().split():
        if word not in word2idx:
            word2idx[word] = word_index
            idx2word[word_index] = word
            word_index += 1

# Tokenize and pad sequences


def tokenize_and_pad(sentences, max_length):
    tokenized = []
    for sentence in sentences:
        tokens = [word2idx['<sos>']] + [word2idx.get(word, word2idx['<unk>'])
                                        for word in sentence.lower().split()] + [word2idx['<eos>']]
        tokenized.append(tokens)
    padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(
        seq) for seq in tokenized], batch_first=True, padding_value=word2idx['<pad>'])
    return padded


max_length = max(len(s.split())
                 for s in questions + answers) + 2  # +2 for <sos> and <eos>
questions_tokenized = tokenize_and_pad(questions, max_length)
answers_tokenized = tokenize_and_pad(answers, max_length)

# Dataset and DataLoader


class ChatDataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]


dataset = ChatDataset(questions_tokenized, answers_tokenized)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Definition (LSTM)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=word2idx['<pad>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=word2idx['<pad>'])
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, question, answer, teacher_forcing_ratio=0.5):
        batch_size, max_length = answer.shape
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_length, vocab_size).to(device)
        hidden, cell = self.encoder(question)

        # First input to the decoder is the <sos> token
        input = answer[:, 0]

        for t in range(1, max_length):
            output, hidden, cell = self.decoder(
                input.unsqueeze(1), hidden, cell)
            outputs[:, t] = output.squeeze(1)
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(2).squeeze(1)
            input = answer[:, t] if teacher_force else top1

        return outputs


# Instantiate the model and move it to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if a saved state dictionary exists
model_path = 'chatbot_model.pth'
if os.path.exists(model_path):
    # Load the saved model
    print("Loading saved model...")
    encoder = Encoder(len(word2idx), embedding_dim, hidden_dim)
    decoder = Decoder(len(word2idx), embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from:", model_path)
else:
    # Instantiate a new model
    print("No saved model found. Training a new model...")
    encoder = Encoder(len(word2idx), embedding_dim, hidden_dim)
    decoder = Decoder(len(word2idx), embedding_dim, hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter('runs')

    def print_welcome_message():
        print("************************************")
        print("*  Welcome to the Chatbot Trainer! *")
        print("************************************")
        print("\nStarting training process...\n")

    # Call the welcome message function
    print_welcome_message()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for question, answer in dataloader:
            question, answer = question.to(device), answer.to(device)

            optimizer.zero_grad()
            output = model(question, answer)

            # Reshape output and target for loss calculation
            output = output[:, 1:].reshape(-1, len(word2idx))
            answer = answer[:, 1:].reshape(-1)

            loss = criterion(output, answer)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', total_loss / len(dataloader), epoch)

        print(
            f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')

    # Close the SummaryWriter
    writer.close()

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print("Model saved to:", model_path)

# Inference (Testing/Chatting)


def generate_response(question, max_response_length=20):
    model.eval()
    with torch.no_grad():
        question_tokenized = tokenize_and_pad(
            [question], max_length).to(device)
        hidden, cell = model.encoder(question_tokenized)
        input = torch.tensor([word2idx['<sos>']]).to(device)
        response = []

        for _ in range(max_response_length):
            output, hidden, cell = model.decoder(
                input.unsqueeze(0), hidden, cell)
            predicted_word_idx = output.argmax(2).item()
            if predicted_word_idx == word2idx['<eos>']:
                break
            response.append(idx2word[predicted_word_idx])
            input = torch.tensor([predicted_word_idx]).to(device)

        return ' '.join(response)


# Example usage:
question = "hi, how are you doing?"
response = generate_response(question)
print(f"Question: {question}")
print(f"Response: {response}")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = generate_response(user_input)
    print(f"Bot: {response}")
