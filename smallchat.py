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
        outputs, (hidden, cell) = self.lstm(
            embedded)  # Note: We need outputs here
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]

        seq_len = encoder_outputs.shape[1]

        # Repeat the hidden state of the decoder across the sequence length dimension
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention energies
        energy = torch.tanh(
            self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))

        # Calculate attention weights
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        # Softmax to get probabilities
        return nn.functional.softmax(attention, dim=1)


class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=word2idx['<pad>'])
        self.lstm = nn.LSTM(embedding_dim + hidden_dim,
                            hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = attention

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: [batch_size]  (previous word)
        # hidden: [batch_size, hidden_dim]
        # cell: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]

        # [batch_size, 1, embedding_dim]
        embedded = self.embedding(x.unsqueeze(1))

        # Calculate attention weights
        attn_weights = self.attention(
            hidden, encoder_outputs)  # [batch_size, seq_len]

        # Calculate context vector
        context = attn_weights.unsqueeze(1).bmm(
            encoder_outputs)  # [batch_size, 1, hidden_dim]

        # Concatenate embedding and context
        # [batch_size, 1, embedding_dim + hidden_dim]
        rnn_input = torch.cat((embedded, context), dim=2)

        # Pass through LSTM
        output, (hidden, cell) = self.lstm(
            rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))

        # hidden: [1, batch_size, hidden_dim]
        # cell: [1, batch_size, hidden_dim]

        # Pass through fully connected layer
        output = self.fc(output.squeeze(1))  # [batch_size, vocab_size]

        return output, hidden.squeeze(0), cell.squeeze(0), attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, question, answer, teacher_forcing_ratio=0.5):
        batch_size, max_length = answer.shape
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, max_length, vocab_size).to(device)
        encoder_outputs, hidden, cell = self.encoder(question)

        # First input to the decoder is the <sos> token
        input = answer[:, 0]

        for t in range(1, max_length):
            output, hidden, cell, attn_weights = self.decoder(
                input, hidden, cell, encoder_outputs)
            outputs[:, t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
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
    attention = Attention(hidden_dim)
    decoder = DecoderWithAttention(
        len(word2idx), embedding_dim, hidden_dim, attention)
    model = Seq2Seq(encoder, decoder).to(device)
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from:", model_path)
else:
    # Instantiate a new model
    print("No saved model found. Training a new model...")
    encoder = Encoder(len(word2idx), embedding_dim, hidden_dim)
    attention = Attention(hidden_dim)
    decoder = DecoderWithAttention(
        len(word2idx), embedding_dim, hidden_dim, attention)
    model = Seq2Seq(encoder, decoder).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter('runs')

    def print_welcome_message():
        print("************************************")
        print("*      Welcome to Small Chat!      *")
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
        encoder_outputs, hidden, cell = model.encoder(question_tokenized)
        input = torch.tensor([word2idx['<sos>']]).to(device)
        response = []

        for _ in range(max_response_length):
            output, hidden, cell, attn_weights = model.decoder(
                input, hidden, cell, encoder_outputs)
            predicted_word_idx = output.argmax(1).item()
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
