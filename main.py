import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import Counter

class WordleTransformer(nn.Module):
    def __init__(self, vocab_size, word_list, vocab, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_list = word_list
        self.vocab = vocab
        # Create reverse vocab (index to char)
        self.idx_to_char = {idx: char for char, idx in vocab.items()}

        # Create word tensors for all valid words
        self.valid_words = torch.stack([word_to_tensor(word, vocab) for word in word_list])

        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(5, d_model)
        self.feedback_embedding = nn.Embedding(3, d_model)
        self.d_model = d_model

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)

        # Changed to output scores for each valid word instead of character predictions
        self.fc = nn.Linear(d_model * 5, len(word_list))

        self.register_buffer('positions', torch.arange(5))

    def forward(self, guesses, feedbacks):
        batch_size = guesses.shape[0]
        num_guesses = guesses.shape[1]

        if num_guesses == 0:
            x = torch.zeros(batch_size, 5, self.d_model, device=guesses.device)
            x = x + self.position_embedding(self.positions).unsqueeze(0)
        else:
            word_emb = self.word_embedding(guesses)
            pos_emb = self.position_embedding(self.positions)
            feedback_emb = self.feedback_embedding(feedbacks)

            x = word_emb + pos_emb.unsqueeze(0).unsqueeze(0) + feedback_emb
            x = x.view(batch_size, num_guesses * 5, -1)

        transformer_out = self.transformer(x)

        if num_guesses == 0:
            # Reshape transformer output to combine all position information
            output = transformer_out.view(batch_size, -1)
        else:
            # Take last 5 positions and reshape
            output = transformer_out[:, -5:, :].view(batch_size, -1)

        # Output scores for each valid word
        word_scores = self.fc(output)
        return word_scores

    def filter_valid_guesses(self, word_scores, previous_guesses, feedbacks):
        """Filter word scores based on feedback from previous guesses"""
        device = word_scores.device
        batch_size = word_scores.shape[0]
        num_words = len(self.word_list)

        # Create a mask for valid words
        valid_mask = torch.ones((batch_size, num_words), dtype=torch.bool, device=device)

        # Convert valid words to a list of tensors for easy comparison
        valid_words_tensor = self.valid_words.unsqueeze(0)  # Shape: (1, num_words, word_length)

        for b in range(batch_size):
            # Check against previous guesses
            for prev_guess in previous_guesses[b]:
                # Ensure prev_guess is correctly shaped
                prev_guess_tensor = prev_guess.unsqueeze(0)  # Shape: (1, word_length)

                # Move tensors to device
                prev_guess_tensor = prev_guess_tensor.to(device)
                valid_words_tensor = valid_words_tensor.to(device)

                # Create a boolean mask where the candidate words match the previous guess
                guess_mask = torch.all(valid_words_tensor == prev_guess_tensor, dim=-1)  # Shape: (num_words,)

                # Update valid_mask: if a word matches a previous guess, it's invalid
                valid_mask[b] = valid_mask[b] & ~guess_mask

            # Check against previous feedbacks
            for guess, feedback in zip(previous_guesses[b], feedbacks[b]):
                current_valid = valid_mask[b]

                for idx in range(num_words):
                    if current_valid[idx]:  # Only check valid candidates
                        if not self.is_consistent_with_feedback(self.valid_words[idx], guess, feedback):
                            valid_mask[b, idx] = False

        # Apply mask to scores
        word_scores[~valid_mask] = float('-inf')
        return word_scores

    def is_consistent_with_feedback(self, candidate, guess, feedback):
        """Check if a candidate word is consistent with the feedback from a previous guess"""
        expected_feedback = evaluate_guess(guess, candidate)
        return torch.all(expected_feedback == feedback)

def evaluate_guess(guess, target):
    """
    Evaluate a guess against the target word.
    Returns: tensor of feedback (0: wrong, 1: misplaced, 2: correct)
    """
    feedback = torch.zeros_like(guess)

    # 
    guess = guess.to(target.device)

    # Step 1: Identify correct positions
    correct_mask = (guess == target)  # Shape: (word_length,)
    feedback[correct_mask] = 2  # Mark correct positions

    # Create a mask to track which characters have been matched
    matched_target = target.clone()
    matched_target[correct_mask] = -1  # Use -1 to mark matched characters

    # Step 2: Count remaining target characters
    remaining_counts = Counter(matched_target[matched_target != -1].tolist())

    # Step 3: Identify misplaced characters
    for i, g in enumerate(guess):
        if feedback[i] == 0:  # Only check if not already correct
            if remaining_counts.get(g.item(), 0) > 0:  # Use .item() to get the integer value
                feedback[i] = 1  # Mark as misplaced
                remaining_counts[g.item()] -= 1  # Decrement the count

    return feedback

def create_vocabulary(words):
    """Create character-level vocabulary"""
    chars = sorted(set(''.join(words)))
    return {char: idx for idx, char in enumerate(chars)}

def word_to_tensor(word, vocab):
    """Convert a word to a tensor of character indices"""
    return torch.tensor([vocab[c] for c in word], dtype=torch.long)

def tensor_to_word(tensor, vocab):
    """Convert a tensor back to a word"""
    idx_to_char = {idx: char for char, idx in vocab.items()}
    return ''.join(idx_to_char[idx.item()] for idx in tensor)

class WordleGame:
    def __init__(self, model, vocab, device):
        self.model = model
        self.vocab = vocab
        self.device = device

    def make_guess(self, current_guesses, current_feedbacks):
        with torch.no_grad():
            # Get word scores from model
            word_scores = self.model(current_guesses.unsqueeze(0), current_feedbacks.unsqueeze(0))

            # Filter valid guesses based on feedback
            filtered_scores = self.model.filter_valid_guesses(
                word_scores,
                current_guesses.unsqueeze(0),
                current_feedbacks.unsqueeze(0)
            )

            # Select the highest scoring valid word
            word_idx = filtered_scores[0].argmax()
            return self.model.valid_words[word_idx]

    def play_game(self, target_word, verbose=True):
        target = word_to_tensor(target_word, self.vocab).to(self.device)
        guesses = []
        feedbacks = []

        if verbose:
            print(f"\nTarget word: {target_word}")

        for turn in range(6):  # Maximum 6 guesses
            # Convert previous guesses and feedbacks to tensors
            if guesses:
                guess_tensor = torch.stack(guesses).to(self.device)
                feedback_tensor = torch.stack(feedbacks).to(self.device)
            else:
                guess_tensor = torch.zeros((0, 5), dtype=torch.long, device=self.device)
                feedback_tensor = torch.zeros((0, 5), dtype=torch.long, device=self.device)

            # Make guess
            new_guess = self.make_guess(guess_tensor, feedback_tensor)
            new_feedback = evaluate_guess(new_guess, target)

            # Store guess and feedback
            guesses.append(new_guess)
            feedbacks.append(new_feedback)

            # Convert guess to word and print
            guess_word = tensor_to_word(new_guess, self.vocab)
            if verbose:
                print(f"Guess {turn + 1}: {guess_word}")

            # Check if solved
            if torch.all(new_feedback == 2):
                return True, turn + 1

        return False, 6

def train_with_gameplay(model, optimizer, train_words, batch_size, device, max_guesses=6):
    model.train()
    total_loss = 0
    # num_batches = len(train_words) // batch_size
    num_batches = 1
    criterion = nn.CrossEntropyLoss()

    # Mini-batch gradient descent.
    for i in range(num_batches):
        optimizer.zero_grad()

        batch_indices = torch.randperm(len(train_words))[:batch_size]
        batch_targets = train_words[batch_indices]

        batch_loss = 0

        # For each word in the batch
        for b in tqdm(range(batch_size), desc="Batch"):
            guesses = []
            feedbacks = []
            target = batch_targets[b]

            # Find target word index
            target_idx = None
            for idx, word in enumerate(model.valid_words):
                word = word.to(device)
                if torch.all(word == target):
                    target_idx = torch.tensor([idx], device=device)
                    break

            # Simulate gameplay for this word
            for guess_num in range(max_guesses):
                if guesses:
                    guess_tensor = torch.stack(guesses).unsqueeze(0).to(device)
                    feedback_tensor = torch.stack(feedbacks).unsqueeze(0).to(device)
                else:
                    guess_tensor = torch.zeros((1, 0, 5), dtype=torch.long, device=device)
                    feedback_tensor = torch.zeros((1, 0, 5), dtype=torch.long, device=device)

                # Get model's word scores
                word_scores = model(guess_tensor, feedback_tensor)

                # Filter valid guesses
                filtered_scores = model.filter_valid_guesses(
                    word_scores,
                    guess_tensor,
                    feedback_tensor
                )

                # Calculate loss using target word index
                batch_loss += criterion(filtered_scores[0].unsqueeze(0), target_idx)

                # Make guess
                guess_idx = filtered_scores[0].argmax()
                guess = model.valid_words[guess_idx]
                guess = guess.to(device)
                feedback = evaluate_guess(guess, target)

                guesses.append(guess)
                feedbacks.append(feedback)

                if torch.all(feedback == 2):
                    break

        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()

    return total_loss / num_batches

def main():
    # Hyperparameters
    num_epochs = 10
    batch_size = 8

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Sample word list
    # word_list = [
    #     "happy", "beach", "chair", "dance", "eagle",
    #     "flame", "grape", "house", "image", "juice",
    #     "knife", "light", "music", "night", "ocean"
    # ]

    # Load the words from a file.
    with open("words.txt", 'r') as file:
        word_list = [line.strip() for line in file.readlines()]

    vocab = create_vocabulary(word_list)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    model = WordleTransformer(vocab_size, word_list, vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    train_words = torch.stack([word_to_tensor(word, vocab) for word in word_list]).to(device)
    best_loss = float('inf')

    print("Starting training...")
    for epoch in range(num_epochs):
        loss = train_with_gameplay(model, optimizer, train_words, batch_size, device)
        scheduler.step(loss)

        # Save the model if the loss is the best we've seen so far.
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pt')

        # Print every epoch.
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

    model.load_state_dict(torch.load('best_model.pt'))

    print("\nTesting the model...")
    game = WordleGame(model, vocab, device)

    test_words = ["happy", "beach", "chair"]
    for word in test_words:
        solved, num_guesses = game.play_game(word)
        result = "solved" if solved else "failed to solve"
        print(f"Word '{word}' {result} in {num_guesses} guesses")

if __name__ == "__main__":
    main()