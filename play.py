import torch
from main import WordleGame, WordleTransformer, create_vocabulary, tensor_to_word

def play_interactive_wordle(game):
    """
    Play an interactive game of Wordle where the user provides feedback for the model's guesses.

    Args:
        game: WordleGame instance
    """
    guesses = []
    feedbacks = []

    print("\nWelcome to Interactive Wordle!")
    print("For each guess, enter feedback as 5 numbers:")
    print("0: Letter is not in the word (gray)")
    print("1: Letter is in the word but wrong position (yellow)")
    print("2: Letter is in the correct position (green)")
    print("Example: '02102' means: gray, green, yellow, gray, green\n")

    for turn in range(6):  # Maximum 6 guesses
        # Convert previous guesses and feedbacks to tensors
        if guesses:
            guess_tensor = torch.stack(guesses).to(game.device)
            feedback_tensor = torch.stack(feedbacks).to(game.device)
        else:
            guess_tensor = torch.zeros((0, 5), dtype=torch.long, device=game.device)
            feedback_tensor = torch.zeros((0, 5), dtype=torch.long, device=game.device)

        # Get model's guess
        new_guess = game.make_guess(guess_tensor, feedback_tensor)
        guess_word = tensor_to_word(new_guess, game.vocab)
        print(f"\nGuess {turn + 1}: {guess_word}")

        # Get feedback from user
        while True:
            try:
                feedback_str = input("Enter feedback (5 digits of 0/1/2): ").strip()
                if len(feedback_str) != 5 or not all(c in '012' for c in feedback_str):
                    raise ValueError
                feedback_list = [int(c) for c in feedback_str]
                break
            except ValueError:
                print("Invalid input! Please enter exactly 5 digits, each being 0, 1, or 2.")

        # Convert feedback to tensor
        new_feedback = torch.tensor(feedback_list, dtype=torch.long, device=game.device)

        # Store guess and feedback
        guesses.append(new_guess)
        feedbacks.append(new_feedback)

        # Check if solved
        if torch.all(new_feedback == 2):
            print(f"\nCongratulations! The word was solved in {turn + 1} guesses!")
            return True, turn + 1

        if turn == 5:
            print("\nGame over! Maximum guesses reached.")
            return False, 6

# Read the list of words from the file.
with open("words.txt", 'r') as file:
        word_list = [line.strip() for line in file.readlines()]
vocab = create_vocabulary(word_list)

# Set the device to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from the checkpoint.
model = WordleTransformer(26, word_list, vocab)
model.load_state_dict(torch.load('best_model.pt'))
model = model.to(device)

# Example usage:
game = WordleGame(model, vocab, device)
play_interactive_wordle(game)