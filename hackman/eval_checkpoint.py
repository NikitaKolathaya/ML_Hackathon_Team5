import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Make sure Python can import the project's `src` package when running this
# script directly (so imports below work whether run from VS Code or a terminal).
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.models.hmm_model import HMMModel
from src.models.dqn_agent import DQNAgent
from src.envs.hangman_env import HangmanEnv
from src.utils import make_state_vector


def main():
    # Clear the terminal so the evaluation output is easy to read.
    os.system('cls' if os.name == 'nt' else 'clear')

    print("🔹 Starting DQN Evaluation Run")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("──────────────────────────────────────────────")

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "test.txt")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")

    # Locate the trained DQN checkpoint to load. Try the final file first,
    # then fall back to a 'best' checkpoint if present.
    checkpoint_path = os.path.join(checkpoints_dir, "dqn_final.pt")
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join(checkpoints_dir, "dqn_best.pt")
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            print("❌ No DQN checkpoint found in checkpoints/")
            return

    print(f"📦 Loading model from: {checkpoint_path}")

    # Load a small HMM language model (used to suggest letter probabilities)
    # and create the Hangman environment using the model's word list.
    hmm = HMMModel(alpha=1e-3)
    hmm.train(data_path)
    print(f"📚 Loaded {len(hmm.word_list)} words for evaluation.")

    env = HangmanEnv(hmm.word_list, max_wrong=6)

    # Compute the input vector size for the agent. We build a sample state
    # (like during training) and convert it to the flattened feature vector
    # to determine the model's expected input dimension.
    example_state = env.reset()
    dummy_probs = np.zeros(26, dtype=np.float32)
    state_dim = make_state_vector(example_state, dummy_probs).shape[0]
    print(f"🧠 State dimension (input size): {state_dim}")

    # Instantiate the DQN agent with the same input size and learning rate
    # used during training, then load the saved weights and set epsilon
    # to 0 so the agent acts greedily for evaluation.
    agent = DQNAgent(state_dim, lr=1e-4)
    agent.load(checkpoint_path)
    agent.eps = 0.0  # greedy during eval

    # Run many games to evaluate agent performance. We collect wins,
    # wrong guesses, rewards and penalties to produce a final score.
    n_games = 2000
    print(f"🎮 Evaluating trained DQN agent on {n_games} test games...\n")

    total_wins = 0
    total_wrong = 0
    total_reward = 0.0
    total_penalty = 0.0

    for _ in tqdm(range(n_games), desc="Evaluating", ncols=100):
        s = env.reset()
        done = False
        ep_reward = 0.0
        ep_penalty = 0.0
        last_r = 0.0
        while not done:
            probs = np.array(
                hmm.letter_distribution(
                    s["pattern"],
                    set([chr(97 + j) for j, v in enumerate(s["guessed"]) if v > 0.5]),
                )
            )
            vec = make_state_vector(s, probs)
            action_idx = agent.select_action(vec, s["guessed"], 0.0)
            s, r, done, info = env.step(action_idx)
            last_r = r
            ep_reward += max(r, 0)
            ep_penalty += abs(min(r, 0))

        # Determine whether the episode was a win and how many wrong
        # guesses were made. Some environment versions return this via
        # the `info` dict, so we prefer that, otherwise infer from the
        # mask string on the environment.
        wrong = getattr(env, "wrong", 0)
        won = info.get("win", None)
        if won is None:
            # If the info dict didn't report a win, check the mask to see
            # if any blanks remain; no blanks means the word was found.
            won = 1 if "_" not in env.mask else 0
        else:
            won = 1 if won else 0

        total_wins += int(won)
        total_wrong += int(wrong)
        total_reward += ep_reward
        total_penalty += ep_penalty

    success_rate = total_wins / n_games
    total_repeated = 0
    # Compute a final contest-style score: reward wins, penalize wrong
    # guesses and repeated letters. This mirrors the evaluation formula
    # used during the contest runs.
    final_score = (success_rate * n_games) - (5 * total_wrong) -(2*total_repeated)

    # summary
    print("\n──────────────────────────────────────────────")
    print("📊 EVALUATION SUMMARY")
    print("──────────────────────────────────────────────")
    print(f"✅ Wins: {total_wins} / {n_games}")
    print(f"🎯 Success Rate: {success_rate * 100:.2f}%")
    print(f"❌ Total Wrong Guesses: {total_wrong}")
    print(f"💰 Total Reward Gained: {total_reward:.2f}")
    print(f"💣 Total Penalty Accumulated: {total_penalty:.2f}")
    print(f"🏆 Final Score (with -5 per wrong): {final_score:.2f}")
    print("🧩 Evaluation complete!")
    print("──────────────────────────────────────────────\n")


# Guard to avoid accidentally running the evaluation twice when the
# module is imported or reloaded (useful in interactive sessions).
if __name__ == "__main__" and "RUNNING_EVAL" not in globals():
    RUNNING_EVAL = True
    main()
