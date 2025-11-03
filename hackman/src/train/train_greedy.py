import os
import sys
from datetime import datetime

# allow importing src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.hmm_model import HMMModel
from src.models.greedy_agent import GreedyAgent
from src.envs.hangman_env import HangmanEnv
from src.utils import eval_agent


def main():
    # clear screen for clean output
    os.system('cls' if os.name == 'nt' else 'clear')

    print("🔹 Starting Greedy (HMM) Evaluation Run")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("──────────────────────────────────────────────")

    # base paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    corpus_path = os.path.join(base_dir, 'data', 'corpus.txt')
    test_path = os.path.join(base_dir, 'data', 'test.txt')

    # --------------------------
    # 1️⃣ Train HMM on corpus
    # --------------------------
    hmm = HMMModel(alpha=1e-3)
    print(f"📘 Training HMM on {corpus_path}")
    hmm.train(corpus_path)
    print(f"✅ HMM trained on {len(hmm.word_list)} training words.")

    # --------------------------
    # 2️⃣ Load test set
    # --------------------------
    if os.path.exists(test_path):
        print(f"🧩 Loading test dataset from: {test_path}")
        with open(test_path, "r") as f:
            test_words = [w.strip() for w in f.readlines() if w.strip()]
        print(f"📚 Loaded {len(test_words)} test words for evaluation.")
    else:
        print("⚠️ No test.txt found! Using training words for evaluation.")
        test_words = hmm.word_list

    # --------------------------
    # 3️⃣ Setup environment and agent
    # --------------------------
    env = HangmanEnv(test_words, max_wrong=6)
    greedy = GreedyAgent(hmm)

    # --------------------------
    # 4️⃣ Evaluate Greedy agent
    # --------------------------
    n_games = 2000
    print(f"🎮 Evaluating Greedy agent on {n_games} test games...\n")

    stats = eval_agent(env, greedy, hmm=hmm, n_games=n_games)

    # --------------------------
    # 5️⃣ Compute final metrics
    # --------------------------
    wins = stats["wins"]
    success_rate = stats["success_rate"]
    total_wrong = stats["total_wrong"]
    total_repeated = 0  # greedy never repeats

    final_score = (success_rate * n_games) - (5 * total_wrong) - (2 * total_repeated)

    # --------------------------
    # 6️⃣ Summary
    # --------------------------
    print("\n──────────────────────────────────────────────")
    print("📊 GREEDY AGENT EVALUATION SUMMARY (Test Data)")
    print("──────────────────────────────────────────────")
    print(f"✅ Wins: {wins} / {n_games}")
    print(f"🎯 Success Rate: {success_rate * 100:.2f}%")
    print(f"❌ Total Wrong Guesses: {total_wrong}")
    print(f"🔁 Total Repeated Guesses: {total_repeated}")
    print(f"🏆 Final Score (with -5 per wrong): {final_score:.2f}")
    print("🧩 Evaluation complete!")
    print("──────────────────────────────────────────────\n")

    return stats, final_score


# prevent double runs
if __name__ == "__main__" and "RUNNING_GREEDY" not in globals():
    RUNNING_GREEDY = True
    main()
