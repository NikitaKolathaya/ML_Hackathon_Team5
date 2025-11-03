import os, sys, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from datetime import datetime
from tqdm import trange
from src.models.hmm_model import HMMModel
from src.envs.hangman_env import HangmanEnv
from src.models.dqn_agent import DQNAgent
from src.utils import make_state_vector, eval_agent


def main():
    # Clear screen for clean start
    os.system('cls' if os.name == 'nt' else 'clear')

    print("🤖 Starting DQN (RL) Training")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("──────────────────────────────────────────────")

    # Base paths
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    corpus = os.path.join(base, 'data', 'corpus.txt')

    # --------------------------
    # 1️⃣ Train HMM on corpus
    # --------------------------
    hmm = HMMModel(alpha=1e-3)
    print(f"📘 Training HMM on: {corpus}")
    hmm.train(corpus)
    print(f"✅ HMM trained on {len(hmm.word_list)} training words.")

    # Setup environment
    env = HangmanEnv(hmm.word_list, max_wrong=6)

    # Compute input dimension
    example_state = env.reset()
    dummy_probs = np.zeros(26, dtype=np.float32)
    state_dim = make_state_vector(example_state, dummy_probs).shape[0]
    print(f"🧠 State dimension (input size): {state_dim}")
    print("──────────────────────────────────────────────\n")

    # --------------------------
    # 2️⃣ Setup DQN agent
    # --------------------------
    agent = DQNAgent(state_dim, lr=1e-4)
    episodes = 3000
    batch_size = 64
    eps_start, eps_end, eps_decay = 1.0, 0.05, 40000
    eps = eps_start
    total_steps = 0

    save_dir = os.path.join(base, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 Training begins...\n")

    # --------------------------
    # 3️⃣ Training loop
    # --------------------------
    for ep in trange(episodes, desc="Training DQN", ncols=100):
        s = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            guessed_mask = s['guessed']
            probs = np.array(
                hmm.letter_distribution(
                    s['pattern'],
                    set([chr(97 + i) for i, v in enumerate(guessed_mask) if v > 0.5])
                )
            )
            vec = make_state_vector(s, probs)
            action = agent.select_action(vec, guessed_mask, eps)
            ns, r, done, info = env.step(action)

            guessed_mask_n = ns['guessed']
            probs_n = np.array(
                hmm.letter_distribution(
                    ns['pattern'],
                    set([chr(97 + i) for i, v in enumerate(guessed_mask_n) if v > 0.5])
                )
            )
            vec_n = make_state_vector(ns, probs_n)

            agent.store(vec, action, r, vec_n, float(done))
            agent.update(batch_size)

            s = ns
            ep_reward += r
            total_steps += 1
            eps = max(eps_end, eps_start - total_steps / eps_decay)

        # Target update
        if ep % 20 == 0:
            agent.update_target()

        # Save checkpoints
        if ep % 200 == 0:
            ckpt_path = os.path.join(save_dir, f'dqn_ep{ep}.pt')
            agent.save(ckpt_path)
            print(f"\n💾 Saved checkpoint → {ckpt_path}")

        # Periodic evaluation
        if ep % 100 == 0:
            stats = eval_agent(env, agent, hmm=hmm, n_games=500)
            print(f"\n📊 Episode {ep} | success={stats['success_rate']:.3f} | wins={stats['wins']} | wrong={stats['total_wrong']} | eps={eps:.3f}")
            print("──────────────────────────────────────────────")

    # --------------------------
    # 4️⃣ Save final model
    # --------------------------
    agent.save(os.path.join(save_dir, 'dqn_final.pt'))
    print("\n✅ Training complete.")
    print("🏁 Final model saved → checkpoints/dqn_final.pt")
    print("──────────────────────────────────────────────")


if __name__ == "__main__":
    main()
