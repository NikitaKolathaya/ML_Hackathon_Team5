# hackman

Project scaffold for a Hangman RL project.

Structure:

- data/
  - corpus.txt         # place your corpus here (one word/phrase per line)
- src/
  - envs/
    - hangman_env.py   # hangman environment (placeholder)
  - models/
    - hmm_model.py     # placeholder for an HMM model
    - greedy_agent.py  # greedy baseline agent
    - dqn_agent.py     # DQN agent placeholder
  - train/
    - train_greedy.py  # runner for greedy agent
    - train_dqn.py     # trainer for DQN agent
  - utils.py           # helper functions

Requirements are listed in `requirements.txt`.

Next steps:
- Fill `data/corpus.txt` with words/phrases.
- Implement `HangmanEnv` with the desired API (Gym-style recommended).
- Implement models and training loops.

