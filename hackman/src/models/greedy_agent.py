# src/models/greedy_agent.py
import numpy as np

class GreedyAgent:
    def __init__(self, hmm_model):
        self.hmm = hmm_model

    def act(self, state):
        pattern = state['pattern']
        guessed_vec = state['guessed']
        guessed = set()
        for i,v in enumerate(guessed_vec):
            if v>0.5:
                guessed.add(chr(97+i))
        probs = self.hmm.letter_distribution(pattern, guessed)
        # pick highest-prob unguessed
        probs = np.array(probs)
        if probs.sum() <= 0:
            # fallback: pick random unseen letter
            unseen = [i for i in range(26) if chr(97+i) not in guessed]
            return chr(97 + (unseen[0] if unseen else 0))
        idx = int(np.argmax(probs))
        return chr(97 + idx)
