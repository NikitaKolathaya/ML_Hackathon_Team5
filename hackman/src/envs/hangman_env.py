# src/envs/hangman_env.py
import random
import numpy as np

class HangmanEnv:
    def __init__(self, word_list, max_wrong=6, max_len=15):
        self.word_list = word_list
        self.max_wrong = max_wrong
        self.max_len = max_len
        self.reset()

    def sample_word(self):
        return random.choice(self.word_list)

    def reset(self, word=None):
        self.word = word or self.sample_word()
        self.word = self.word.lower()
        self.L = len(self.word)
        self.mask = ['_'] * self.L
        self.guessed = set()
        self.wrong = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        # masked word padded to max_len as ints (0=pad, 1='a' ... 26='z', 27='_')
        enc = np.zeros(self.max_len, dtype=np.int32)
        for i in range(self.max_len):
            if i < self.L:
                c = self.mask[i]
                if c == '_':
                    enc[i] = 27
                else:
                    enc[i] = ord(c) - ord('a') + 1
            else:
                enc[i] = 0
        guessed_vec = np.zeros(26, dtype=np.float32)
        for ch in self.guessed:
            guessed_vec[ord(ch)-97] = 1.0
        lives = np.array([self.max_wrong - self.wrong], dtype=np.float32) / self.max_wrong
        return {
            'masked': enc,
            'guessed': guessed_vec,
            'lives': lives,
            'pattern': ''.join(self.mask)  # convenient for HMM
        }

    def step(self, action_letter):
        """
        action_letter: 'a'..'z' (string) OR int 0..25
        returns state, reward, done, info
        """
        if isinstance(action_letter, int):
            action_letter = chr(action_letter + 97)
        info = {}
        if self.done:
            return self._get_state(), 0.0, True, info

        reward = 0.0
        if action_letter in self.guessed:
            reward -= 0.5
            info['repeat'] = True
        else:
            self.guessed.add(action_letter)
            if action_letter in self.word:
                # reveal
                for i, ch in enumerate(self.word):
                    if ch == action_letter:
                        self.mask[i] = action_letter
                reward += 0.5
            else:
                self.wrong += 1
                reward -= 1.0

        if '_' not in self.mask:
            reward += 10.0
            self.done = True
            info['win'] = True
        elif self.wrong >= self.max_wrong:
            reward -= 5.0
            self.done = True
            info['win'] = False

        return self._get_state(), reward, self.done, info
