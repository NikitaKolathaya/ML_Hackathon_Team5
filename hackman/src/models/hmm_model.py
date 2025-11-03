# src/models/hmm_model.py
import math
from collections import defaultdict, Counter

class HMMModel:
    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.trans_counts = defaultdict(Counter)
        self.pos_counts = defaultdict(lambda: defaultdict(Counter))
        self.unigram = Counter()
        self.word_list = []
        self.max_len = 0

    def train(self, corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                w = line.strip().lower()
                if not w: 
                    continue
                self.word_list.append(w)
                self.max_len = max(self.max_len, len(w))
                prev = '^'
                self.unigram.update(w)
                for i, ch in enumerate(w):
                    self.trans_counts[prev][ch] += 1
                    self.pos_counts[len(w)][i][ch] += 1
                    prev = ch
                self.trans_counts[prev]['$'] += 1

        # compute normalized probabilities with Laplace smoothing
        self.trans_prob = {}
        for prev, cnts in self.trans_counts.items():
            total = sum(cnts.values()) + self.alpha * 27  # 26 letters + end
            self.trans_prob[prev] = {ch: (cnt + self.alpha) / total for ch, cnt in cnts.items()}
            # ensure unseen letters get alpha:
            for c in "abcdefghijklmnopqrstuvwxyz$":
                if c not in self.trans_prob[prev]:
                    self.trans_prob[prev][c] = self.alpha / total

        # positional probabilities with smoothing
        self.pos_prob = {}
        for L, posdict in self.pos_counts.items():
            self.pos_prob[L] = {}
            for pos, cnts in posdict.items():
                total = sum(cnts.values()) + self.alpha * 26
                self.pos_prob[L][pos] = {ch: (cnt + self.alpha) / total for ch, cnt in cnts.items()}
                for c in "abcdefghijklmnopqrstuvwxyz":
                    if c not in self.pos_prob[L][pos]:
                        self.pos_prob[L][pos][c] = self.alpha / total

    def score_word(self, word):
        # log-prob under simple transition model
        s = 0.0
        prev = '^'
        for ch in word:
            p = self.trans_prob.get(prev, {}).get(ch, 1e-12)
            s += math.log(p)
            prev = ch
        s += math.log(self.trans_prob.get(prev, {}).get('$', 1e-12))
        return s

    def candidates(self, pattern, guessed):
        """
        pattern: string like '_ppl_'
        guessed: set of letters guessed
        returns list of (word, score)
        """
        pattern = pattern.lower()
        L = len(pattern)
        cands = []
        for w in self.word_list:
            if len(w) != L: continue
            ok = True
            for pc, wc in zip(pattern, w):
                if pc != '_' and pc != wc:
                    ok = False
                    break
                if pc == '_' and wc in guessed:
                    # can't place an already-guessed letter in unknown position
                    ok = False
                    break
            if ok:
                cands.append(w)
        scored = [(w, self.score_word(w)) for w in cands]
        # sort descending by score (higher = better)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def letter_distribution(self, pattern, guessed, top_k=None):
        scored = self.candidates(pattern, guessed)
        if not scored:
            # fallback: use unigram distribution ignoring guessed
            total = sum(self.unigram.values())
            base = {c: self.unigram[c]/total for c in self.unigram}
            dist = [base.get(chr(ord('a')+i), 0.0) for i in range(26)]
            for i,c in enumerate(dist):
                if chr(ord('a')+i) in guessed:
                    dist[i] = 0.0
            s = sum(dist)
            if s==0:
                return [1/26.0 if chr(ord('a')+i) not in guessed else 0.0 for i in range(26)]
            return [d/s for d in dist]

        # weight candidates by exp(score - max_score) for stability
        scores = [s for _, s in scored]
        maxs = max(scores)
        weights = [math.exp(s - maxs) for s in scores]  # softmax-ish
        letter_total = Counter()
        for (w, _), wgt in zip(scored, weights):
            for i,ch in enumerate(w):
                if pattern[i] == '_' and ch not in guessed:
                    letter_total[ch] += wgt

        # create distribution
        dist = []
        total = sum(letter_total.values())
        for i in range(26):
            ch = chr(ord('a') + i)
            if ch in guessed:
                dist.append(0.0)
            else:
                dist.append(letter_total.get(ch, 0.0)/ (total if total>0 else 1.0))
        # if total==0 (no info), return uniform over unseen letters
        if total == 0:
            unseen = [i for i in range(26) if chr(ord('a')+i) not in guessed]
            if not unseen:
                return [0.0]*26
            u = 1.0/len(unseen)
            return [u if i in unseen else 0.0 for i in range(26)]
        return dist
