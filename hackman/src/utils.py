# src/utils.py
import numpy as np

def make_state_vector(state, hmm_probs=None, max_len=15):
    """
    state: dict from env._get_state()
    returns a flat float32 vector for NN input
    components: masked (as ints 0..27) -> one-hot embeddings or normalized ints
    For simplicity: use normalized ints + guessed vector + hmm_probs + lives
    """
    masked = state['masked'].astype(np.float32) / 27.0  # normalize
    guessed = state['guessed'].astype(np.float32)
    lives = state['lives'].astype(np.float32)
    if hmm_probs is None:
        hmm_probs = np.zeros(26, dtype=np.float32)
    vec = np.concatenate([masked, guessed, hmm_probs, lives])
    return vec

def eval_agent(env, agent, hmm=None, n_games=2000):
    wins = 0
    total_wrong = 0
    total_repeats = 0
    for _ in range(n_games):
        s = env.reset()
        done = False
        while not done:
            if isinstance(agent, type) and agent == 'random':
                # not used
                pass
            if hasattr(agent, 'act'):
                # greedy agent
                action = agent.act(s)
                action_idx = ord(action)-97
            else:
                # assume DQN agent expects vector
                guessed_mask = s['guessed']
                if hmm:
                    probs = np.array(hmm.letter_distribution(s['pattern'], set([chr(i+97) for i,v in enumerate(s['guessed']) if v>0.5])))
                else:
                    probs = np.zeros(26, dtype=np.float32)
                vec = make_state_vector(s, probs)
                action_idx = agent.select_action(vec, guessed_mask, eps=0.0)
            ns, r, done, info = env.step(action_idx)
            s = ns
        if info.get('win', False):
            wins += 1
        # compute wrong and repeat guesses heuristically:
        # guessed letters not in word = wrong
        # repeats tracked in reward? Not easily here; we'll skip exact repeat count
        total_wrong += sum(1 for ch in env.guessed if ch not in env.word)
    return {
        'wins': wins,
        'success_rate': wins / n_games,
        'total_wrong': total_wrong
    }
