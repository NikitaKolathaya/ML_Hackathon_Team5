# 🧠 Hangman AI: HMM-Driven Q-Learning Agent

A Reinforcement Learning approach to solving the Hangman game using Hidden Markov Models (HMM) and Tabular Q-Learning.

### 📘 Overview



This project combines probabilistic modeling (HMM) and reinforcement learning (Q-Learning) to create an AI agent that can intelligently play and learn the Hangman word-guessing game.

The system learns letter guessing strategies based on observed rewards and uses an HMM to model the probability distribution of letters and word patterns — allowing the agent to learn how English-like words behave.



### 🎯 Objectives



Design an HMM-based oracle to model letter and word distributions.

Implement a Q-Learning agent that interacts with a Hangman environment.

Train the agent to maximize guessing accuracy and minimize wrong guesses.

Visualize letter patterns, training metrics, and learning performance.

### 

### 🧩 System Components

##### 🧮 1. Hidden Markov Model (HMM) Oracle



Learns initial letter probabilities and transition probabilities between letters.

Generates synthetic words and letter distributions similar to natural language.

Provides insights into linguistic structure used by the RL agent.

Visualization:

Key Insights:

Some letters (like s, p, a, r, t) appear frequently at the start or middle of words.

Transition heatmaps show probable letter pairs (like th, er, on).

Word length distribution follows a natural language-like bell curve (avg length ≈ 8–10).



##### 🕹️ 2. Hangman Environment

Simulates the game logic: correct guesses, wrong guesses, remaining tries.

Provides rewards for correct actions and penalties for mistakes.

Used as the environment for the reinforcement learning agent.

##### 

##### 🧠 3. Q-Learning Agent

Learns from trial and error using state-action-reward feedback.

Adopts an epsilon-greedy policy for exploration vs exploitation.

Uses a feature extractor to encode partial word patterns and guesses.

Updates Q-values iteratively to improve performance across episodes.

##### 🔍 4. Feature Extractor

Converts he Hangman game state into numeric features:

Revealed letters

Count of wrong guesses

Remaining guesses

Letter position patterns

Enables the agent to generalize across similar word structures.

### 📊 Training and Evaluation Results

After training the Q-learning agent over 5000 episodes, the following metrics were recorded:

Key Observations:

Metric	Description	Trend

Average Reward per Episode	Mean score over time	Gradual improvement and stabilization after ~1500 episodes

Success Rate	Fraction of games won	Steadily rises and stabilizes between 35–40%

Average Wrong Guesses	Mean incorrect guesses per game	Decreases from ~6 to ~5

Exploration Rate (Epsilon)	Agent’s randomness	Decays smoothly, allowing convergence

### 

### 🚀 Future Work

Upgrade to Deep Q-Learning (DQN) for complex word spaces

Integrate Transformer-based language modeling

Create a web-based Hangman game vs. AI

Expand dataset for multilingual support🧠 Hangman AI: HMM-Driven Q-Learning Agent

A Reinforcement Learning approach to solving the Hangman game using Hidden Markov Models (HMM) and Tabular Q-Learning.

### 🧩 System Components

🧮 1. Hidden Markov Model (HMM) Oracle



Learns initial letter probabilities and transition probabilities between letters.

Generates synthetic words and letter distributions similar to natural language.

Provides insights into linguistic structure used by the RL agent.

Visualization:

Key Insights:

Some letters (like s, p, a, r, t) appear frequently at the start or middle of words.

Transition heatmaps show probable letter pairs (like th, er, on).

Word length distribution follows a natural language-like bell curve (avg length ≈ 8–10).



🕹️ 2. Hangman Environment

Simulates the game logic: correct guesses, wrong guesses, remaining tries.

Provides rewards for correct actions and penalties for mistakes.

Used as the environment for the reinforcement learning agent.

🧠 3. Q-Learning Agent

Learns from trial and error using state-action-reward feedback.

Adopts an epsilon-greedy policy for exploration vs exploitation.

ses a feature extractor to encode partial word patterns and guesses.

Updates Q-values iteratively to improve performance across episodes.





🔍 4. Feature Extractor



Converts the Hangman game state into numeric features:

Revealed letters

Count of wrong guesses

Remaining guesses

Letter position patterns

Enables the agent to generalize across similar word structures.



### 📊 Training and Evaluation Results



After training the Q-learning agent over 5000 episodes, the following metrics were recorded:



Key Observations:

Metric	Description	Trend

Average Reward per Episode	Mean score over time	Gradual improvement and stabilization after ~1500 episodes

Success Rate	Fraction of games won	Steadily rises and stabilizes between 35–40%

Average Wrong Guesses	Mean incorrect guesses per game	Decreases from ~6 to ~5

Exploration Rate (Epsilon)	Agent’s randomness	Decays smoothly, allowing convergence

### 🧩 Workflow Summary

Step	Description

1️⃣	HMM Oracle generates words and letter patterns

2️⃣	Agent interacts with Hangman environment

3️⃣	Rewards assigned for correct guesses, penalties for wrong ones

4️⃣	Q-Table updated using RL rule

5️⃣	Exploration rate gradually decays

6️⃣	Agent improves guessing efficiency over episodes

### ⚙️ Setup and Execution

Requirements

Python >= 3.9

numpy

matplotlib

pickle

random

Run all cells sequentially to train and visualize.



### 🧠 Learnings and Insights

Reinforcement learning can effectively learn pattern-based word guessing.

HMM provides realistic probabilistic context that improves training quality.

Combining probability (HMM) and learning (Q-Learning) yields strong generalization even without deep networks.

Epsilon decay balances exploration for faster convergence.

### 

### 🏆 Hackathon Impact

This project showcases an AI system that:

Learns without labeled supervision.

Adapts to new words dynamically.

Demonstrates intelligent decision-making using only rewards.

Combines multiple AI paradigms — NLP, Probability, and Reinforcement Learning.



### 👩‍💻 Team



Project: ML Hackathon – Hangman AI (HMM + Q-Learning)

Team Members:TEAM -05

Naman Nagar     		PES2UG23CS361

Namritha Diya Lobo		PES2UG23CS362

Nikita K			PES2UG23CS387		

Nandana Mathew			PES2UG23CS913



