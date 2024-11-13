# Lokey

Lokey is a project designed to study how poker-playing LLM's might influence each others' decisions if allowed to communicate and collude amongst each other.

# Training LLMs for Multi-Agent Poker with Natural Collusion

This project explores training LLMs to play poker while allowing private communication between agents. We combine counterfactual regret minimization (CFR) with preference learning via Direct Preference Optimization (DPO) to develop strong, natural-looking strategies.

## Reward Structure

### 1. Hard Rewards (RH) via CFR
- Direct poker winnings/losses computed from game outcomes
- Counterfactual regret values tracked for each decision point
- Regret matching used to convert regrets to action probabilities
- Implementation:
  ```python
  strategy = {
      action: max(0, regret)/total_regret 
      for action, regret in regrets.items()
  }
  ```

### 2. Naturalness Rewards (RN)
- Evaluates how human-like the LLM outputs are
- DPO training with preferences from:
  - Advanced LLM critics
- Considers:
  - Language patterns
  - Betting patterns
  - Quality of reasoning

### 3. Collaboration Rewards (RC)
- Measures effectiveness and subtlety of coordination
- Also trained using DPO preference learning
- Evaluates:
  - Implicit signaling
  - Natural table talk
  - Complementary actions

## Regret-Aware DPO Training

### Core Components

1. Preference Scoring
```python
combined_score = style_score * exp(regret_score)
# OR
combined_score = w1*style_score + w2*regret_score
```

2. Modified DPO Loss
```python
reward_diff = (chosen_rewards - rejected_rewards) * regret_diffs
loss = -log(sigmoid(beta * reward_diff)).mean()
```

### Training Process

1. CFR Updates
- Track regrets for all states and actions
- Update regret values based on game outcomes
- Convert to strategies via regret matching

2. DPO Training Step
- Generate action/communication samples
- Score using combined preference function
- Update model to prefer high-regret, natural actions

3. Integration Methods
- Regret-weighted preference scoring
- Scaled DPO rewards based on regret differences
- Curriculum from pure CFR to style-aware training

## Implementation Details

### Regret-Aware Preference Trainer
```python
class RegretAwareDPOTrainer:
    def get_combined_preferences(self, samples):
        # Combine style and regret preferences
        style_score = critique_model.score(response)
        regret_score = compute_regret_weighted_preference(state, actions)
        return combine_scores(style_score, regret_score)
```

### Training Loop
1. Update CFR solver with new game data
2. Generate paired samples
3. Score using combined preference function
4. Train using regret-aware DPO loss

## Key Features

### Preference Balance
- Natural tension between optimal play and style
- Regret values guide exploration of action space
- Style preferences maintain human-like outputs

### Exploration Strategy
- Temperature sampling for diverse actions
- Regret matching provides exploration bias
- Progressive annealing of exploration

### Performance Metrics
- Poker winning rate
- Naturalness scores
- Collaboration effectiveness
- Regret minimization progress

## Next Steps
- [ ] Implement regret tracking system
- [ ] Develop preference combination methods
- [ ] Create curriculum for training phases
- [ ] Evaluate emergence of sophisticated strategies