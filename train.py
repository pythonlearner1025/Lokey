import os
import sys

# Add mlx-examples/llms to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "mlx-examples/llms"))

from mlx_lm import load, generate
from kuhn_poker_env import KuhnPokerEnv
from prompts import ser_llm_input, generate_comparison_samples, get_preferences
import mlx.core as mx
import mlx.optimizers.optimizers as optim

def generate_comparison_samples(model, game_states, batch_size):
    def sample_with_temp(state, temp):
        # Generate two different outputs for same input with different temps
        return {
            'high_temp': model.generate(state, temperature=0.9),
            'low_temp': model.generate(state, temperature=0.7)
        }
    
    samples = []
    for state in game_states[:batch_size]:
        responses = sample_with_temp(state, temp=0.8)
        samples.append({
            'state': state,
            'response_a': responses['high_temp'],
            'response_b': responses['low_temp']
        })
    return samples

class DPOTrainer:
    def __init__(self, model, ref_model, beta=0.1):
        self.model = model
        self.ref_model = ref_model  # Reference model (usually frozen)
        self.beta = beta  # Temperature parameter
        
    def compute_dpo_loss(self, chosen, rejected, states):
        # Get log probs from current model
        chosen_logprobs = self.model.forward(states, chosen)
        rejected_logprobs = self.model.forward(states, rejected)
        
        # Get log probs from reference model
        ref_chosen_logprobs = self.ref_model.forward(states, chosen)
        ref_rejected_logprobs = self.ref_model.forward(states, rejected)
        
        # Compute reward difference
        chosen_rewards = chosen_logprobs - ref_chosen_logprobs
        rejected_rewards = rejected_logprobs - ref_rejected_logprobs
        
        # DPO loss
        loss = -mx.log(
            mx.sigmoid(self.beta * (chosen_rewards - rejected_rewards))
        ).mean()
        
        return loss
    
    def train_step(self, preferences, optimizer):
        """Single DPO training step"""
        optimizer.zero_grad()
        
        # Prepare batches
        states = [p['sample']['state'] for p in preferences]
        chosen = [p['sample'][f'response_{p["winner"].lower()}'] for p in preferences]
        rejected = [p['sample'][f'response_{("b" if p["winner"]=="A" else "a")}'] 
                   for p in preferences]
        
        # Compute and backprop loss
        loss = self.compute_dpo_loss(chosen, rejected, states)
        loss.backward()
        optimizer.step()
        
        return loss.item()

def train_with_dpo(model, ref_model, game_data, num_steps):
    trainer = DPOTrainer(model, ref_model)
    optimizer = optim.Adam(model.parameters())
    
    for step in range(num_steps):
        # Generate samples
        samples = generate_comparison_samples(model, game_data, batch_size=32)
        
        # Get preferences
        preferences = get_preferences(samples)
        
        # Train step
        loss = trainer.train_step(preferences, optimizer)
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")

class Agent:
    def __init__(self, llm, position):
        model, tokenizer = llm
        self.model = model
        self.tokenizer = tokenizer
        self.position = position
        
    def act(self, game_state, private_messages):
        # Serialize state + messages into context
        context = ser_llm_input(game_state, private_messages)
        messages = [{"role": "user", "content": context}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Get LLM generation
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt,
            max_tokens=512
        )
        return messages

def test_model():
    model, tokenizer = load(
        #'Qwen/Qwen2-7B-Instruct-MLX',
        'Qwen/Qwen2-1.5B-Instruct-MLX',
        tokenizer_config={"eos_token": "<|im_end|>", "trust_remote_code": True},
    )

    print(dir(model))

    prompt = "You are playing heads up poker. You have the hand AA. The pot has 10bb. The opponent bets 30bb. It is your turn - the turn is preflop. What is your action?"

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    response = generate(model, tokenizer, prompt=prompt, verbose=True, top_p=0.8, temp=0.7, repetition_penalty=1.05, max_tokens=512)

def train():
    env = KuhnPokerEnv(n_players=3)
    obs = env.reset()
    print(env.ser())


if __name__ == '__main__':
    train()