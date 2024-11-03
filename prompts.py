def ser_llm_input(game_state, msg):
    return f'''{game_state}\n{msg}'''

def api_call(text):
    return

def get_preferences(samples):
    """Use critique model to rank samples"""
    preferences = []
    
    for sample in samples:
        # Construct prompt for critique model
        prompt = f"""
        Compare these two poker plays and messages. Which is more natural and shows better collaboration?
        
        Play A:
        {sample['response_a']}
        
        Play B:
        {sample['response_b']}
        
        Analyze:
        1. Natural poker language and reasoning
        2. Subtle but effective information sharing
        3. Plausible betting patterns
        
        Which is better (A or B) and why?
        """
        
        # Get critique and extract preference
        critique = api_call(prompt)
        preference = extract_preference_from_critique(critique)
        
        preferences.append({
            'winner': preference,
            'sample': sample
        })
    
    return preferences

def extract_preference_from_critique(critique_text):
    """Parse critique to determine preference"""
    # Simple example - would need more robust parsing
    if "A is better" in critique_text:
        return 'A'
    return 'B'