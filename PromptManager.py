from typing import List


class PromptManager:
    """Manages different prompt styles for ASL classification"""
    
    PROMPTS = {
        'standard': """Look at this image of an American Sign Language (ASL) hand sign. 
                      Identify which letter of the alphabet (A-Z) is being shown.
                      Respond with ONLY the single letter in capital letters, nothing else.""",
        
        'detailed': """This image shows a hand gesture representing a letter in American Sign Language (ASL).
                      Please analyze the hand position, finger configuration, and orientation.
                      Identify which letter from A to Z is being signed.
                      Respond with only the letter (e.g., 'A', 'B', 'C', etc.).""",
        
        'few_shot': """You are an ASL alphabet classifier. Given an image of a hand sign, 
                      identify the letter being shown.
                      Examples of ASL signs:
                      - Closed fist with thumb to the side = 'A'
                      - Open palm with fingers together = 'B'
                      - Curved hand in C-shape = 'C'
                      
                      Now look at the provided image and respond with only the letter being signed.""",
        
        'chain_of_thought': """Analyze this ASL hand sign step by step:
                               1. First, observe the overall hand position
                               2. Note the finger configuration
                               3. Check thumb position
                               4. Identify the letter being signed
                               
                               Final answer (letter only):""",
        
        'expert': """As an expert in American Sign Language, analyze this hand sign image.
                    Consider the precise finger positioning, thumb placement, and hand orientation
                    that distinguishes each letter in the ASL alphabet.
                    Identify the letter being signed and respond with only that letter.""",
        
        'simple': """What ASL letter is shown in this image? Answer with just the letter.""",
        
        'context_aware': """This image shows an ASL hand sign from the standard 26-letter alphabet.
                          Each letter has distinct characteristics:
                          - Some use finger spelling (individual fingers)
                          - Others use hand shapes and orientations
                          - Motion letters may appear static in images
                          
                          Identify which letter (A-Z) is being signed. Response: just the letter."""
    }
    
    @classmethod
    def get_prompt(cls, prompt_style: str) -> str:
        """Get prompt by style name"""
        return cls.PROMPTS.get(prompt_style, cls.PROMPTS['standard'])
    
    @classmethod
    def list_prompts(cls) -> List[str]:
        """Get list of available prompt styles"""
        return list(cls.PROMPTS.keys())
    
    @classmethod
    def add_custom_prompt(cls, name: str, prompt: str):
        """Add a custom prompt style"""
        cls.PROMPTS[name] = prompt