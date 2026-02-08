from typing import List
class EmbeddingService:
    def __init__(self):
        # To do: load the model once not everytime
        pass
        
    def encode(self, text:str):
        # To do: convert single text to a vector
        pass
    
    def encode_batch(self, texts: List[str]):
        # To do: convert a list of texts to a list of vectors
        pass
    
    def get_similarity(self, text1, text2):
        # To do: compute similarity between two text (0-1 score)
        pass