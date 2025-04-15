import os
from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
from abc import ABC, abstractmethod

__all__ = ['TextFileLoader', 'CharacterTextSplitter', 'SentenceTextSplitter']


class TextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split a single text into chunks."""
        pass
    
    def split_texts(self, texts: List[str]) -> List[str]:
        """Split multiple texts into chunks."""
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks

class TokenCounter:
    """Utility class for counting tokens in text."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.encoding = tiktoken.encoding_for_model(model_name)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        if text is None:
            raise ValueError("Input text cannot be None")
        if not text.strip():
            return 0  # Empty or whitespace-only text has 0 tokens
            
        return len(self.encoding.encode(text))

class CharacterTextSplitter(TextSplitter):
    """Splits text based on character count."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        assert chunk_size > chunk_overlap, "Chunk size must be greater than chunk overlap"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, text: str) -> List[str]:
        if text is None:
            raise ValueError("Input text cannot be None")
        if not text.strip():
            return []  # Return empty list for empty or whitespace-only text
        
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

class SentenceTokenizer:
    """Handles sentence tokenization with special case handling."""
    
    def __init__(self):
        # Set up NLTK data path to be local to the project
        nltk_data_dir = os.path.join(os.getcwd(), '.nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.insert(0, nltk_data_dir)

        # Download required NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', download_dir=nltk_data_dir)
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', download_dir=nltk_data_dir)
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into sentences with special case handling."""
        if text is None:
            raise ValueError("Input text cannot be None")
        if not text.strip():
            return []  # Return empty list for empty or whitespace-only text
            
        # Handle common PDF extraction issues
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Special handling for bullet points and numbered lists
        text = re.sub(r'([.!?])\s*([•\-\d]+\s)', r'\1\n\2', text)
        
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Post-process sentences
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if re.match(r'^[•\-\d]+\s', sentence):
                    processed_sentences.append(sentence)
                else:
                    processed_sentences.append(sentence)
        
        return processed_sentences

class SentenceTextSplitter(TextSplitter):
    """Splits text based on sentences with token limits."""
    
    def __init__(
        self,
        max_sentences: int = 5,
        sentence_overlap: int = 1,
        model_name: str = "text-embedding-3-small",
        max_tokens: int = 8191
    ):
        assert max_sentences > sentence_overlap, "Max sentences must be greater than overlap"
        self.max_sentences = max_sentences
        self.sentence_overlap = sentence_overlap
        self.max_tokens = max_tokens
        self.token_counter = TokenCounter(model_name)
        self.sentence_tokenizer = SentenceTokenizer()
        self.character_splitter = CharacterTextSplitter(
            chunk_size=max_tokens * 4,
            chunk_overlap=50
        )
    
    def split(self, text: str) -> List[str]:
        if text is None:
            raise ValueError("Input text cannot be None")
        if not text.strip():
            return []  # Return empty list for empty or whitespace-only text
        
        sentences = self.sentence_tokenizer.tokenize(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for i in range(len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self.token_counter.count_tokens(sentence)
            
            # Handle long sentences
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                
                sub_chunks = self.character_splitter.split(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Check limits
            if (len(current_chunk) >= self.max_sentences or 
                current_token_count + sentence_tokens > self.max_tokens):
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = current_chunk[-self.sentence_overlap:] if self.sentence_overlap > 0 else []
                    current_token_count = sum(self.token_counter.count_tokens(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = SentenceTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
