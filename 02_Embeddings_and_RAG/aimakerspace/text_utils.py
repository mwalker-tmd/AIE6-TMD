import os
from typing import List
import re
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken

__all__ = ['TextFileLoader', 'CharacterTextSplitter', 'SentenceTextSplitter']

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


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


class SentenceTextSplitter:
    def __init__(
        self,
        max_sentences: int = 5,
        sentence_overlap: int = 1,
        model_name: str = "text-embedding-3-small",
        max_tokens: int = 8191  # text-embedding-3-small token limit
    ):
        """
        Initialize the sentence-based text splitter.
        
        Args:
            max_sentences: Maximum number of sentences per chunk
            sentence_overlap: Number of sentences to overlap between chunks
            model_name: Name of the model for token counting
            max_tokens: Maximum tokens per chunk
        """
        assert max_sentences > sentence_overlap, "Max sentences must be greater than overlap"
        self.max_sentences = max_sentences
        self.sentence_overlap = sentence_overlap
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model_name)
        
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK's sentence tokenizer."""
        # Handle common PDF extraction issues
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Special handling for bullet points and numbered lists
        text = re.sub(r'([.!?])\s*([•\-\d]+\s)', r'\1\n\2', text)
        
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Post-process sentences
        processed_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Only add non-empty sentences
                # Preserve bullet points and numbering at start of sentences
                if re.match(r'^[•\-\d]+\s', sentence):
                    processed_sentences.append(sentence)
                else:
                    processed_sentences.append(sentence)
        
        return processed_sentences
        
    def split(self, text: str) -> List[str]:
        """Split a single text into chunks based on sentences and token limits."""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for i in range(len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self._count_tokens(sentence)
            
            # If a single sentence exceeds token limit, split it (fallback to character splitting)
            if sentence_tokens > self.max_tokens:
                if current_chunk:  # Add current chunk if it exists
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
                
                # Split long sentence into smaller parts
                char_splitter = CharacterTextSplitter(
                    chunk_size=self.max_tokens * 4,  # Approximate chars per token
                    chunk_overlap=50
                )
                sub_chunks = char_splitter.split(sentence)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this sentence would exceed either max_sentences or max_tokens
            if (len(current_chunk) >= self.max_sentences or 
                current_token_count + sentence_tokens > self.max_tokens):
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep overlap sentences for next chunk
                    current_chunk = current_chunk[-self.sentence_overlap:] if self.sentence_overlap > 0 else []
                    current_token_count = sum(self._count_tokens(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
        
    def split_texts(self, texts: List[str]) -> List[str]:
        """Split multiple texts into chunks based on sentences."""
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


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
