import os
from typing import List
from PyPDF2 import PdfReader


class PDFFileLoader:
    def __init__(self, path: str):
        self.documents = []
        self.path = path

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        reader = PdfReader(self.path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        self.documents.append(text)

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    reader = PdfReader(os.path.join(root, file))
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    self.documents.append(text)

    def load_documents(self):
        self.load()
        return self.documents 