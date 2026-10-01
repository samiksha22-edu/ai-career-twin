import os
import PyPDF2
import docx

class ResumeParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.extension = os.path.splitext(file_path)[1].lower()

    def extract_text(self) -> str:
        """Determines file type and extracts raw text string."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.extension == '.pdf':
            return self._parse_pdf()
        elif self.extension == '.docx':
            return self._parse_docx()
        elif self.extension == '.txt':
            return self._parse_txt()
        else:
            raise ValueError(f"Unsupported file extension: {self.extension}")

    def _parse_pdf(self) -> str:
        text = ""
        with open(self.file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text

    def _parse_docx(self) -> str:
        doc = docx.Document(self.file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])

    def _parse_txt(self) -> str:
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()