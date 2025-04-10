import os
import pandas as pd
import pdfplumber
from docx import Document
from concurrent.futures import ProcessPoolExecutor

class ResumeProcessor:
    
    def __init__(self, folder_path, csv_path):
        """Initialize with folder path and CSV file path."""
        self.folder_path = folder_path
        self.csv_path = csv_path
# --------------------------- Extract Text From Resume ------------------------------------
    def extract_text(self, file_path):
        """Extract text based on file format."""
        try:
            if file_path.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            elif file_path.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as file:
                    return file.read().strip()
            elif file_path.endswith(".docx"):
                doc = Document(file_path)
                return "\n".join(para.text for para in doc.paragraphs).strip()
            else:
                return ""  # Handle unsupported file types
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""  # Return empty text in case of error

# -------------------------- Use Multiprocessing to process resume ------------------------------

    def process_resumes(self):
        """Process all resumes in the given folder using multiprocessing."""
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"Error: {self.folder_path} is not a valid directory!")

        files = [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if os.path.isfile(os.path.join(self.folder_path, f)) and f.endswith((".pdf", ".txt", ".docx"))
        ]

        with ProcessPoolExecutor() as executor:
            resume_texts = list(executor.map(self.extract_text, files))

        return pd.DataFrame({"file_name": [os.path.basename(f) for f in files], "resume_content": resume_texts})
    
# -------------------------- Append Resume Content to CSV ----------------------------------------------
    def append_to_csv(self, df):
        """Append data to CSV if it exists, otherwise create the CSV."""
        if os.path.exists(self.csv_path):
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(self.csv_path, mode='w', header=True, index=False)

    def process_and_append(self):
        """Process resumes and append the results to the CSV file."""
        df = self.process_resumes()
        self.append_to_csv(df)
        print(f"Data successfully appended to {self.csv_path}")

# ---------------------------- Run the Process -----------------------------------
if __name__ == "__main__":
    folder_path = "./data/resumes"  # Ensure this is a directory
    csv_path = "./data/resumes_data     .csv"  # Path to the CSV file

    # Create an instance of ResumeProcessor and run the process
    processor = ResumeProcessor(folder_path, csv_path)
    processor.process_and_append()

