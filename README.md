# Legal-LM


### 1. Prerequisites

- Python 3.x (recommended 3.8+)
- Virtual environment (recommended)
- Either Ollama installed locally or OpenAI API access


### 2. Installation Steps

1. **Create and activate a virtual environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```


2. **Install dependencies** from `requirements.txt`:
```bash
pip install -r requirements.txt
```


3. **Environment Setup**:

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI configuration (required if using OpenAI)
OPENAI_API_KEY=your_api_key_here

# Ollama configuration (required if using Ollama)
OLLAMA_HOST=http://localhost:11434  # Default Ollama host
```



### 3. Running the Framework


The main entry point is in `main.py`. There are two ways to use the framework:

1. **Using Ollama (Local)**:
```python
from main import LegalDiscrepancyDetectionFramework

# Initialize with Ollama
framework = LegalDiscrepancyDetectionFramework(use_ollama=True, model_name="llama3")

# Process a document
result = framework.process_document("path/to/your/legal_document.pdf")
print("Analysis complete. Results:", result)
```


2. **Using OpenAI API**:
```python
from main import LegalDiscrepancyDetectionFramework

# Initialize with OpenAI
framework = LegalDiscrepancyDetectionFramework(use_ollama=False, model_name="gpt-4")

# Process a document
result = framework.process_document("path/to/your/legal_document.pdf")
print("Analysis complete. Results:", result)
```



### 4. Supported Document Types

The framework supports the following file formats:

- PDF (`.pdf`)
- Microsoft Word (`.docx`, `.doc`)
- Plain text (`.txt`)


### 5. Framework Components


The framework consists of several agents that work together:

- Preprocessor: Handles document parsing and initial processing

- Knowledge Agent: Retrieves relevant legal information

- Compliance Checker: Detects contradictions and issues

- Clause Rewriter: Suggests fixes for problematic clauses

- Postprocessor: Generates final reports

