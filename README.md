# NotebookPRO 📚

> A powerful AI-powered learning assistant that outperforms NotebookLM by providing intelligent explanations, summaries, and study notes from your lecture slides and reference materials.

## ⚡ **No Training Required!**

NotebookPRO uses **RAG (Retrieval-Augmented Generation)** - upload your documents and start chatting immediately with pre-trained AI models. No GPU, no waiting, no fine-tuning needed!

```
Upload Documents → Instant Embedding → Ask Questions → Get Answers
                    (5 seconds)                         (Real-time)
```

## 🌟 Features

- **🔍 RAG-Powered**: Uses Retrieval-Augmented Generation - **No training required!**
- **⚡ Two Modes**: Fast Mode (instant) or Advanced AI (5GB download)
- **🚀 Works Immediately**: Upload docs and start chatting - no setup needed
- **🎓 Smart Study Assistant**: Upload lecture slides and reference books to get intelligent explanations
- **💬 Ollama-like Interface**: Clean, intuitive chat interface with conversation history
- **📂 Organized Spaces**: Manage different subjects in separate spaces with dedicated contexts
- **🎯 Multiple Use Cases**: Choose between explanation, summary, Q&A, and structured notes
- **💾 Chat History**: Save and restore previous conversations
- **📄 Multi-Format Support**: Process PDFs, DOCX, and TXT files

## ⚡ Two Operation Modes

### Fast Mode (Default - Recommended)
- ✅ **Instant responses** - no waiting, no downloads
- ✅ **Uses RAG only** - directly formats retrieved document chunks
- ✅ **Perfect for**: Quick summaries, Q&A, note generation
- ✅ **Works immediately** - upload docs and start chatting
- ✅ **Grounded responses** - 100% based on your documents

### Advanced AI Mode (Optional)
- 🤖 **Uses Phi-2 LLM** - more natural language generation
- ⚠️ **5GB download** on first use (10-30 minutes)
- 🎯 **Better for**: Complex explanations, creative responses
- 🔧 **Enable in Settings** → "Enable Advanced AI Model"
- 💡 **Tip**: Try Fast Mode first, enable Advanced only if needed

## 🧠 How RAG Works (Training-Free!)

```
Your Documents → Text Extraction → Embedding → Vector DB (ChromaDB)
                                                       ↓
Your Question → Semantic Search → Retrieve Relevant Chunks
                                                       ↓
                  Pre-trained LLM + Retrieved Context → Answer
```

**No fine-tuning needed** - the model stays frozen, and your documents are retrieved dynamically!

👉 **[Read detailed RAG architecture explanation](RAG_ARCHITECTURE.md)**

## 🏗️ Project Structure

```
NoteBookPRO/
├── App.py                          # Main Streamlit application (RAG interface)
├── config.py                       # Configuration (uses pre-trained models)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── RAG_ARCHITECTURE.md             # Detailed RAG explanation
├── QUICKSTART.md                   # 5-minute getting started
├── training_colab.ipynb           # OPTIONAL: Advanced fine-tuning
├── .env.example                    # Environment variables template
├── utils/
│   ├── document_processor.py      # Extract & chunk documents
│   ├── vector_db.py               # ChromaDB for semantic search
│   ├── model_inference.py         # Pre-trained LLM inference
│   └── chat_manager.py            # Chat history management
├── data/
│   ├── uploads/                   # Uploaded documents
│   ├── vector_db/                 # Embedded vectors (ChromaDB)
│   └── chats/                     # Saved chat histories
└── models/                        # (Empty - uses HuggingFace pre-trained)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
cd NoteBookPRO

# Install dependencies
pip install -r requirements.txt

# Create environment file
copy .env.example .env
```
(created from `.env.example`):

```env
# Default configuration for RAG (no changes needed!)
MODEL_NAME=microsoft/phi-2
USE_PRETRAINED=true
TEMPERATURE=0.7
MAX_TOKENS=2048
```

**That's it!** No model training, no downloading large files - uses HuggingFace directly.TOR_DB_PATH=./data/vector_db
TEMPERATURE=0.7
MAX_TOKENS=2048
```

### 3. Run the Application

```bash
streamlit run App.py
```

The application will open in your browser at `http://localhost:8501`

## 🎓 Optional: Fine-tuning Your Own Model

⚠️ **You can skip this section!** The app works perfectly with pre-trained models using RAG.

### When to Fine-tune:
- You have a **very large** specialized corpus (medical, legal, etc.)
- You want **maximum** domain-specific performance
- You want the model to learn specific terminology/style

### Using Google Colab (Advanced - Optional)

1. **Upload the notebook**: Open `training_colab.ipynb` in Google Colab

2. **Prepare your data**: 
   - Create a folder in Google Drive (e.g., `NotebookPRO_Data`)
   - Upload your lecture slides, reference books, and notes (PDF, DOCX, TXT)

3. **Run the notebook**:
   - Update `DATA_DIR` to point to your Google Drive folder
   - Run all cells in order
   - Training will take 1-3 hours depending on your dataset size

4. **Download the model**:
   - The last cell will download a ZIP file
   - Extract it to `NoteBookPRO/models/`

5. **Update configuration**:
   ```python
   # In .env file (create from .env.example)
   USE_PRETRAINED=false
   MODEL_PATH=./models/notebookpro_model
   ```

### Pre-trained Models (Used by Default)

The app works immediately with these models via RAG:

- **microsoft/phi-2** (Default): 2.7B parameters, fast and efficient
- **mistralai/Mistral-7B-v0.1**: 7B parameters, more capable
- **meta-llama/Llama-2-7b-hf**: 7B parameters, requires HuggingFace token

## 📖 Usage Guide

### How RAG Works in NotebookPRO

1. **Upload Documents** → Extracted and embedded into ChromaDB
2. **Ask Question** → Semantic search finds relevant chunks  
3. **LLM Generates** → Uses retrieved context to answer
4. **No Training** → Everything happens in real-time!

### Creating a Space

1. Navigate to the **Spaces** section from the sidebar
2. Click "Create New Space"
3. Enter a name (e.g., "Machine Learning 101")
4. Add a description
5. Click "Create Space"

### Uploading Study Materials (RAG Embedding)

1. Select your space from the dropdown
2. Click "Upload Study Materials"
3. Choose your files (PDFs, DOCX, or TXT)
4. Click "Process Documents"
5. **Behind the scenes**:
   - Text is extracted from files
   - Split into semantic chunks (RAG Retrieval)

1. Go to the **Chats** section
2. Select a space from the dropdown
3. Choose your use case:
   - **Explanation**: Detailed concept explanations
   - **Summary**: Concise summaries
   - **Q&A**: Answer specific questions
   - **Notes**: Structured study notes
4. Type your question or request
5. **Behind the scenes**:
   - Question is embedded
   - Top 3-5 relevant chunks retrieved from ChromaDB
   - Chunks + question sent to pre-trained LLM
   - LLM generates answer using your documents as context
6. View the AI-generated response (grounded in your materials!)ncept explanations
   - **Summary**: Concise summaries
   - **Q&A**: Answer specific questions
   - **Notes**: Structured study notes
4. Type your question or request
5. View the AI-generated response

### Managing Chat History

1. Navigate to the **History** section
2. Browse previous conversations
3. Click "Load" to continue a previous chat
4. All chats are automatically saved

## 🛠️ Advanced Configuration

### Using a Pre-trained Model

If you don't want to train your own model, you can use a pre-trained model from HuggingFace:

```python
# In config.py
MODEL_NAME = "microsoft/phi-2"  # or any compatible model
```

### Adjusting Response Parameters

```python
# In config.py
TEMPERATURE = 0.7      # Higher = more creative (0.0-1.0)
MAX_TOKENS = 2048      # Maximum response length
CHUNK_SIZE = 512       # Document chunk size
CHUNK_OVERLAP = 50     # Chunk overlap for context
```

### Customizing Use Cases

Edit the `USE_CASES` dictionary in `config.py`:

```python
USE_CASES = {
    "explanation": "Provide detailed explanation of concepts",
    "summary": "Generate concise summary of content",
    "qa": "Answer questions based on content",
    "notes": "Create structured study notes",
    "custom": "Your custom use case description"
}
```

## 📊 System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **Python**: 3.9 or higher

### For Model Training (Google Colab)

- Google Colab Pro (recommended for faster training)
- GPU runtime (T4 or better)
- 15-30GB Google Drive space for training data and model

### For Running the Application

- **With GPU**: NVIDIA GPU with 8GB+ VRAM (for local model inference)
- **Without GPU**: CPU-only mode (slower but works)

## 🐛 Troubleshooting

### Common Issues

**Issue**: First run takes a long time / downloads models
- **Solution**: The embedding model (~100MB) downloads on first run. Subsequent runs are instant.

**Issue**: PDF font warnings flooding the console
- **Solution**: Fixed in latest version. Warnings are now suppressed. Update if you see this.

**Issue**: "Processing documents..." seems stuck
- **Solution**: Check the Streamlit interface for progress bars. First-time embedding model download can take 2-3 minutes.

**Issue**: Model loading is very slow
- **Solution**: Using pre-trained models with RAG is instant. Only fine-tuned models need loading time.

**Issue**: Out of memory errors
- **Solution**: Reduce document size, process fewer files at once, or use CPU-only mode

**Issue**: Documents not being processed
- **Solution**: 
  - Check file formats are supported (PDF, DOCX, TXT)
  - Ensure PDFs aren't password-protected or corrupted
  - Try processing one file at a time first
  - Check the progress messages in the UI

**Issue**: Vector database errors
- **Solution**: Delete `data/vector_db/` folder and reprocess documents

**Issue**: No relevant context found in responses
- **Solution**:
  - Make sure you've uploaded and processed documents
  - Try rephrasing your question
  - Upload more relevant documents
  - Check that you're in the correct Space

**Issue**: Import errors after installation
- **Solution**: 
  ```bash
  pip install --upgrade -r requirements.txt
  ```

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
- Vector database by [ChromaDB](https://www.trychroma.com/)
- Training framework by [PEFT](https://github.com/huggingface/peft) and [TRL](https://github.com/huggingface/trl)

## 📧 Support

For questions, issues, or feedback, please open an issue on the GitHub repository.

---

**Made with ❤️ for students and lifelong learners**
