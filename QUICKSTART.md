# Quick Start Guide - NotebookPRO

## 🚀 Get Started in 5 Minutes (No Training Required!)

NotebookPRO uses **RAG (Retrieval-Augmented Generation)** - it works immediately with pre-trained AI models. Just upload your documents and start chatting!

### How It Works:
1. **Upload** your lecture slides/books (PDF, DOCX, TXT)
2. **Documents are embedded** and stored in a vector database
3. **Ask questions** - relevant content is retrieved automatically
4. **Get answers** - the AI uses retrieved context to respond

**No training, no waiting - works instantly!**

### Step 1: Launch the Application

**Windows:**
```bash
Double-click start.bat
```

**Mac/Linux:**
```bash
chmod +x start.sh
./start.sh
```

or manually:
```bash
pip install -r requirements.txt
streamlit run App.py
```

### Step 2: Create Your First Space

1. Click on **"Spaces"** in the sidebar
2. Click **"Create New Space"**
3. Enter a name (e.g., "Computer Science 101")
4. Add a description (optional)
5. Click **"Create Space"**

### Step 3: Upload Study Materials

1. Go to **"Chats"** in the sidebar
2. Select your newly created space from the dropdown
3. Expand **"Upload Study Materials"**
4. Click **"Browse files"** and select your PDFs, DOCX, or TXT files
5. Click **"Process Documents"**
6. Wait for processing (shows progress)

### Step 4: Start Learning!

1. Select a **use case** from the dropdown:
   - **Explanation**: Get detailed explanations of concepts
   - **Summary**: Get concise summaries
   - **Q&A**: Ask specific questions
   - **Notes**: Generate structured study notes

2. Type your question in the chat box, for example:
   - "Explain the main concepts covered in lecture 1"
   - "Summarize chapter 3 of the reference book"
   - "What is machine learning?"
   - "Create study notes for this week's topics"

3. Press Enter and wait for the AI response

## 📚 Sample Use Cases

### For Explanations
```
"Explain the concept of neural networks in detail"
"Help me understand how gradient descent works"
"What are the key differences between supervised and unsupervised learning?"
```

### For Summaries
```
"Summarize the main points from today's lecture"
"Give me a brief overview of chapter 5"
"What are the key takeaways from this week's material?"
```

### For Q&A
```
"What is the formula for calculating accuracy?"
"How does backpropagation work?"
"What are the advantages of using CNNs?"
```

### For Study Notes
```
"Create structured notes for the entire module"
"Generate a study guide for the upcoming exam"
"Organize the key concepts from all lectures"
```

## 🎓 Optional: Fine-tuning (Advanced Users Only)

**Skip this!** RAG works perfectly without training.

If you really want to fine-tune for maximum domain-specific performance:

1. Open `training_colab.ipynb` in Google Colab
2. Upload your study materials to Google Drive
3. Update the `DATA_DIR` path in the notebook
4. Run all cells (takes 1-3 hours)
5. Download the trained model
6. Extract to `models/` folder
7. Update `.env` file: `USE_PRETRAINED=false` and `MODEL_PATH=./models/your_model`

**But again - this is totally optional!** The app works great with pre-trained models.

## ⚙️ Configuration Tips

### For Faster Responses
In [config.py](config.py):
```python
MAX_TOKENS = 1024  # Reduce for shorter responses
TEMPERATURE = 0.3  # Lower for more focused responses
```

### For More Creative Responses
```python
MAX_TOKENS = 2048  # Increase for longer responses
TEMPERATURE = 0.9  # Higher for more creative responses
```

### For Better Context Retrieval
```python
CHUNK_SIZE = 1024  # Larger chunks for more context
CHUNK_OVERLAP = 100  # More overlap for better continuity
```

## 🐛 Common Issues

### "Model loading is slow"
- First load always takes time
- Subsequent responses will be faster
- Consider using a smaller model for your device

### "Out of memory"
- Close other applications
- Restart the app
- Use CPU-only mode (edit config.py)

### "Documents not processing"
- Check file format (PDF, DOCX, TXT only)
- Ensure files aren't password-protected
- Try smaller files first

### "No relevant context found"
- Upload more documents
- Try rephrasing your question
- Make sure documents are relevant to your query

## 💡 Pro Tips

1. **Organize by Subject**: Create separate spaces for each course/subject
2. **Upload Regularly**: Add new materials as you get them
3. **Be Specific**: More specific questions get better answers
4. **Use History**: Review past chats to track your learning
5. **Combine Use Cases**: Use explanation first, then summary for review

## 📬 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Look at the training notebook for model training details
- Review the code in `utils/` for understanding how it works

---

Happy Learning! 🎉
