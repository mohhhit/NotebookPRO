# Quick Test Guide

## Test NotebookPRO in 2 Minutes

### Step 1: Start the App
```bash
streamlit run App.py
```

### Step 2: First-Time Setup (Automatic)
On first run, the app will:
1. Download sentence-transformers embedding model (~100MB) - takes 1-2 minutes
2. Initialize ChromaDB vector database
3. Load the interface

**This only happens once!** Subsequent starts are instant.

### Step 3: Test with Sample Document

1. Click on **"Spaces"** in sidebar → Create a new space called "ML Test"
2. Go to **"Chats"** → Select "ML Test" space
3. Expand **"📁 Upload Study Materials"**
4. Upload the test document: `data/test_document.txt`
5. Click **"📤 Process Documents"**

You should see:
```
🔧 Initializing vector database (first time setup)...
📄 Processing test_document.txt (1/1)...
📖 Extracting text from test_document.txt...
✂️ Chunking text from test_document.txt...
🔢 Embedding X chunks from test_document.txt...
✅ Processed test_document.txt - X chunks added
```

### Step 4: Test RAG Retrieval

Try these questions:

**For Explanation:**
```
Explain how backpropagation works in neural networks
```

**For Summary:**
```
Summarize the types of machine learning
```

**For Q&A:**
```
What is gradient descent?
```

**For Notes:**
```
Create study notes about CNNs and RNNs
```

### Expected Results

You should get responses that:
- ✅ Reference content from the uploaded document
- ✅ Are accurate and grounded (no hallucination)
- ✅ Match the selected use case (explanation vs summary)

### Troubleshooting

**If processing seems stuck:**
- Check the progress bar in the UI
- First-time embedding model download takes 1-2 minutes
- Check terminal for "Loading embedding model..." message

**If you see font warnings:**
- Make sure you have the latest version of utils/document_processor.py
- Warnings are suppressed in the latest code

**If no response or "no relevant context":**
- Make sure documents were processed successfully
- Check you're in the correct Space
- Try a more specific question

## Performance Expectations

**First Run:**
- Embedding model download: 1-2 minutes
- Processing 1 document: 5-15 seconds
- First query with model loading: 30-60 seconds (downloads Phi-2)

**Subsequent Runs:**
- App starts: <5 seconds
- Processing 1 document: 5-15 seconds  
- Each query: 5-15 seconds

## What's Happening Behind the Scenes

### When Processing Documents:
```
PDF/TXT → Extract Text → Split into Chunks → Embed with sentence-transformers
                                                        ↓
                                              Store in ChromaDB
```

### When Asking Questions:
```
Your Question → Embed → Search ChromaDB → Top 3-5 Chunks
                                                ↓
                        Pre-trained Phi-2 + Retrieved Context
                                                ↓
                                          Generated Answer
```

## Success Indicators

✅ You've successfully set up NotebookPRO if:
1. Documents process without errors
2. You can ask questions and get relevant responses
3. Responses reference your uploaded content
4. Different use cases produce different response styles

## Next Steps

Once the test works:
1. Create spaces for your actual subjects
2. Upload your lecture slides and reference books
3. Start learning!

No training required - the system uses RAG (Retrieval-Augmented Generation) with pre-trained models!
