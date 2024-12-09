# puceny: A Simple Local Search Engine

**puceny** is a simplified search engine prototype that indexes and searches documents from a local directory. It supports reading text from files (such as `.txt`, `.md`, `.html`, `.pdf`), building an inverted index, and then performing keyword queries using a simple scoring method. Additionally, there's a Flask-based web UI to interact with the index.

## Features

- **Indexing Local Documents**:  
  The system crawls a local directory, extracting text from supported file formats:

  - `.txt` / `.md`: Directly read as text
  - `.html`: Parsed using BeautifulSoup to extract text
  - `.pdf`: Parsed using PyPDF2 to extract text
  - Other file types are skipped or return empty text

- **Inverted Index**:  
  Each token is processed by an Analyzer (tokenization, lowercasing, stopword removal) and mapped to the documents in which it appears, along with positional data.

- **Segment-Based Index**:  
  New documents are committed into immutable segments. These can later be merged into a single segment for optimization.

- **Simple Query Interface**:  
  The `Searcher` uses an in-memory inverted index to quickly retrieve documents matching query terms. A basic scoring function (TF-IDF-like) is implemented.

- **Flask Web Frontend**:  
  A Flask server provides a simple webpage where users can enter queries, rebuild the index, and view highlighted snippets of matched documents.

## Prerequisites

- Python 3.7+
- Packages:
  - `beautifulsoup4` (for HTML parsing)
  - `PyPDF2` (for PDF parsing)
  - `Flask` (for the web UI)

Install the required packages:

```bash
pip install beautifulsoup4 PyPDF2 flask
```

## Directory Structure

- `puceny.py`: The main indexing and searching implementation (Analyzer, IndexWriter, IndexReader, Searcher, etc.).
- `app.py`: A Flask application that:
  - Builds or rebuilds the index from a specified data directory.
  - Provides a web interface to perform searches.
- `DATA_DIR`: The directory containing the documents to be indexed.

## Configuration

In `app.py`, set the following variables before running:

- `DATA_DIR`: Path to the local directory containing your documents.
- `INDEX_DIR`: Directory where the index files will be stored.

Both `DATA_DIR` and `INDEX_DIR` should be absolute paths. You can edit these directly in the code:

```python
DATA_DIR = "/path/to/your/data"
INDEX_DIR = "puceny_index_cs61a"
```

## Building the Index

The first time you run the app, if no index exists, it will automatically be created. To manually rebuild the index, you can press the "Rebuild Index" button on the web interface.

When rebuilding, the console will show the indexing progress and, after completion, print the total number of documents indexed.

## Running the Application

1. Ensure `DATA_DIR` and `INDEX_DIR` are correctly set.
2. Run:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5050` in your browser.

## Using the Web Interface

- **Rebuild Index**: Click the "Rebuild Index" button at the top of the page to re-crawl the data directory and rebuild the index. This can be useful if you have updated, added, or removed documents.
- **Search**: Enter a keyword or multiple keywords in the search box and submit.

  - The results page will show matched documents along with highlighted snippets of the text where your search terms appear.
  - Document file paths are shown, and you can click on them to access the file directly (if supported by your environment).

- **Performance**: The code preloads the entire index into memory to speed up queries. For very large datasets, consider merging segments or optimizing the data structures further.

## Customization

- **Stopwords**: You can edit the default stopwords list in `Analyzer` if you need language-specific or domain-specific adjustments.
- **Scoring**: The `search_with_scores` method in `Searcher` uses a simple IDF formula. You can replace or enhance this with TF-IDF, BM25, or other scoring methods.
- **File Types**: Add more file parsers for other formats if needed.

## Limitations

- This is a prototype, not a production-ready search engine.
- The scoring and query parsing are rudimentary.
- No authentication or security checks are enforced beyond directory checks.

## License

This code is provided as-is for demonstration and educational purposes. Feel free to modify and integrate it into your own projects.

---

This `README.md` should provide a clear overview and instructions for someone looking to run and interact with the code you've provided.
