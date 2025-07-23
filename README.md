# ArXiv Paper Recommendation Engine

This project is a sophisticated paper recommendation system that leverages the vast arXiv dataset of scholarly articles. It utilizes state-of-the-art natural language processing techniques to understand the semantic content of paper abstracts and recommend similar articles. The system is composed of a data preparation pipeline built with Python and a user-facing chat interface developed in React.

## Project Overview

The core of this project is to provide users with relevant paper recommendations based on a query. It achieves this by:

1.  **Processing a large dataset of arXiv papers:** The system starts with a snapshot of the arXiv metadata, containing information like titles, authors, and abstracts for over 2 million papers.
2.  **Generating semantic embeddings:** Using a powerful SentenceTransformer model, it converts the abstract of each paper into a high-dimensional vector representation (embedding). These embeddings capture the semantic meaning of the text.
3.  **Storing embeddings efficiently:** The generated embeddings are stored in a Parquet file, a columnar storage format optimized for large-scale data analysis.
4.  **Providing an interactive interface:** A React-based chat application allows users to input a topic or abstract, which is then used to find and display the most similar papers from the dataset.

## Components

### 1. Data Preparation (`data_preparation.ipynb`)

This Jupyter Notebook is the heart of the data processing pipeline. It performs the following key steps:

*   **Data Loading and Parsing:** It reads the `arxiv-metadata-oai-snapshot.json` file, parsing each line as a JSON record to extract the title, authors, abstract, and publication year.
*   **Initial Data Storage:** The parsed data is initially stored in a pandas DataFrame for ease of manipulation.
*   **Efficient Serialization:** The pandas DataFrame is then converted to a Polars DataFrame and saved as a Parquet file (`arxiv_abstracts.parquet`). Polars is a highly efficient DataFrame library, and Parquet is a columnar storage format that is well-suited for large datasets.
*   **Embedding Generation:** It utilizes the `sentence-transformers` library to load a pre-trained model (`jinaai/jina-embeddings-v3`) and generate embeddings for all the paper abstracts. This process is computationally intensive and is accelerated using a CUDA-enabled GPU.
*   **Chunking and Merging:** To handle the large volume of data and embeddings, the notebook processes the data in chunks, saves them as individual Parquet files, and then efficiently merges them into a single, comprehensive Parquet file (`arxiv_embeddings_full.parquet`).

### 2. Frontend Application

The frontend is a single-page application built with React. It provides a user-friendly interface for interacting with the recommendation engine.

*   **`App.js` and `index.js`:** These are the main entry points for the React application.
*   **`ChatPage.js` (Inferred):** The CSS file `ChatPage.css` strongly suggests the existence of a `ChatPage.js` component. This component is responsible for rendering the chat interface, handling user input, and displaying the conversation history, including the paper recommendations.
*   **Styling (`App.css`, `ChatPage.css`, `index.css`):** These files contain the CSS rules for styling the application, giving it a clean and modern look.
*   **Testing (`App.test.js`, `setupTests.js`):** The project includes a basic testing setup using React Testing Library and Jest to ensure the application's components render correctly.
*   **Web Vitals (`reportWebVitals.js`):** This file includes functionality to measure and report on the performance of the web application.

## How to Run

### Prerequisites

*   Python 3.x
*   Node.js and npm
*   A CUDA-enabled GPU is highly recommended for the data preparation step.

### Data Preparation

1.  **Download the dataset:** Obtain the `arxiv-metadata-oai-snapshot.json` file from the [arXiv dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
2.  **Install Python dependencies:**
    ```bash
    pip install pandas polars sentence-transformers torch pyarrow tqdm jupyter
    ```
3.  **Run the Jupyter Notebook:** Execute the cells in `data_preparation.ipynb` sequentially. This will generate the `arxiv_embeddings_full.parquet` file.

### Frontend Application

1.  **Navigate to the frontend directory:**
    ```bash
    cd path/to/your/react/app
    ```
2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```
3.  **Start the development server:**
    ```bash
    npm start
    ```
4.  Open your browser and go to `http://localhost:3000` to interact with the chat interface.

## Future Enhancements

*   **Backend API:** Develop a backend API (e.g., using Flask or FastAPI) to serve the paper recommendations. The React application would then fetch data from this API.
*   **Advanced Filtering:** Implement options to filter recommendations by year, author, or category.
*   **User Profiles:** Allow users to create profiles and save their search history and favorite papers.
*   **Scalability:** For even larger datasets, consider using a distributed computing framework like Spark for the data preparation and a dedicated vector database for storing and querying the embeddings.