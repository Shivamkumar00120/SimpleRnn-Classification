# IMDB Movie Review Sentiment Analysis using Simple RNN

An end-to-end Deep Learning pipeline built using **TensorFlow and Keras** to classify IMDB movie reviews. The network estimates the probability that a textual movie review contains positive or negative sentiment using a recurrent layout.

---

## 🔍 Architecture & Decision Boundary
The model processes tokenized text sequences and applies a binary probability distribution over a single-neuron final layer:

| Prediction Score (y) | Sentiment Classification | Core Architectural Cause |
| :--- | :--- | :--- |
| **0.0 ≤ y < 0.5** | 🟥 **Negative Sentiment** | Explicitly driven by critical terms like *terrible*, *waste*, or *worst* weighing down sequence vector activations. |
| **0.5 < y ≤ 1.0** | 🟩 **Positive Sentiment** | Driven by optimistic patterns like *brilliant*, *fantastic*, or *thrilling*. |

### Model Pipeline Breakdown
1. **Text Preprocessing:** User input strings are cast to lower case, split into isolated tokens, and converted to integer mappings using the native Keras IMDB word index.
2. **Pre-Padding Integration:** Sequences are formatted uniformly using pre-padding (`padding='pre'`) up to a maximum length threshold of **500 words** (`maxlen=500`).
3. **Embedding Layer:** Projects the mapped indices from a total vocabulary size of **10,000 words** into a dense feature representation space of **128 dimensions**.
4. **SimpleRNN Layer:** Recurrent layer configured with **128 hidden units** utilizing a **ReLU activation function** to model sequential context.
5. **Output Dense Layer:** A single dense neuron operating with a **Sigmoid activation function** to construct the target prediction probability boundary.

---

## 🛠️ Project Structure
```text
├── simple_rnn_imdb.h5       # Pre-trained HDF5 model weight blueprint
├── main.py                   # Streamlit web interface deployment file
├── requirements.txt         # Package ecosystem definition file
└── simplernn.ipynb # Jupyter notebook tracking training & callbacks
```

---

## 🚀 Quick Start & Model Evaluation

### 1. Verification Inputs
To validate if your classification threshold registers negative classes below the `0.5` boundary, try testing your deployed model with these sample reviews:

* **Short Test Input String:**
  > `"The movie was an absolute waste of time. The acting was terrible and the plot made no sense."`
* **Long Dataset-Style String:**
  > `"Avoid this movie at all costs! The direction was completely messy and the special effects looked incredibly cheap. It fails as entertainment, and the ending is completely abrupt. I give it a 2 out of 10."`

### 2. Running Locally via Streamlit
You can serve your architecture through an interactive dashboard. Run this terminal command inside your active virtual environment:

```bash
streamlit run main.py
```

## To run app click on given link below-> 
https://simplernn-classification-vliyw8jkub2hmr6d9yq5kk.streamlit.app/
