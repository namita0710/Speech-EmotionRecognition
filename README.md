# Speech and Emotion Recognition Project  

## Introduction  
The **Speech and Emotion Recognition Project** focuses on identifying emotions from speech and text data using advanced machine learning techniques. This project demonstrates audio feature extraction, model training, and emotion classification.  

---

## Features  
- **Speech Emotion Recognition**: Processes audio files to detect emotions.  
- **Text Emotion Recognition**: Analyzes text data for emotional context.  
- **Visualization**: Plots and visualizes features like Mel spectrograms for better understanding.  
- **Accuracy Analysis**: Evaluates model performance using metrics such as accuracy score.  

---

## Libraries Used  
The project utilizes the following Python libraries:  

### For Speech Processing  
- `os`: For file management.  
- `librosa`: For audio feature extraction.  
- `IPython.display`: For audio playback.  
- `scipy.io.wavfile`: For handling WAV files.  

### For Text Processing  
- `nltk`: For text preprocessing and tokenization.  
- `re`: For regular expression operations.  
- `pandas`: For data manipulation.  

### For Machine Learning  
- `sklearn`:
  - `TfidfVectorizer`: For feature extraction from text data.  
  - `train_test_split`: For splitting datasets into training and testing sets.  
  - `RandomForestClassifier`: For classification.  
  - `accuracy_score`: To evaluate model performance.  

### For Visualization  
- `matplotlib.pyplot`: For plotting audio features and visualizations.  

---

## Setup Instructions  
1. Clone the repository:  
   ```bash
   git clone https://github.com/namita0710/Speech-EmotionRecognition.git
   cd speech-emotion-recognition
