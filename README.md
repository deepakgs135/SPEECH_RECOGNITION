# SPEECH_RECOGNITION

# Speech Processing Dashboard

This project is a **Speech Processing Dashboard** built using Python's Dash framework. It allows users to upload media files, extract audio, perform transcription, analyze sentiments, and generate keyword frequency visualizations. It also supports audio segmentation for segment-wise analysis of the audio content.

## Features

- **Media File Upload**: Upload audio or video files (supported formats: `.mp4`, `.avi`, `.mov`, `.mp3`).
- **Audio Transcription**: Automatic transcription of the uploaded audio using the `Wav2Vec2` model from Hugging Face.
- **Sentiment Analysis**: Analyzes the sentiment of each segmented transcription.
- **Keyword Frequency**: Extracts the most frequent keywords from the transcribed text and displays them in a bar chart.
- **Audio Segmentation**: Users can adjust the number of segments for the audio. Each segment is analyzed individually for transcription and sentiment.
- **Timeline Visualization**: Displays a timeline graph showing the segmentation of the audio.
- **Word Count and WER/CER**: Displays the word count and calculates the Word Error Rate (WER) and Character Error Rate (CER) for transcription.
- **Text Summarization**: Provides a summary of the entire transcription.

## Dependencies

This project relies on several Python packages:

- `dash`: Web framework for building dashboards.
- `dash-bootstrap-components`: Bootstrap components for Dash apps.
- `transformers`: Hugging Face transformers for using `Wav2Vec2`, `Sentiment Analysis`, and `Summarization` models.
- `librosa`: Python package for audio processing.
- `soundfile`: Library for reading and writing sound files.
- `jiwer`: Package for calculating Word Error Rate (WER) and Character Error Rate (CER).
- `spacy`: Natural Language Processing package for keyword extraction.
- `plotly`: Library for interactive visualizations.
- `moviepy`: For video file processing and audio extraction.
- `langdetect`: For language detection.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/speech-processing-dashboard.git
   cd speech-processing-dashboard
Install the required Python libraries:

pip install -r requirements.txt
Ensure that the requirements.txt file contains the following:


dash
dash-bootstrap-components
transformers
librosa
soundfile
jiwer
spacy
plotly
moviepy
langdetect
Download SpaCy model:


python -m spacy download en_core_web_md
Running the App
To run the app locally, execute the following command:


python app.py

This will launch the app at http://127.0.0.1:8050/ in your browser.

Usage
Upload Media: Click on "Upload Media File" and upload an .mp4, .avi, .mov, or .mp3 file.
Audio Segmentation: Use the slider to choose how many segments to divide the audio into.
Transcription & Sentiment: View the transcription of the audio and sentiment analysis for each segment.
Keyword Frequencies: A bar chart showing the top 10 most frequent keywords from the transcription.
Word Count & WER/CER: Displays the word count and error rates of the transcription.
Timeline: Visualize the audio segmentation on a timeline.
File Structure

speech-processing-dashboard/
│
├── app.py                 # Main Python file containing the Dash app
├── README.md              # This readme file
├── requirements.txt       # Python dependencies
Models Used
Wav2Vec2: Pretrained model for automatic speech recognition.
Sentiment Analysis: Sentiment analysis pipeline from Hugging Face.
Summarization: Summarization pipeline from Hugging Face.
Future Enhancements
Automatic speaker detection to determine the number of segments dynamically.
Support for more file formats and enhanced error handling.
Add more NLP features like entity recognition and advanced sentiment metrics.
License
This project is licensed under the MIT License. See the LICENSE file for details.



This `README.md` provides detailed instructions on how to install, run, and use the Dash app, along with a description of its features and structure.





