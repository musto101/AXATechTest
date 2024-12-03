# a testing framework for the transcriptLLM.py script
from src.transcriptLLM import TranscriptLLM
import pandas as pd
import os
from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis")
sentiment_analyzer_val = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

transcript_dir = "data/transcripts_v3"
transcript_files = [os.path.join(transcript_dir, file) for file in os.listdir(transcript_dir) if
                    file.endswith(".txt")]

# Read the content of the transcript
with open(transcript_files[0], 'r') as file:
    transcript = file.read()

# Initialize the TranscriptLLM instance
transcript_analyzer = TranscriptLLM(transcript, sentiment_analyzer, sentiment_analyzer_val)

# Initialize the TranscriptLLM instance
# test that the object is correctly initialized
assert isinstance(transcript_analyzer, TranscriptLLM), "transcript_analyzer should be an instance of TranscriptLLM"

# Evaluate the models
with open(transcript_files[0], 'r') as file:
    transcript = file.read()
    acc = transcript_analyzer.evaluate_models(transcript)

# test that the evaluate_models method returns a float
assert isinstance(acc, float), "The method should return a float"

# Test consistency of the model
with open(transcript_files[0], 'r') as file:
    transcript = file.read()
    consistency_results = transcript_analyzer.test_consistency(transcript)

# test that the test_consistency method returns a dictionary
assert isinstance(consistency_results, dict), "The method should return a dictionary"

print("All tests passed!")