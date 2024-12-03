from transformers import pipeline
import re
from collections import Counter
import os

# Load a sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/distilbert-base-uncased-finetuned-sst-2-english")

def determine_call_outcome(transcript):
    """
    Infers the call outcome based on common phrases.
    Returns 'Issue Resolved' or 'Follow-up Action Needed'.
    """
    resolved_keywords = ["issue resolved", "problem fixed", "refund processed", "resolved", "successful"]
    follow_up_keywords = ["call back", "follow-up", "additional information needed", "escalate"]

    # Check for keywords in the transcript
    transcript_lower = transcript.lower()
    if any(phrase in transcript_lower for phrase in resolved_keywords):
        return "Issue Resolved"
    elif any(phrase in transcript_lower for phrase in follow_up_keywords):
        return "Follow-up Action Needed"
    else:
        return "Follow-up Action Needed"  # Default assumption if uncertain

def analyze_transcript(transcript):
    """
    Analyzes the sentiment and outcome of the customer part of a transcript.
    """
    # Extract customer-only lines (assuming "Member" indicates the customer)
    customer_lines = "\n".join([line for line in transcript.splitlines() if line.startswith("Member:")])
    customer_text = re.sub(r"Member:\s*", "", customer_lines)

    # Analyze sentiment
    sentiment_result = sentiment_analyzer(customer_text)
    sentiment = sentiment_result[0]['label']

    # Convert sentiment labels to match the evaluation format
    if sentiment == "POSITIVE":
        sentiment = "Positive"
    elif sentiment == "NEGATIVE":
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Determine call outcome
    outcome = determine_call_outcome(customer_text)

    return sentiment, outcome

def test_consistency(transcript, iterations=5):
    """
    Runs the model multiple times on the same input to test consistency.
    """
    sentiments = []
    for _ in range(iterations):
        prediction = analyze_transcript(transcript)
        sentiment = prediction[0]
        sentiments.append(sentiment)

    # Count occurrences of each label
    label_counts = Counter(sentiments)

    # Most frequent label
    most_common_label, frequency = label_counts.most_common(1)[0]

    consistency_rate = frequency / iterations

    return {
        "most_common_label": most_common_label,
        "consistency_rate": consistency_rate,
        "predictions": sentiments
    }

def main():
    # Load transcripts
    transcript_dir = "data/transcripts_v3"
    transcript_files = [os.path.join(transcript_dir, file) for file in os.listdir(transcript_dir) if
                        file.endswith(".txt")]

    results = []
    for file_path in transcript_files:
        with open(file_path, 'r') as file:
            transcript = file.read()

        # Test consistency for the transcript
        consistency_result = test_consistency(transcript)
        results.append({
            "file": file_path,
            **consistency_result
        })

    # Print results
    for result in results:
        print(f"File: {result['file']}")
        print(f"Most Common Label: {result['most_common_label']}")
        print(f"Consistency Rate: {result['consistency_rate']:.2f}")
        print(f"Predictions: {result['predictions']}")
        print("-" * 50)

if __name__ == "__main__":
    main()

sentiment_analyzer_val = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_transcript_val(transcript):
    """
    Analyzes the sentiment and outcome of the customer part of a transcript using a different model.
    """
    # Extract customer-only lines (assuming "Member" indicates the customer)
    customer_lines = "\n".join([line for line in transcript.splitlines() if line.startswith("Member:")])
    customer_text = re.sub(r"Member:\s*", "", customer_lines)

    # Analyze sentiment
    sentiment_result = sentiment_analyzer_val(customer_text)
    sentiment = sentiment_result[0]['label']

    # Convert sentiment labels to match the evaluation format
    if sentiment == "POSITIVE":
        sentiment = "Positive"
    elif sentiment == "NEGATIVE":
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    # Determine call outcome
    outcome = determine_call_outcome(customer_text)

    return sentiment, outcome

# compare the two models
def evaluate_models(transcripts):
    """
    Evaluates the performance of two sentiment analysis models on the given transcripts.
    """
    predicted_sentiments = []
    predicted_sentiments_val = []
    for file_path in transcripts:
        with open(file_path, 'r') as file:
            transcript = file.read()

        # Model 1
        predicted_sentiment, predicted_outcomes = analyze_transcript(transcript)
        predicted_sentiments.append(predicted_sentiment)

    # Model 2
        predicted_sentiment_val, predicted_outcomes_val = analyze_transcript_val(transcript)
        predicted_sentiments_val.append(predicted_sentiment_val)

        # compare agreement between the two models on sentiment
    sentiment_agreement = [pred1 == pred2 for pred1, pred2 in zip(predicted_sentiments, predicted_sentiments_val)]

    # print agreement rate
    agreement_rate = sum(sentiment_agreement) / len(sentiment_agreement)
    print(f"Sentiment Agreement Rate: {agreement_rate:.2f}")


def main():
    # Load transcripts
    # Load transcripts
    transcript_dir = "data/transcripts_v3"
    transcript_files = [os.path.join(transcript_dir, file) for file in os.listdir(transcript_dir) if
                        file.endswith(".txt")]

    evaluate_models(transcript_files)

if __name__ == "__main__":
    main()
