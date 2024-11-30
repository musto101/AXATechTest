import os
from transformers import pipeline
from sklearn.metrics import classification_report, accuracy_score
import re
import json

# Load a sentiment analysis model locally
sentiment_analyzer = pipeline("sentiment-analysis")

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

def evaluate_model(transcripts, ground_truth):
    """
    Evaluates the model's performance on sentiment and outcome predictions.
    """
    predicted_sentiments = []
    predicted_outcomes = []
    true_sentiments = []
    true_outcomes = []

    for file_path, labels in zip(transcripts, ground_truth):
        with open(file_path, 'r') as file:
            transcript = file.read()

        predicted_sentiment, predicted_outcome = analyze_transcript(transcript)
        predicted_sentiments.append(predicted_sentiment)
        predicted_outcomes.append(predicted_outcome)
        true_sentiments.append(labels["sentiment"])
        true_outcomes.append(labels["outcome"])

    # Calculate metrics for sentiment
    sentiment_report = classification_report(true_sentiments, predicted_sentiments, output_dict=True)
    outcome_report = classification_report(true_outcomes, predicted_outcomes, output_dict=True)

    # Display metrics
    print("Sentiment Analysis Performance:")
    print(json.dumps(sentiment_report, indent=4))
    print("\nCall Outcome Detection Performance:")
    print(json.dumps(outcome_report, indent=4))

    return sentiment_report, outcome_report

def main():
    # Transcripts to evaluate
    transcript_dir = "data/transcripts_v3"
    transcript_files = [os.path.join(transcript_dir, file) for file in os.listdir(transcript_dir) if file.endswith(".txt")]

    # Ground truth for evaluation
    # Each entry contains the true sentiment and outcome for the corresponding transcript
    ground_truth = [
        {"sentiment": "Negative", "outcome": "Issue Resolved"},
        {"sentiment": "Positive", "outcome": "Issue Resolved"},
        {"sentiment": "Positive", "outcome": "Follow-up Action Needed"},
        {"sentiment": "Neutral", "outcome": "Follow-up Action Needed"},
        {"sentiment": "Neutral", "outcome": "Issue Resolved"},
        {"sentiment": "Positive", "outcome": "Issue Resolved"},
        {"sentiment": "Neutral", "outcome": "Follow-up Action Needed"},
        {"sentiment": "Neutral", "outcome": "Follow-up Action Needed"},
        {"sentiment": "Positive", "outcome": "Issue Resolved"},
        {"sentiment": "Neutral", "outcome": "Follow-up Action Needed"}
    ]

    # Evaluate the model
    evaluate_model(transcript_files, ground_truth)

if __name__ == "__main__":
    main()

#
# # Load a sentiment analysis model locally
# sentiment_analyzer = pipeline("sentiment-analysis")
#
# def determine_call_outcome(transcript):
#     """
#     Infers the call outcome based on common phrases.
#     Returns 'Issue Resolved' or 'Follow-up Action Needed'.
#     """
#     resolved_keywords = ["issue resolved", "problem fixed", "refund processed", "resolved", "successful"]
#     follow_up_keywords = ["call back", "follow-up", "additional information needed", "escalate"]
#
#     # Check for keywords in the transcript
#     transcript_lower = transcript.lower()
#     if any(phrase in transcript_lower for phrase in resolved_keywords):
#         return "Issue Resolved"
#     elif any(phrase in transcript_lower for phrase in follow_up_keywords):
#         return "Follow-up Action Needed"
#     else:
#         return "Follow-up Action Needed"  # Default assumption if uncertain
#
# def analyze_transcript(transcript):
#     """
#     Analyzes the sentiment and outcome of the customer part of a transcript.
#     """
#     # Extract customer-only lines (assuming "Member" indicates the customer)
#     customer_lines = "\n".join([line for line in transcript.splitlines() if line.startswith("Member:")])
#     customer_text = re.sub(r"Member:\s*", "", customer_lines)
#
#     # Analyze sentiment
#     sentiment_result = sentiment_analyzer(customer_text)
#     sentiment = sentiment_result[0]['label']
#
#     # Determine call outcome
#     outcome = determine_call_outcome(customer_text)
#
#     return f"Sentiment: {sentiment}\nOutcome: {outcome}"
#
# # Example usage
# def main():
#     # Load transcripts from data/transcripts_v3 folder
#     transcript_dir = "data/transcripts_v3"
#     transcript_files = [os.path.join(transcript_dir, file) for file in os.listdir(transcript_dir) if file.endswith(".txt")]
#
#     for file_path in transcript_files:
#         with open(file_path, 'r') as file:
#             transcript = file.read()
#
#         print(f"Analyzing transcript: {file_path}")
#         analysis = analyze_transcript(transcript)
#         print(analysis)
#         print("-" * 80)
#
# if __name__ == "__main__":
#     main()
