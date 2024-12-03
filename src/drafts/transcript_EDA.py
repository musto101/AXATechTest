import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os


# Load the transcripts into a DataFrame
def load_transcripts(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            transcript = file.read()
            transcript_id = file_path.split("/")[-1]  # Use filename as ID
            for line in transcript.split("\n"):
                line = line.strip()  # Remove leading/trailing whitespaces
                if ":" in line:  # Ensure the line contains the expected delimiter
                    speaker, text = line.split(":", 1)
                    data.append({"Transcript ID": transcript_id, "Speaker": speaker.strip(), "Text": text.strip()})
    return pd.DataFrame(data)


# Perform EDA on the DataFrame
def perform_eda(df):
    # Top keywords and frequent words
    all_text = " ".join(df["Text"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

    # Sentiment analysis (basic positive/negative keyword count)
    positive_keywords = ["thank", "great", "help", "appreciate", "welcome", "good"]
    negative_keywords = ["frustration", "mistake", "issue", "problem", "error", "inconvenience"]

    df["Positive"] = df["Text"].apply(lambda x: sum(kw in x.lower() for kw in positive_keywords))
    df["Negative"] = df["Text"].apply(lambda x: sum(kw in x.lower() for kw in negative_keywords))
    sentiment_summary = df.groupby("Speaker")[["Positive", "Negative"]].sum()

    # Display visualizations
    print("Word Cloud for Transcripts:")
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # save the word cloud
    plt.savefig("data/figures/wordcloud.png")

    print("Sentiment Analysis Summary:")
    print(sentiment_summary)

    # Most common issues
    issues = df[df["Speaker"] == "Member"]["Text"].value_counts().head(10)
    print("Most Common Issues Raised by Customers:")
    print(issues)


def plot_sentiment_over_call(df):
    sentiment_data = df[df["Speaker"] == "Member"]
    sentiment_data["Cumulative Positive"] = sentiment_data["Positive"].cumsum()
    sentiment_data["Cumulative Negative"] = sentiment_data["Negative"].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(sentiment_data["Cumulative Positive"], label="Positive Sentiment")
    plt.plot(sentiment_data["Cumulative Negative"], label="Negative Sentiment", linestyle="--")
    plt.title("Sentiment Over Call Duration")
    plt.xlabel("Call Progression")
    plt.ylabel("Cumulative Sentiment Score")
    plt.legend()
    # save the plot
    plt.savefig("data/figures/sentiment_over_call.png")



def plot_speaker_interactions(df):
    """
    Creates a bar chart to visualize the frequency of each speaker's interactions.

    Parameters:
    - df: pandas DataFrame with a column "Speaker" containing speaker names.

    Returns:
    - None (saves the bar chart)
    """
    # Count the frequency of interactions by each speaker
    speaker_counts = df["Speaker"].value_counts()

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    speaker_counts.plot(kind="bar")

    # Add titles and labels
    plt.title("Frequency of Speaker Interactions", fontsize=16)
    plt.xlabel("Speaker", fontsize=14)
    plt.ylabel("Number of Interactions", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig("data/figures/speaker_interactions.png")

# Main
transcript_dir = "data/transcripts_v3"
file_paths = [os.path.join(transcript_dir, file) for file in os.listdir(transcript_dir) if
                        file.endswith(".txt")]

# Load transcripts and perform EDA
df_transcripts = load_transcripts(file_paths)
perform_eda(df_transcripts)

# Plot sentiment over call duration
plot_sentiment_over_call(df_transcripts)

# Plot speaker interactions
plot_speaker_interactions(df_transcripts)
