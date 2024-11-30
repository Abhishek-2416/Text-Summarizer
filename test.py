import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import heapq

# Example dataset
text = """
Machine learning is transforming industries worldwide.
AI-powered tools are becoming essential in tech.
Cats are wonderful pets and make great companions.
I love programming in Python and using AI to solve problems.
Technology is evolving rapidly, and so is the field of AI.
Data science is closely related to machine learning.
The future of AI is exciting and full of potential.
The integration of AI in various sectors is accelerating.
Python is one of the best programming languages.
AI is revolutionizing healthcare.
AI can predict diseases with high accuracy.
Machine learning models are improving every day.
Data analytics is crucial for businesses today.
Robotics is a field that AI is heavily influencing.
AI-driven robots are being used in industries.
AI applications are growing at an exponential rate.
"""

####################################################################################################################################################################

# Step 1: Clean the text by removing unwanted characters
text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
text = re.sub(r'http\S+|www\S+|@\S+', '', text)  # Remove URLs and email-like patterns
text = re.sub(r'[^a-zA-Z ]', ' ', text)  # Remove non-alphabetical characters and preserve spaces
text = text.lower()  # Convert to lowercase

# Step 2: Tokenize and calculate word frequencies

# Tokenize sentences
sentences = sent_tokenize(text, language='english')

# Tokenize words and remove stopwords
words = word_tokenize(text)
words = [word for word in words if word not in set(stopwords.words('english'))]

# Get word frequencies
word_frequency = Counter(words)

# Normalize word frequencies (relative importance)
max_frequency = max(word_frequency.values())
for word in word_frequency:
    word_frequency[word] = word_frequency[word] / max_frequency

#####################################################################################################################################################################

# Step 3: Score the sentences based on word frequencies
sentence_scores = {}

for sent in sentences:
    sentence_score = 0
    for word in word_tokenize(sent.lower()):
        if word in word_frequency:  # Check if the word is in the word frequency list
            sentence_score += word_frequency[word]  # Sum word frequencies in the sentence
    sentence_scores[sent] = sentence_score  # Store sentence score

# Display all sentences and their scores
print("All Sentences and their Scores:")
for sentence, score in sentence_scores.items():
    print(f"Sentence: {sentence}, Score: {score:.3f}")

# Rank sentences by their score
ranked_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

# Display ranked sentences
print("\nRanked Sentences:")
for sentence, score in ranked_sentences:
    print(f"Sentence: {sentence}, Score: {score:.3f}")

#####################################################################################################################################################################

# Step 4: Generate the summary by selecting the top 25% sentences
# Select the sentences with the highest scores
top_sentences = heapq.nlargest(int(len(sentences) * 0.25), sentence_scores.items(), key=lambda x: x[1])

# Extract the sentences from the tuples and join them into a summary
summary_sentences = [sentence for sentence, score in top_sentences]
summary = ' '.join(summary_sentences)

# Print the original and summarized texts
print("\nOriginal Text:\n", text)
print("\nSummarized Text:\n", summary)
