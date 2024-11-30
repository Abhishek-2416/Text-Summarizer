import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import heapq

# Example dataset (you can replace this with actual text data)
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

## First step we should do now is to clean this text by removing everything else except words
# Removing the HTML tags,URL and emails

text = re.sub(r'<.*?>','',text) ## This will remove anything which is in this format <>, example of that is <string>,<h1>
text = re.sub(r'http\S+|www\S+|@\S+','',text) ## This regex is used to remove URLs, email-like patterns, or mentions (e.g., Twitter handles) from a given text.
text = re.sub(r'[^a-zA-Z ]', '', text)
text = text.lower()

####################################################################################################################################################################

## Step 2
## Next what we can do is now we can Tokenize and calculate the word frequency

## Sentences Tokenize
sentences = sent_tokenize(text,language='english')

## Word Tokenization
words = word_tokenize(text)

## Remove Stopwords
for i in range(len(sentences)):
    words = [word for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)


## Get the frequency of words
word_frequency = Counter(words)

## Now we need to normalize these words with respect to the max frequency to get importance of word to that
max_frequency = max(word_frequency.values())

for word in word_frequency:
    word_frequency[word] = word_frequency[word] / max_frequency


#####################################################################################################################################################################

## Step 3 
## Now we need to score the sentences , because we might think that just getting word frequency is important but also we will get the original context from how important which sentence is 

"""
Consider this document:

Sentence 1: "Machine learning is revolutionizing various industries."
Sentence 2: "The cat sat on the mat."
Sentence 3: "AI-powered tools are being developed for a wide range of applications."
Sentence 4: "I love my pet cat."
From a word frequency perspective, words like "machine learning", "AI", and "tools" might be important.
But if we skip sentence scoring, we might end up selecting Sentence 4 (about the cat) just because the word "cat" is repeated, even though it's not really relevant to the topic of the document.
"""

## Now we need to score senteces based on their Word Frequencies
# Step 3: Score the sentences based on word frequencies
sentence_scores = {}

# Calculate scores for each sentence
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

"""# Step 4: Generate the summary by selecting the top 25% sentences
# Select the sentences with the highest scores
top_sentences = heapq.nlargest(int(len(sentences) * 0.25), sentence_scores.items(), key=lambda x: x[1])

# Extract the sentences from the tuples and join them into a summary
summary_sentences = [sentence for sentence, sentence_score in top_sentences]
summary = ' '.join(summary_sentences)

# Print the original and summarized texts
print("\nOriginal Text:\n", text)
print("\nSummarized Text:\n", summary)"""