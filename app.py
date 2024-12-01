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
text = """Deep learning (also known as deep structured learning) is part of a 
broader family of machine learning methods based on artificial neural networks with 
representation learning. Learning can be supervised, semi-supervised or unsupervised. 
Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, 
recurrent neural networks and convolutional neural networks have been applied to
fields including computer vision, speech recognition, natural language processing, 
machine translation, bioinformatics, drug design, medical image analysis, material
inspection and board game programs, where they have produced results comparable to 
and in some cases surpassing human expert performance. Artificial neural networks
(ANNs) were inspired by information processing and distributed communication nodes
in biological systems. ANNs have various differences from biological brains. Specifically, 
neural networks tend to be static and symbolic, while the biological brain of most living organisms
is dynamic (plastic) and analogue. The adjective "deep" in deep learning refers to the use of multiple
layers in the network. Early work showed that a linear perceptron cannot be a universal classifier, 
but that a network with a nonpolynomial activation function with one hidden layer of unbounded width can.
Deep learning is a modern variation which is concerned with an unbounded number of layers of bounded size, 
which permits practical application and optimized implementation, while retaining theoretical universality 
under mild conditions. In deep learning the layers are also permitted to be heterogeneous and to deviate widely 
from biologically informed connectionist models, for the sake of efficiency, trainability and understandability, 
whence the structured part. 99 @abhishekalimchadani wwww.abhishekalimchandani.com """

####################################################################################################################################################################

## First step we should do now is to clean this text by removing everything else except words
# Removing the HTML tags,URL and emails

text = re.sub(r'<.*?>','',text) ## This will remove anything which is in this format <>, example of that is <string>,<h1>
text = re.sub(r'http\S+|www\S+|@\S+','',text) ## This regex is used to remove URLs, email-like patterns, or mentions (e.g., Twitter handles) from a given text.
text = re.sub(r'[^a-zA-Z.!? ]', '', text) ## Here my mistake was i removing . ! and ? also which made it a problem
text = text.lower()

####################################################################################################################################################################

## Step 2
## Next what we can do is now we can Tokenize and calculate the word frequency

# Step 1: Sentence Tokenization
sentences = sent_tokenize(text, language='english')

# Step 2: Word Tokenization (and stopword removal for individual sentences)
stop_words = set(stopwords.words('english'))
processed_sentences = []
words = []

for sentence in sentences:
    # Tokenize words in each sentence and remove stopwords
    tokenized_words = word_tokenize(sentence)
    filtered_words = [word for word in tokenized_words if word.lower() not in stop_words and word.isalpha()]
    words.extend(filtered_words)  # Collect all words for frequency calculation
    processed_sentences.append(' '.join(filtered_words))  # Rebuild sentence without stopwords

# Step 3: Get the Frequency of Words
word_frequency = Counter(words)

# Step 4: Normalize Frequencies
max_frequency = max(word_frequency.values())
for word in word_frequency:
    word_frequency[word] /= max_frequency

print("Normalized Word Frequencies:")
print(word_frequency)

    
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
for sent in sentences:
    sentence_score = 0
    sentence_words = word_tokenize(sent)  # Tokenize words in the sentence
    for word in sentence_words:
        if word in word_frequency:
            sentence_score += word_frequency[word]  # Add word frequency to the sentence score
    sentence_scores[sent] = sentence_score

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
summary_sentences = [sentence for sentence, sentence_score in top_sentences]
summary = ' '.join(summary_sentences)

# Print the original and summarized texts
print("\nOriginal Text:\n", text)
print("\nSummarized Text:\n", summary)