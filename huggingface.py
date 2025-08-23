# Sample sentiment analysis using Hugging Face Transformers
from transformers import pipeline

# Specify model and revision for production use
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
revision = "main"

classifier = pipeline(
    "sentiment-analysis",
    model=model_name,
    revision=revision
)

# Example text
text = "I hate learning AI with Python and Hugging Face!"

# Get prediction
result = classifier(text)

print("Text:", text)
print("Sentiment:", result[0]['label'])
print("Score:", result[0]['score'])
