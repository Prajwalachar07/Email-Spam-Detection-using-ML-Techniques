import pandas as pd
import random

# Define spam and non-spam text patterns
spam_text_patterns = [
    "Win a free gift",
    "Click here to claim your prize",
    "Congratulations! You've won",
    "Limited time offer",
    "Urgent! Act now",
    "Free money just for you",
]

spam_emojis = ["🎉", "🔥", "💰", "🤑", "📢", "‼"]
non_spam_texts = [
    "Let's catch up tomorrow",
    "Meeting at 3 PM confirmed",
    "Happy Birthday! 🎂",
    "Can you review this document?",
    "Your order has been shipped",
    "Thanks for reaching out",
]
non_spam_emojis = ["😊", "🎂", "🙏", "👍", "📧"]

# Function to create random spam and non-spam examples
def generate_data(num_samples=100):
    data = []
    for _ in range(num_samples // 2):
        # Generate spam email
        spam_text = random.choice(spam_text_patterns)
        spam_emoji = "".join(random.choices(spam_emojis, k=random.randint(1, 3)))
        spam_email = f"{spam_text} {spam_emoji}"
        data.append({"email_text": spam_email, "label": "spam"})

# Generate non-spam email
        non_spam_text = random.choice(non_spam_texts)
        non_spam_emoji = "".join(random.choices(non_spam_emojis, k=random.randint(0, 2)))
        non_spam_email = f"{non_spam_text} {non_spam_emoji}"
        data.append({"email_text": non_spam_email, "label": "not spam"})
    
    return data

# Generate the dataset
dataset = generate_data(200)

# Convert to a DataFrame
df = pd.DataFrame(dataset)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a CSV file
df.to_csv("spam_dataset_with_emojis.csv", index=False)

# Display the first few rows
print(df.head())