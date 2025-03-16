# data_generator.py
import pandas as pd
import numpy as np

# Predefined spam and non-spam emojis
SPAM_EMOJIS = ["🎉", "🔥", "💰", "🤑", "📢", "‼️","🤯","🚀","🔎","📈","👋"]
NON_SPAM_EMOJIS = ["😊", "🎂", "🙏", "👍", "📧"]

def generate_synthetic_data(num_samples=200):
    spam_text_patterns = [
        "Win a free gift",
        "Click here to claim your prize",
        "Congratulations! You've won",
        "Limited time offer",
        "Urgent! Act now",
        "Free money just for you",
        "Crypto",
        "Immediate",
        "Wow",
        "offer",
        "Free tickets",
        "Lottry",
        "Buy Now",
        "Limited",
        "winner",
        "$",
        "cash price",
        "rummy circle",
        "cash",
        "Register Now!",
        "Rewards",
        "Boosted",
        "No Cost!",
        "Festive",
        "Save your Seat",
        "Reserve",
        "Stand out",
        "Jackpot",
    ]
    non_spam_texts = [
        "Let's catch up tomorrow",
        "Meeting at 3 PM confirmed",
        "Happy Birthday!",
        "Can you review this document?",
        "Your order has been shipped",
        "Thanks for reaching out",
        "My name is",
        "HELLO Everyone",
    ]
    
    data = []
    for _ in range(num_samples // 2):
        # Spam emails
        spam_text = np.random.choice(spam_text_patterns)
        spam_emoji = "".join(np.random.choice(SPAM_EMOJIS, np.random.randint(1, 4)))
        spam_email = f"{spam_text} {spam_emoji}"
        data.append({"message": spam_email, "label": 1})  # Spam label
        
        # Non-spam emails
        non_spam_text = np.random.choice(non_spam_texts)
        non_spam_emoji = "".join(np.random.choice(NON_SPAM_EMOJIS, np.random.randint(0, 3)))
        non_spam_email = f"{non_spam_text} {non_spam_emoji}"
        data.append({"message": non_spam_email, "label": 0})  # Non-spam label

    return pd.DataFrame(data)

