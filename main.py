import streamlit as st
import pandas as pd
import ollama
from mlxtend.frequent_patterns import apriori, association_rules
from transformers import pipeline

# Load sample transactional data (Simulated for ARM)
data = {
    'TransactionID': [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
    'Product': ['Laptop', 'Mouse', 'Laptop', 'Keyboard', 'Tablet', 'Case', 'Keyboard', 'Smartphone', 'Earbuds', 'Smartwatch']
}
df = pd.DataFrame(data)

# Convert transactional data into one-hot encoding
basket = df.pivot_table(index='TransactionID', columns='Product', aggfunc=lambda x: 1, fill_value=0)

# Apply Apriori for frequent itemsets
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Sentiment analysis with BERT (via Ollama)
sentiment_analyzer = pipeline("sentiment-analysis")

def get_recommendations(cart_items):
    """Generate recommendations based on ARM and BERT sentiment analysis."""
    # ARM-based recommendations
    related_items = set()
    for item in cart_items:
        related_rules = rules[rules['antecedents'].apply(lambda x: item in x)]
        for _, row in related_rules.iterrows():
            related_items.update(row['consequents'])
    
    # BERT-based analysis via Ollama Llama 3.2
    user_input = f"User is interested in {', '.join(cart_items)}. Suggest relevant products."
    ai_response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': user_input}])
    ai_suggestions = ai_response['message']['content'].split(',')
    
    return list(related_items), ai_suggestions

# Streamlit UI
st.title("Live Cart Recommendations Using ARM & BERT")

cart = st.multiselect("Select items in your cart:", df['Product'].unique())

if st.button("Get Recommendations"):
    if cart:
        arm_recommendations, ai_recommendations = get_recommendations(cart)
        st.subheader("ARM-Based Recommendations")
        st.write(arm_recommendations)
        st.subheader("AI (BERT) Enhanced Recommendations")
        st.write(ai_recommendations)
    else:
        st.warning("Please select at least one item in your cart.")