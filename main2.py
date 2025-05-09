import streamlit as st
import pandas as pd
import ollama
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Simulated transaction dataset with stronger item associations
transactions = [
    ['Laptop', 'Mouse', 'Laptop Bag'],
    ['Laptop', 'Mouse', 'Keyboard'],
    ['Smartphone', 'Earbuds', 'Charger'],
    ['Smartphone', 'Phone Case', 'Screen Protector'],
    ['Tablet', 'Tablet Cover', 'Stylus'],
    ['Smartwatch', 'Earbuds'],
    ['Laptop', 'Keyboard', 'Monitor'],
    ['Monitor', 'HDMI Cable'],
    ['Smartphone', 'Smartwatch'],
    ['Laptop', 'Mouse', 'Monitor'],
    ['Tablet', 'Stylus'],
    ['Smartwatch', 'Charger'],
    ['Phone Case', 'Screen Protector'],
    ['Smartphone', 'Earbuds'],
]

# Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
basket = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori algorithm
frequent_itemsets = apriori(basket, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

# Define hybrid recommender
def get_recommendations(cart_items):
    """Hybrid recommendations from ARM and BERT."""
    # ARM Recommendations
    arm_suggestions = set()
    for item in cart_items:
        matching_rules = rules[rules['antecedents'].apply(lambda x: item in x)]
        for _, row in matching_rules.iterrows():
            arm_suggestions.update(row['consequents'])

    # LLM Recommendations via Ollama (mocked if unavailable)
    user_query = f"User is interested in {', '.join(cart_items)}. Recommend complementary products relevant to it. In a good format, with title of the product and crisp description."
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': user_query}])
        ai_suggestions = response['message']['content'].split(',')
    except Exception as e:
        ai_suggestions = ["Power Bank", "USB Hub", "Bluetooth Speaker"]  # fallback
        print("Ollama error:", e)

    # Combine & clean hybrid recommendations
    combined = list(set(arm_suggestions).union([item.strip() for item in ai_suggestions]))
    combined = [item for item in combined if item not in cart_items]  # remove already selected items

    return list(arm_suggestions), ai_suggestions, combined

# Streamlit UI
st.title("🛒 Smart Product Recommendations (ARM + BERT Hybrid)")

cart = st.multiselect("Select items in your shopping cart:", te.columns_)

if st.button("Get Recommendations"):
    if cart:
        arm, ai, hybrid = get_recommendations(cart)
        st.subheader("🧠 ARM-Based Recommendations")
        st.write(arm if arm else "No strong associations found.")

        st.subheader("🤖 BERT Suggestions (via LLaMA 3.2)")
        st.write(ai)

        st.subheader("💡 Hybrid Final Recommendations")
        st.success(hybrid if hybrid else "No recommendations found.")
    else:
        st.warning("Please select at least one item.")
