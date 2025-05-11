import streamlit as st
import pandas as pd
import joblib as jb
from sklearn.neighbors import NearestNeighbors

# Load model and data (cache to make it faster)
@st.cache_resource
def load_model():
    return jb.load('knn_model.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv('user_item_matrix.csv', index_col=0)
    df.index = df.index.astype(str)
    return df

# Load resources
knn = load_model()
user_item_matrix = load_data()

# Recommendation function
def recommend_products(customer_id, knn_model, interaction_matrix, top_n=5):
    customer_id = str(customer_id)
    
    if customer_id in interaction_matrix.index:
        user_vector = interaction_matrix.loc[customer_id].values.reshape(1, -1)
        distances, indices = knn_model.kneighbors(user_vector, n_neighbors=6)  # +1 because first neighbor is user itself
        neighbor_ids = interaction_matrix.index[indices.flatten()[1:]]  # skip self
        
        neighbor_data = interaction_matrix.loc[neighbor_ids]
        product_scores = neighbor_data.sum(axis=0)
        already_purchased = interaction_matrix.loc[customer_id]
        
        # Recommend products not already purchased
        product_scores = product_scores[already_purchased == 0]
        recommended_items = product_scores.sort_values(ascending=False).head(top_n).index.tolist()

        return recommended_items, "existing"
    
    else:
        # New user — show popular products
        popular_items = interaction_matrix.sum(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
        return popular_items, "new"

# Streamlit App
st.title("🛒 Personalized Product Recommendations")

st.markdown("Welcome to your personalized shopping assistant! Enter your **Customer ID** below to see recommendations.")

customer_id_input = st.text_input("🔎 Enter your CustomerID:")

if st.button("Get Recommendations"):
    if customer_id_input.strip() == "":
        st.warning("⚠️ Please enter a valid CustomerID.")
    else:
        recommendations, user_type = recommend_products(customer_id_input, knn, user_item_matrix)
        
        if user_type == "existing":
            st.success("🎯 Found your profile! Here are your personalized recommendations:")
        else:
            st.info("🚀 New user detected! Showing popular products:")

        if recommendations:
            for idx, item in enumerate(recommendations, start=1):
                st.write(f"**{idx}. {item}**")
        else:
            st.warning("😅 No new recommendations available. You've purchased everything!")

# Optional: add footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
