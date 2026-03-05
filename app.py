# ============================================================
# BARISTABENCH HW1 — STREAMLIT APP
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(
    page_title="BaristaBench — Coffee Order Price Predictor",
    page_icon="☕",
    layout="wide"
)

# --- Load Models ---
@st.cache_resource
def load_models():
    lr    = joblib.load("model_linear_regression.pkl")
    dt    = joblib.load("model_decision_tree.pkl")
    rf    = joblib.load("model_random_forest.pkl")
    xgb   = joblib.load("model_xgboost.pkl")
    scaler = joblib.load("scaler.pkl")

    class MLP(nn.Module):
        def __init__(self, input_dim):
            super(MLP, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 128),       nn.ReLU(),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.network(x)

    mlp = MLP(input_dim=14)
    mlp.load_state_dict(torch.load("model_mlp.pt",
                        map_location=torch.device('cpu')))
    mlp.eval()
    return lr, dt, rf, xgb, mlp, scaler

lr, dt, rf, xgb, mlp, scaler = load_models()

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df['parsed']    = df['expected_json'].apply(json.loads)
    df['total_price'] = df['parsed'].apply(lambda x: x['total_price'])

    size_order = {'Short':1,'Tall':2,'Grande':3,'Venti':4,'Trenta':5,None:0}
    change_words     = ['actually','wait','scratch','cancel',
                        'remove','no ','instead','nevermind']
    hesitation_words = ['uh','um','hmm','like','literally']

    df['num_items']      = df['parsed'].apply(lambda x: len(x['items']))
    df['total_quantity'] = df['parsed'].apply(
        lambda x: sum(i['quantity'] for i in x['items']))
    df['total_modifiers'] = df['parsed'].apply(
        lambda x: sum(len(i['modifiers']) for i in x['items']))
    df['has_food'] = df['parsed'].apply(
        lambda x: int(any(i['name'] in [
            'Butter Croissant','Blueberry Muffin','Bagel',
            'Avocado Toast','Bacon Gouda Sandwich'
        ] for i in x['items'])))
    df['has_milk_sub'] = df['parsed'].apply(
        lambda x: int(any(m in [
            'Oat Milk','Almond Milk','Soy Milk',
            'Coconut Milk','Breve','Skim Milk'
        ] for i in x['items'] for m in i['modifiers'])))
    df['has_syrup'] = df['parsed'].apply(
        lambda x: int(any(m in [
            'Vanilla Syrup','Caramel Syrup','Hazelnut Syrup',
            'Peppermint Syrup','Sugar Free Vanilla','Classic Syrup'
        ] for i in x['items'] for m in i['modifiers'])))
    df['has_extra_shot'] = df['parsed'].apply(
        lambda x: int(any(m == 'Extra Shot'
        for i in x['items'] for m in i['modifiers'])))
    df['has_cold_foam'] = df['parsed'].apply(
        lambda x: int(any(m == 'Cold Foam'
        for i in x['items'] for m in i['modifiers'])))
    df['has_whip'] = df['parsed'].apply(
        lambda x: int(any(m == 'Whip Cream'
        for i in x['items'] for m in i['modifiers'])))
    df['has_cancelled_item'] = df['parsed'].apply(
        lambda x: int(len(x['items']) == 0))
    df['avg_size_rank'] = df['parsed'].apply(
        lambda x: np.mean([size_order.get(i['size'], 0)
                           for i in x['items']]) if x['items'] else 0)
    df['num_words'] = df['order'].apply(
        lambda x: len(str(x).split()))
    df['num_changes'] = df['order'].apply(
        lambda x: sum(str(x).lower().count(w) for w in change_words))
    df['num_hesitations'] = df['order'].apply(
        lambda x: sum(str(x).lower().count(w) for w in hesitation_words))
    return df

df = load_data()

feature_cols = [
    'num_items','total_quantity','total_modifiers',
    'has_food','has_milk_sub','has_syrup',
    'has_extra_shot','has_cold_foam','has_whip',
    'has_cancelled_item','avg_size_rank',
    'num_words','num_changes','num_hesitations'
]

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "☕ Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction"
])

# ============================================================
# TAB 1 — EXECUTIVE SUMMARY
# ============================================================
with tab1:
    st.title("☕ BaristaBench: Coffee Order Price Predictor")
    st.subheader("Can we predict what a customer will be charged "
                 "before they finish ordering?")

    st.markdown("""
    ### 📋 The Dataset
    **BaristaBench** contains **500 real-world style coffee orders** — messy,
    spoken-language orders like *"Lemme get a venti latte, actually make that
    decaf, with oat milk and no foam."* Each order comes with a structured
    ground truth showing exactly what was ordered and what it costs.

    We engineered **14 features** from these orders — capturing item counts,
    sizes, modifiers, and even how often customers changed their mind — and
    trained machine learning models to predict the **total order price**.

    ---
    ### 💡 Why This Matters
    Accurate price prediction enables:
    - **POS error detection** — flag orders where the charged price
      deviates from the predicted price
    - **Revenue forecasting** — estimate transaction value in real time
    - **Drive-through optimization** — predict order complexity before
      the customer reaches the window

    ---
    ### 🔑 Key Findings
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size",       "500 orders")
    col2.metric("Features Engineered","14")
    col3.metric("Best Model",         "Random Forest")
    col4.metric("Best R²",            "0.9154")

    st.markdown("""
    ---
    ### 🏆 Approach
    1. **Descriptive Analytics** — explored price distributions,
       feature relationships, and correlations
    2. **5 ML Models** — Linear Regression, Decision Tree,
       Random Forest, XGBoost, Neural Network (MLP)
    3. **SHAP Explainability** — identified that total quantity,
       average size, and modifier count drive prices most
    4. **Interactive Prediction** — try Tab 4 to predict any
       custom order price in real time!
    """)

# ============================================================
# TAB 2 — DESCRIPTIVE ANALYTICS
# ============================================================
with tab2:
    st.title("📊 Descriptive Analytics")

    st.subheader("Target Distribution — Total Order Price")
    st.image("plot_target_distribution.png")
    st.caption("""The target variable is right-skewed — most orders fall
    in the $10–40 range, with a long tail for large multi-item orders.
    Mean price is ~$26.50.""")

    st.divider()

    st.subheader("Feature Distributions & Relationships")
    st.image("plot_feature_distributions.png")
    st.caption("""Top-left: total quantity strongly predicts price.
    Top-right: more items = higher median price.
    Bottom-left: food orders skew higher in price.
    Bottom-right: more modifiers correlate with higher price.""")

    st.divider()

    st.subheader("Correlation Heatmap")
    st.image("plot_correlation_heatmap.png")
    st.caption("""total_quantity and num_items are most correlated
    with price. Text features (num_words, num_changes) show weaker
    but non-zero correlation.""")

# ============================================================
# TAB 3 — MODEL PERFORMANCE
# ============================================================
with tab3:
    st.title("🤖 Model Performance")

    # Comparison table
    st.subheader("Model Comparison Summary")
    results_df = pd.read_csv("model_comparison.csv")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("RMSE & R² Comparison")
    st.image("plot_model_comparison.png")

    st.divider()
    st.subheader("Best Hyperparameters")
    params = {
        "Linear Regression" : "No hyperparameters tuned (baseline)",
        "Ridge Regression"  : "alpha = 1.0",
        "Lasso Regression"  : "alpha = 0.1",
        "Decision Tree"     : "max_depth=7, min_samples_leaf=10",
        "Random Forest"     : "max_depth=5, n_estimators=200",
        "XGBoost"           : "learning_rate=0.1, max_depth=3, n_estimators=50",
        "Neural Network"    : "2 hidden layers (128 units), ReLU, Adam lr=0.001"
    }
    for model, param in params.items():
        st.markdown(f"**{model}:** {param}")

    st.divider()
    st.subheader("Predicted vs Actual Plots")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Linear Models**")
        st.image("plot_linear_models.png")
        st.markdown("**Decision Tree**")
        st.image("plot_decision_tree_pred.png")
        st.markdown("**XGBoost**")
        st.image("plot_xgboost_pred.png")
    with col2:
        st.markdown("**Random Forest**")
        st.image("plot_random_forest_pred.png")
        st.markdown("**Neural Network**")
        st.image("plot_mlp_pred.png")
        st.markdown("**MLP Training History**")
        st.image("plot_mlp_training.png")

    st.divider()
    st.subheader("Feature Importances")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        st.image("plot_random_forest_importance.png")
    with col2:
        st.markdown("**XGBoost**")
        st.image("plot_xgboost_importance.png")

# ============================================================
# TAB 4 — EXPLAINABILITY & INTERACTIVE PREDICTION
# ============================================================
with tab4:
    st.title("🔍 Explainability & Interactive Prediction")

    # SHAP plots
    st.subheader("SHAP Analysis — Random Forest")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Beeswarm Summary Plot**")
        st.image("plot_shap_beeswarm.png")
        st.caption("""Features ranked by impact. Red = high feature value,
        Blue = low. total_quantity and avg_size_rank push prices up most.""")
    with col2:
        st.markdown("**Feature Importance Bar Plot**")
        st.image("plot_shap_bar.png")
        st.caption("""Mean absolute SHAP values confirm total_quantity,
        avg_size_rank, total_modifiers, and has_food as top drivers.""")

    st.subheader("Waterfall Plot — Most Complex Order")
    st.image("plot_shap_waterfall.png")
    st.caption("""Shows how each feature pushes the prediction above or
    below the baseline for the highest-value order in the test set.""")

    st.divider()

    # Interactive Prediction
    st.subheader("🎮 Interactive Price Predictor")
    st.markdown("Set your order details below and see the predicted price!")

    col1, col2, col3 = st.columns(3)

    with col1:
        num_items      = st.slider("Number of distinct items",  0, 4,  2)
        total_quantity = st.slider("Total quantity of items",   0, 15, 4)
        total_modifiers= st.slider("Total number of modifiers", 0, 8,  1)
        avg_size_rank  = st.select_slider(
            "Average size",
            options=[0, 1, 2, 3, 4, 5],
            value=2,
            format_func=lambda x: {
                0:'None',1:'Short',2:'Tall',
                3:'Grande',4:'Venti',5:'Trenta'}[x])

    with col2:
        has_food        = st.checkbox("Contains food item?")
        has_milk_sub    = st.checkbox("Milk substitution? (Oat, Almond...)")
        has_syrup       = st.checkbox("Has syrup add-on?")
        has_extra_shot  = st.checkbox("Extra espresso shot?")

    with col3:
        has_cold_foam       = st.checkbox("Cold foam?")
        has_whip            = st.checkbox("Whipped cream?")
        has_cancelled_item  = st.checkbox("Order was cancelled?")
        num_words       = st.slider("Words in order",      5, 85, 20)
        num_changes     = st.slider("Number of changes",   0, 15,  1)
        num_hesitations = st.slider("Number of hesitations",0, 4,  1)

    # Model selector
    model_choice = st.selectbox(
        "Select model for prediction:",
        ["Random Forest","XGBoost","Lasso Regression",
         "Linear Regression","Decision Tree","Neural Network (MLP)"]
    )

    # Build input
    input_data = np.array([[
        num_items, total_quantity, total_modifiers,
        int(has_food), int(has_milk_sub), int(has_syrup),
        int(has_extra_shot), int(has_cold_foam), int(has_whip),
        int(has_cancelled_item), avg_size_rank,
        num_words, num_changes, num_hesitations
    ]])

    input_scaled = scaler.transform(input_data)

    # Predict
    model_map = {
        "Random Forest"      : lambda: rf.predict(input_data)[0],
        "XGBoost"            : lambda: xgb.predict(input_data)[0],
        "Lasso Regression"   : lambda: lasso.predict(input_scaled)[0],
        "Linear Regression"  : lambda: lr.predict(input_scaled)[0],
        "Decision Tree"      : lambda: dt.predict(input_data)[0],
        "Neural Network (MLP)": lambda: mlp(
            torch.tensor(input_scaled, dtype=torch.float32)
        ).item()
    }

    predicted_price = model_map[model_choice]()
    predicted_price = max(0, predicted_price)

    st.markdown("---")
    st.metric(
        label=f"💰 Predicted Order Price ({model_choice})",
        value=f"${predicted_price:.2f}"
    )

    # SHAP waterfall for custom input
    st.subheader("SHAP Explanation for Your Input")
    explainer_tab4  = shap.TreeExplainer(rf)
    input_df        = pd.DataFrame(input_data, columns=feature_cols)
    shap_input      = explainer_tab4(input_df)

    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_input[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest**")
        st.image("plot_random_forest_importance.png")
    with col2:
        st.markdown("**XGBoost**")
        st.image("plot_xgboost_importance.png")
        st.divider()
    st.subheader("🎯 Bonus: Neural Network Hyperparameter Tuning")
    st.image("plot_mlp_tuning.png")
    tuning_df = pd.read_csv("mlp_tuning_results.csv")
    st.markdown("**Top 5 Configurations:**")
    st.dataframe(tuning_df.head(), use_container_width=True)
    st.markdown("""
    **Finding:** Low learning rate (0.001) consistently wins across all 
    hidden sizes. Larger networks (256 units) with dropout=0.2 perform best,
    suggesting regularization helps on this small dataset.
    """)