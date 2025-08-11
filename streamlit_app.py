import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Risk Analysis",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Simplified CSS for clean UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #333;
        margin-bottom: 2rem;
    }
    .section-header {
        padding: 0.5rem 0;
        margin: 1rem 0;
        border-bottom: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Database and Data Utils
# -----------------------

@st.cache_resource
def get_database_connection():
    """Create and cache database connection"""
    return create_engine("postgresql://postgres:123@localhost:5432/test_db")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess raw data to match expected model features."""
    # Clean column names
    df.columns = df.columns.str.replace(" ", "_").str.replace(":", "").str.lower()

    # Rename to consistent names when present
    df = df.rename(columns={
        "profile_hypertensive": "hypertensive",
        "total_income": "income_level"
    })

    # Derive gender_male but keep original gender for charts
    if "gender" in df.columns and "gender_male" not in df.columns:
        df["gender_male"] = df["gender"].apply(lambda x: 1 if str(x).strip().lower() in ["male", "m", "1"] else 0)

    # Derive low_income but keep original income_level for display
    if "income_level" in df.columns and "low_income" not in df.columns:
        df["low_income"] = df["income_level"].apply(lambda x: 1 if str(x).strip().lower() in ["lower class", "low", "lower"] else 0)

    # Cast boolean-like columns
    for col in ["is_poor", "is_freedom_fighter", "had_stroke", "diabetic", "hypertensive", "has_cardiovascular_disease"]:
        if col in df.columns:
            if df[col].dtype == "bool":
                df[col] = df[col].astype(int)
            else:
                df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1"] else (0 if str(x).strip().lower() in ["no", "false", "0"] else x))
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    return df

@st.cache_data
def load_data():
    """Load raw dataset from PostgreSQL and preprocess it."""
    engine = get_database_connection()
    query = "SELECT * FROM test_dataset LIMIT 5000;"
    df = pd.read_sql(query, engine)
    return preprocess_data(df)

# -----------------------
# Statistical Significance
# -----------------------

@st.cache_data
def perform_statistical_analysis():
    """Perform Chi-square test for categorical features against the target."""
    df = load_data()
    target = "has_cardiovascular_disease"
    categorical_features = [
        "gender", "income_level", "is_poor", "is_freedom_fighter",
        "had_stroke", "diabetic", "hypertensive"
    ]

    results = []
    for feature in categorical_features:
        if feature in df.columns and target in df.columns:
            try:
                contingency = pd.crosstab(df[feature], df[target])
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                results.append({
                    "Feature": feature,
                    "Chi-square Statistic": chi2,
                    "P-value": p_value,
                    "Degrees of Freedom": dof,
                    "Statistically Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No",
                    "Significance Level": "High" if p_value < 0.001 else "Medium" if p_value < 0.01 else "Low" if p_value < 0.05 else "Not Significant"
                })
            except Exception as e:
                results.append({
                    "Feature": feature,
                    "Chi-square Statistic": "Error",
                    "P-value": "Error",
                    "Degrees of Freedom": "Error",
                    "Statistically Significant (p < 0.05)": f"Error: {e}",
                    "Significance Level": "Error"
                })

    return pd.DataFrame(results)

# -----------------------
# Model Training
# -----------------------

@st.cache_data
def train_model():
    """Train Random Forest model focusing on key features"""
    df = load_data()
    
    # Focus on key features: had_stroke, diabetic, hypertensive + supporting features
    key_features = ["had_stroke", "diabetic", "hypertensive"]
    supporting_features = ["age", "gender_male", "systolic", "diastolic", "bmi"]
    
    # Check which features are available
    available_features = []
    for feature in key_features + supporting_features:
        if feature in df.columns:
            available_features.append(feature)
    
    if not available_features:
        st.error("No required features found in the dataset")
        return None, None, None, None, None, None
    
    X = df[available_features]
    y = df["has_cardiovascular_disease"]
    
    # Remove rows with missing target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        st.error("No valid data for training")
        return None, None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    feature_importance = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return model, X_test, y_test, y_pred, y_proba, feature_importance

# -----------------------
# AI Agent (Ollama)
# -----------------------

@st.cache_resource
def initialize_ai_agent():
    """Initialize the Ollama-based AI agent"""
    try:
        from langchain_community.llms import Ollama
        from langchain.schema import HumanMessage
        
        # Initialize Ollama with tinyllama model
        llm = Ollama(model="tinyllama")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize Ollama AI agent: {e}")
        st.info("Please install Ollama and pull tinyllama model (e.g., 'ollama pull tinyllama')")
        return None

def query_ai_agent(llm, question, context=""):
    """Query the AI agent with context"""
    if llm is None:
        return "AI agent is not available. Please install Ollama and pull tinyllama model."
    
    try:
        prompt = f"""
        Context: You are analyzing cardiovascular disease data with focus on three key risk factors:
        1. Had Stroke (had_stroke)
        2. Diabetic (diabetic) 
        3. Hypertensive (hypertensive)
        
        Additional context: {context}
        
        Question: {question}
        
        Please provide a clear, concise answer focusing on the medical and statistical significance.
        """
        
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error querying AI agent: Ollama call failed with status code 404. Maybe your model is not found and you should pull the model with ollama pull tinyllama."

# -----------------------
# Main Application
# -----------------------

def main():
    # Simple Header
    st.markdown("""
    <div class="main-header">
        <h1>Cardiovascular Disease Risk Analysis</h1>
        <p>Focus on Key Risk Factors: Stroke, Diabetes, and Hypertension</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data available. Please check your database connection.")
        return

    # Key metrics overview
    st.markdown("""<div class="section-header"><h2>Overview Dashboard</h2></div>""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", f"{len(df):,}")
    with col2:
        cvd_rate = (df["has_cardiovascular_disease"].sum() / max(len(df), 1)) * 100 if "has_cardiovascular_disease" in df.columns else 0
        st.metric("CVD Rate", f"{cvd_rate:.2f}%")
    with col3:
        stroke_rate = (df["had_stroke"].sum() / max(len(df), 1)) * 100 if "had_stroke" in df.columns else 0
        st.metric("Stroke Rate", f"{stroke_rate:.2f}%")
    with col4:
        diabetes_rate = (df["diabetic"].sum() / max(len(df), 1)) * 100 if "diabetic" in df.columns else 0
        st.metric("Diabetes Rate", f"{diabetes_rate:.2f}%")

    # Top 3 Significant Features Chart
    st.markdown("""<div class="section-header"><h3>Top 3 Statistically Significant Risk Factors</h3></div>""", unsafe_allow_html=True)
    
    with st.spinner("Analyzing statistical significance..."):
        results_df = perform_statistical_analysis()
    
    # Get top 3 significant features
    significant_df = results_df[results_df["Statistically Significant (p < 0.05)"] == "Yes"]
    top_3_features = significant_df.nsmallest(3, "P-value").copy()
    
    if not top_3_features.empty:
        # Calculate -log10(P-value) BEFORE plotting
        top_3_features["-log10(P-value)"] = -np.log10(top_3_features["P-value"].replace(0, np.finfo(float).eps))

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Simple bar chart of top 3 features
            fig = px.bar(
                top_3_features,
                x="Feature",
                y="-log10(P-value)",
                title="Top 3 Most Significant Risk Factors"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Key Findings")
            for idx, row in top_3_features.iterrows():
                st.write(f"**{row['Feature'].replace('_', ' ').title()}**")
                st.write(f"P-value: {row['P-value']:.2e}")
                st.write(f"Significance: {row['Significance Level']}")
                st.write("---")
    else:
        st.info("No statistically significant features found at p < 0.05")

    # Simplified tabbed interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Risk Factor Analysis", 
        "Model Performance", 
        "Risk Prediction", 
        "Statistical Details",
        "AI Assistant"
    ])

    with tab1:
        st.markdown("""<div class="section-header"><h3>Key Risk Factors Analysis</h3></div>""", unsafe_allow_html=True)
        
        # Focus on the three key features
        key_features = ["had_stroke", "diabetic", "hypertensive"]
        
        col1, col2, col3 = st.columns(3)
        
        for i, feature in enumerate(key_features):
            if feature in df.columns:
                with [col1, col2, col3][i]:
                    st.subheader(f"{feature.replace('_', ' ').title()}")
                    
                    # Simple distribution chart
                    if "has_cardiovascular_disease" in df.columns:
                        crosstab = pd.crosstab(df[feature], df["has_cardiovascular_disease"])
                        fig = px.bar(
                            x=crosstab.index,
                            y=[crosstab.get(0, pd.Series()), crosstab.get(1, pd.Series())],
                            title=f"{feature.replace('_', ' ').title()} vs CVD"
                        )
                        fig.update_layout(barmode="stack", height=300, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    positive_rate = (df[feature].sum() / len(df)) * 100
                    st.metric(f"{feature.replace('_', ' ').title()} Rate", f"{positive_rate:.1f}%")
                    
                    if "has_cardiovascular_disease" in df.columns:
                        cvd_with_condition = df[df[feature] == 1]["has_cardiovascular_disease"].mean() * 100
                        cvd_without_condition = df[df[feature] == 0]["has_cardiovascular_disease"].mean() * 100
                        st.metric("CVD Rate (with condition)", f"{cvd_with_condition:.1f}%")
                        st.metric("CVD Rate (without condition)", f"{cvd_without_condition:.1f}%")

    with tab2:
        st.markdown("""<div class="section-header"><h3>Model Performance</h3></div>""", unsafe_allow_html=True)
        
        with st.spinner("Training model..."):
            model_results = train_model()
        
        if model_results[0] is not None:
            model, X_test, y_test, y_pred, y_proba, feature_importance = model_results
            
            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                accuracy = float((y_pred == y_test).mean())
                st.metric("Accuracy", f"{accuracy:.3f}")
            with col2:
                auc_score = roc_auc_score(y_test, y_proba)
                st.metric("AUC Score", f"{auc_score:.3f}")
            with col3:
                report = classification_report(y_test, y_pred, output_dict=True)
                precision = report.get("1", {}).get("precision", float("nan"))
                st.metric("Precision", f"{precision:.3f}" if not np.isnan(precision) else "N/A")

            # Feature importance
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Importance")
                fig = px.bar(
                    feature_importance,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Feature Importance in CVD Prediction"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC Curve (AUC = {auc_score:.3f})"))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random"))
                fig.update_layout(
                    title="ROC Curve",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model training failed. Please check your data.")

    with tab3:
        st.markdown("""<div class="section-header"><h3>Individual Risk Prediction</h3></div>""", unsafe_allow_html=True)
        
        model_results = train_model()
        if model_results[0] is not None:
            model = model_results[0]
            
            st.subheader("Enter Patient Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information**")
                age = st.slider("Age", 18, 100, 45)
                gender = st.selectbox("Gender", ["Female", "Male"])
                
                st.markdown("**Key Risk Factors**")
                had_stroke = st.selectbox("Had Stroke", ["No", "Yes"])
                diabetic = st.selectbox("Diabetic", ["No", "Yes"])
                hypertensive = st.selectbox("Hypertensive", ["No", "Yes"])
            
            with col2:
                st.markdown("**Additional Information**")
                systolic = st.slider("Systolic BP", 80, 200, 120)
                diastolic = st.slider("Diastolic BP", 40, 120, 80)
                bmi = st.slider("BMI", 15.0, 50.0, 25.0)
                
                if st.button("Predict Risk"):
                    # Prepare input data
                    input_data = pd.DataFrame({
                        "age": [age],
                        "gender_male": [1 if gender == "Male" else 0],
                        "had_stroke": [1 if had_stroke == "Yes" else 0],
                        "diabetic": [1 if diabetic == "Yes" else 0],
                        "hypertensive": [1 if hypertensive == "Yes" else 0],
                        "systolic": [systolic],
                        "diastolic": [diastolic],
                        "bmi": [bmi]
                    })
                    
                    # Get available features from model
                    available_features = model.feature_names_in_
                    input_data = input_data[available_features]
                    
                    # Make prediction
                    risk_probability = model.predict_proba(input_data)[0][1]
                    risk_level = "High" if risk_probability > 0.7 else "Medium" if risk_probability > 0.3 else "Low"
                    
                    st.success(f"**Cardiovascular Disease Risk: {risk_level}**")
                    st.write(f"Risk Probability: {risk_probability:.2%}")
                    
                    # Risk interpretation
                    if risk_probability > 0.7:
                        st.warning("High risk detected. Please consult with a healthcare professional.")
                    elif risk_probability > 0.3:
                        st.info("Moderate risk. Consider lifestyle modifications and regular check-ups.")
                    else:
                        st.success("Low risk. Maintain healthy lifestyle habits.")

    with tab4:
        st.markdown("""<div class="section-header"><h3>Statistical Analysis Details</h3></div>""", unsafe_allow_html=True)
        
        results_df = perform_statistical_analysis()
        st.dataframe(results_df, use_container_width=True)
        
        st.markdown("**Interpretation:**")
        st.write("- P-value < 0.001: High statistical significance")
        st.write("- P-value < 0.01: Medium statistical significance")
        st.write("- P-value < 0.05: Low statistical significance")
        st.write("- P-value ≥ 0.05: Not statistically significant")

    with tab5:
        st.markdown("""<div class="section-header"><h3>AI Assistant</h3></div>""", unsafe_allow_html=True)
        
        # Initialize AI agent
        llm = initialize_ai_agent()
        
        if llm is not None:
            st.success("AI Assistant is ready! (Using TinyLlama model)")
        else:
            st.error("AI Assistant is not available. Please install Ollama and pull tinyllama model.")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about cardiovascular disease risk factors..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get context from current data
                    df = load_data()
                    context = f"Dataset contains {len(df)} patients with {df['has_cardiovascular_disease'].sum()} CVD cases."
                    
                    response = query_ai_agent(llm, prompt, context)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

