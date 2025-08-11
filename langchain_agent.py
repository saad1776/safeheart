import os
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from sqlalchemy import create_engine
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, pearsonr

class CardiovascularAnalysisAgent:
    def __init__(self, db_connection_string):
        """Initialize the LangChain agent for cardiovascular analysis"""
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Initialize database connection
        self.engine = create_engine(db_connection_string)
        self.db = SQLDatabase(self.engine)
        
        # Create SQL toolkit
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create custom tools
        self.custom_tools = self._create_custom_tools()
        
        # Create the agent
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            extra_tools=self.custom_tools
        )
        
        # Define system context
        self.system_context = """
        You are an expert cardiovascular disease analyst with access to a patient database containing 30,000 records.
        
        Key dataset information:
        - Target variable: has_cardiovascular_disease (0/1)
        - Statistically significant features (from Chi-square test): had_stroke, diabetic, hypertensive
        - Non-significant features: gender, income_level, is_poor, is_freedom_fighter
        - Numerical features: age, systolic, diastolic, bmi
        
        Your role is to provide insightful analysis and interpretation of cardiovascular disease patterns,
        explain statistical results in medical context, and suggest further analytical approaches.
        
        Always provide evidence-based responses and acknowledge limitations of the analysis.
        """
    
    def _create_custom_tools(self):
        """Create custom analysis tools"""
        
        def calculate_correlation(feature1: str, feature2: str) -> str:
            """Calculate correlation between two numerical features"""
            try:
                query = f"SELECT {feature1}, {feature2} FROM test_dataset WHERE {feature1} IS NOT NULL AND {feature2} IS NOT NULL LIMIT 5000"
                df = pd.read_sql(query, self.engine)
                
                if len(df) < 10:
                    return f"Insufficient data for correlation analysis between {feature1} and {feature2}"
                
                correlation, p_value = pearsonr(df[feature1], df[feature2])
                
                return f"Correlation between {feature1} and {feature2}: {correlation:.4f} (p-value: {p_value:.4f})"
            except Exception as e:
                return f"Error calculating correlation: {str(e)}"
        
        def perform_chi_square_analysis(feature: str) -> str:
            """Perform Chi-square test for a specific feature"""
            try:
                query = f"SELECT {feature}, has_cardiovascular_disease FROM test_dataset WHERE {feature} IS NOT NULL LIMIT 5000"
                df = pd.read_sql(query, self.engine)
                
                contingency_table = pd.crosstab(df[feature], df['has_cardiovascular_disease'])
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                significance = "Highly significant" if p_value < 0.001 else "Moderately significant" if p_value < 0.01 else "Significant" if p_value < 0.05 else "Not significant"
                
                return f"Chi-square test for {feature}: χ² = {chi2:.4f}, p-value = {p_value:.2e}, {significance}"
            except Exception as e:
                return f"Error performing Chi-square test: {str(e)}"
        
        def get_feature_summary(feature: str) -> str:
            """Get summary statistics for a feature"""
            try:
                query = f"SELECT {feature}, COUNT(*) as count FROM test_dataset WHERE {feature} IS NOT NULL GROUP BY {feature} ORDER BY count DESC LIMIT 10"
                df = pd.read_sql(query, self.engine)
                
                return f"Summary for {feature}:\n" + df.to_string(index=False)
            except Exception as e:
                return f"Error getting feature summary: {str(e)}"
        
        def analyze_risk_groups() -> str:
            """Analyze cardiovascular disease risk by different groups"""
            try:
                query = """
                SELECT 
                    CASE 
                        WHEN age < 40 THEN 'Young (<40)'
                        WHEN age < 60 THEN 'Middle-aged (40-59)'
                        ELSE 'Older (60+)'
                    END as age_group,
                    gender,
                    AVG(has_cardiovascular_disease::float) as cvd_rate,
                    COUNT(*) as total_patients
                FROM test_dataset 
                WHERE age IS NOT NULL AND gender IS NOT NULL
                GROUP BY age_group, gender
                ORDER BY cvd_rate DESC
                """
                df = pd.read_sql(query, self.engine)
                
                return "CVD Risk by Age Group and Gender:\n" + df.to_string(index=False)
            except Exception as e:
                return f"Error analyzing risk groups: {str(e)}"
        
        # Create tool objects
        tools = [
            Tool(
                name="correlation_analysis",
                description="Calculate correlation between two numerical features. Use this to understand relationships between variables like age, systolic, diastolic, bmi.",
                func=calculate_correlation
            ),
            Tool(
                name="chi_square_test",
                description="Perform Chi-square test for a categorical feature against cardiovascular disease. Use this to test statistical significance.",
                func=perform_chi_square_analysis
            ),
            Tool(
                name="feature_summary",
                description="Get summary statistics and distribution for any feature. Use this to understand the data distribution.",
                func=get_feature_summary
            ),
            Tool(
                name="risk_group_analysis",
                description="Analyze cardiovascular disease risk across different demographic groups. Use this for population-level insights.",
                func=analyze_risk_groups
            )
        ]
        
        return tools
    
    def query(self, user_question: str) -> str:
        """Process user query and return analytical response"""
        
        # Enhance the question with context
        enhanced_prompt = f"""
        {self.system_context}
        
        User Question: {user_question}
        
        Please provide a comprehensive analysis addressing the user's question. Use the available tools to gather data and provide evidence-based insights. 
        Consider medical context and statistical significance in your response.
        """
        
        try:
            response = self.agent.run(enhanced_prompt)
            return response
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or ask about a specific aspect of the cardiovascular disease analysis."
    
    def get_predefined_insights(self) -> dict:
        """Get predefined insights about the dataset"""
        
        insights = {
            "significant_features": {
                "title": "Statistically Significant Features",
                "content": """
                Based on Chi-square analysis, three features show strong statistical significance:
                
                1. **Had Stroke** (p ≈ 1.95e-252): Extremely strong association with CVD
                2. **Diabetic** (p ≈ 1.01e-21): Very strong association with CVD  
                3. **Hypertensive** (p ≈ 5.77e-52): Very strong association with CVD
                
                These results align with established medical knowledge about cardiovascular risk factors.
                """
            },
            "non_significant_features": {
                "title": "Non-Significant Features",
                "content": """
                Several features did not show statistical significance:
                
                - **Gender**: May indicate equal risk distribution or insufficient sample size in certain categories
                - **Income Level**: Socioeconomic factors might be more complex than simple income classification
                - **Poverty Status**: Similar to income, may require more nuanced analysis
                - **Freedom Fighter Status**: Likely not a direct medical risk factor
                
                This doesn't mean these factors are unimportant, but they may require different analytical approaches.
                """
            },
            "clinical_implications": {
                "title": "Clinical Implications",
                "content": """
                The significant features represent well-established cardiovascular risk factors:
                
                - **Stroke History**: Strong predictor due to shared vascular pathology
                - **Diabetes**: Contributes to atherosclerosis and vascular damage
                - **Hypertension**: Direct mechanical stress on cardiovascular system
                
                These findings support evidence-based risk stratification and intervention strategies.
                """
            }
        }
        
        return insights

# Example usage and testing
def test_agent():
    """Test the cardiovascular analysis agent"""
    
    # Initialize agent
    db_connection = "postgresql://postgres:123@localhost:5432/test_db"
    agent = CardiovascularAnalysisAgent(db_connection)
    
    # Test queries
    test_questions = [
        "What does it mean that stroke has such a low p-value?",
        "Why might gender not be statistically significant?",
        "What are the clinical implications of these findings?",
        "How do age and blood pressure relate to cardiovascular disease?"
    ]
    
    print("Testing Cardiovascular Analysis Agent:")
    print("=" * 50)
    
    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 30)
        try:
            response = agent.query(question)
            print(f"A: {response}")
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    test_agent()