import pandas as pd
from sqlalchemy import create_engine
import psycopg2

def demonstrate_postgres_filtering():
    """
    Demonstrate various PostgreSQL filtering techniques for creating personalized datasets
    """
    # Create connection to PostgreSQL
    engine = create_engine('postgresql://cardiovascular_user:password123@localhost/cardiovascular_db')
    
    print("=== PostgreSQL Data Filtering Examples ===\n")
    
    # Example 1: Basic demographic filtering
    print("1. Filtering by Demographics (Age and Gender)")
    query1 = """
    SELECT user_id, profile_name, age, gender, has_cardiovascular_disease
    FROM cardiovascular_data 
    WHERE age BETWEEN 40 AND 60 
    AND gender = 'Male'
    LIMIT 10;
    """
    result1 = pd.read_sql(query1, engine)
    print(f"Found {len(result1)} records for males aged 40-60:")
    print(result1[['user_id', 'age', 'gender', 'has_cardiovascular_disease']])
    print()
    
    # Example 2: Health condition filtering
    print("2. Filtering by Health Conditions")
    query2 = """
    SELECT user_id, profile_name, age, gender, diabetic, profile_hypertensive, has_cardiovascular_disease
    FROM cardiovascular_data 
    WHERE diabetic = true 
    OR profile_hypertensive = true
    LIMIT 10;
    """
    result2 = pd.read_sql(query2, engine)
    print(f"Found {len(result2)} records with diabetes or hypertension:")
    print(result2[['user_id', 'age', 'diabetic', 'profile_hypertensive', 'has_cardiovascular_disease']])
    print()
    
    # Example 3: Socioeconomic filtering
    print("3. Filtering by Socioeconomic Status")
    query3 = """
    SELECT user_id, profile_name, total_income, is_poor, union_name, has_cardiovascular_disease
    FROM cardiovascular_data 
    WHERE total_income = 'Lower class' 
    AND is_poor = 1
    LIMIT 10;
    """
    result3 = pd.read_sql(query3, engine)
    print(f"Found {len(result3)} records for lower class, poor individuals:")
    print(result3[['user_id', 'total_income', 'is_poor', 'has_cardiovascular_disease']])
    print()
    
    # Example 4: Clinical measurements filtering
    print("4. Filtering by Clinical Measurements")
    query4 = """
    SELECT user_id, profile_name, age, systolic, diastolic, result_stat_bp, has_cardiovascular_disease
    FROM cardiovascular_data 
    WHERE systolic IS NOT NULL 
    AND diastolic IS NOT NULL
    AND (result_stat_bp = 'Hypertension' OR result_stat_bp = 'Prehypertension')
    LIMIT 10;
    """
    result4 = pd.read_sql(query4, engine)
    print(f"Found {len(result4)} records with high blood pressure:")
    print(result4[['user_id', 'age', 'systolic', 'diastolic', 'result_stat_bp', 'has_cardiovascular_disease']])
    print()
    
    # Example 5: Complex filtering for high-risk individuals
    print("5. Complex Filtering for High-Risk Cardiovascular Patients")
    query5 = """
    SELECT user_id, profile_name, age, gender, diabetic, profile_hypertensive, 
           systolic, diastolic, result_stat_bp, has_cardiovascular_disease
    FROM cardiovascular_data 
    WHERE age > 50 
    AND (diabetic = true OR profile_hypertensive = true)
    AND systolic IS NOT NULL
    AND has_cardiovascular_disease = 1
    LIMIT 15;
    """
    result5 = pd.read_sql(query5, engine)
    print(f"Found {len(result5)} high-risk cardiovascular patients:")
    print(result5[['user_id', 'age', 'gender', 'diabetic', 'profile_hypertensive', 'has_cardiovascular_disease']])
    print()
    
    # Example 6: Creating a personalized view/table
    print("6. Creating a Personalized View for Analysis")
    create_view_query = """
    CREATE OR REPLACE VIEW high_risk_patients AS
    SELECT 
        user_id,
        profile_name,
        age,
        gender,
        total_income,
        diabetic,
        profile_hypertensive,
        systolic,
        diastolic,
        result_stat_bp,
        bmi,
        result_stat_bmi,
        has_cardiovascular_disease,
        CASE 
            WHEN age > 60 THEN 'High Age Risk'
            WHEN age > 40 THEN 'Medium Age Risk'
            ELSE 'Low Age Risk'
        END as age_risk_category,
        CASE 
            WHEN diabetic = true AND profile_hypertensive = true THEN 'Very High Risk'
            WHEN diabetic = true OR profile_hypertensive = true THEN 'High Risk'
            ELSE 'Normal Risk'
        END as health_risk_category
    FROM cardiovascular_data
    WHERE systolic IS NOT NULL 
    AND diastolic IS NOT NULL;
    """
    
    # Execute the view creation
    with engine.connect() as conn:
        conn.execute(create_view_query)
        conn.commit()
    
    print("Created 'high_risk_patients' view successfully!")
    
    # Query the new view
    view_query = """
    SELECT user_id, age, gender, age_risk_category, health_risk_category, has_cardiovascular_disease
    FROM high_risk_patients 
    WHERE health_risk_category = 'Very High Risk'
    LIMIT 10;
    """
    view_result = pd.read_sql(view_query, engine)
    print(f"Sample from high_risk_patients view ({len(view_result)} very high risk patients):")
    print(view_result)
    print()
    
    # Example 7: Aggregation for insights
    print("7. Aggregation Queries for Insights")
    agg_query = """
    SELECT 
        gender,
        COUNT(*) as total_patients,
        SUM(CASE WHEN has_cardiovascular_disease = 1 THEN 1 ELSE 0 END) as cvd_patients,
        ROUND(
            (SUM(CASE WHEN has_cardiovascular_disease = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)), 2
        ) as cvd_percentage
    FROM cardiovascular_data
    GROUP BY gender;
    """
    agg_result = pd.read_sql(agg_query, engine)
    print("Cardiovascular disease prevalence by gender:")
    print(agg_result)
    print()
    
    return True

def create_personalized_datasets():
    """
    Create specific datasets for different analysis purposes
    """
    engine = create_engine('postgresql://cardiovascular_user:password123@localhost/cardiovascular_db')
    
    print("=== Creating Personalized Datasets ===\n")
    
    # Dataset 1: Complete cases for machine learning
    print("1. Creating complete cases dataset for ML modeling")
    ml_query = """
    CREATE TABLE IF NOT EXISTS ml_ready_data AS
    SELECT 
        user_id,
        age,
        CASE WHEN gender = 'Male' THEN 1 ELSE 0 END as gender_male,
        CASE WHEN total_income = 'Lower class' THEN 1 ELSE 0 END as low_income,
        is_poor,
        is_freedom_fighter,
        had_stroke,
        CASE WHEN diabetic = true THEN 1 ELSE 0 END as diabetic,
        CASE WHEN profile_hypertensive = true THEN 1 ELSE 0 END as hypertensive,
        systolic,
        diastolic,
        bmi,
        has_cardiovascular_disease
    FROM cardiovascular_data
    WHERE systolic IS NOT NULL 
    AND diastolic IS NOT NULL 
    AND bmi IS NOT NULL
    AND age IS NOT NULL;
    """
    
    with engine.connect() as conn:
        conn.execute(ml_query)
        conn.commit()
    
    # Check the created table
    check_query = "SELECT COUNT(*) as complete_cases FROM ml_ready_data;"
    result = pd.read_sql(check_query, engine)
    print(f"Created ML-ready dataset with {result.iloc[0]['complete_cases']} complete cases")
    
    # Dataset 2: High-risk population for targeted intervention
    print("\n2. Creating high-risk population dataset")
    high_risk_query = """
    CREATE TABLE IF NOT EXISTS high_risk_population AS
    SELECT *
    FROM cardiovascular_data
    WHERE (age > 50 OR diabetic = true OR profile_hypertensive = true)
    AND systolic IS NOT NULL;
    """
    
    with engine.connect() as conn:
        conn.execute(high_risk_query)
        conn.commit()
    
    check_query2 = "SELECT COUNT(*) as high_risk_count FROM high_risk_population;"
    result2 = pd.read_sql(check_query2, engine)
    print(f"Created high-risk population dataset with {result2.iloc[0]['high_risk_count']} individuals")
    
    print("\n✅ Personalized datasets created successfully!")
    
    return True

if __name__ == "__main__":
    print("Starting PostgreSQL filtering demonstrations...\n")
    
    # Run filtering examples
    demonstrate_postgres_filtering()
    
    # Create personalized datasets
    create_personalized_datasets()
    
    print("\n🎉 All PostgreSQL filtering examples completed successfully!")

