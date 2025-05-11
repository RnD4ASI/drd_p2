"""
Script to generate sample data for testing the deduplication pilots.
This creates a Parquet file with sample data attributes and definitions,
including intentional semantic duplications.
"""

import pandas as pd
import numpy as np
import os

def generate_sample_data(output_path, num_samples=100):
    """
    Generate sample data attributes and definitions with intentional duplications.
    
    Args:
        output_path (str): Path to save the generated Parquet file
        num_samples (int): Number of sample data attributes to generate
    
    Returns:
        str: Path to the generated Parquet file
    """
    # Create base attributes and definitions
    base_attributes = [
        "customer_id", "client_id", "user_identifier", 
        "transaction_amount", "payment_value", "monetary_sum",
        "product_category", "item_type", "product_classification",
        "timestamp", "event_time", "transaction_datetime",
        "email_address", "contact_email", "electronic_mail",
        "phone_number", "contact_phone", "telephone",
        "address", "location", "residence",
        "first_name", "given_name", "forename",
        "last_name", "surname", "family_name",
        "age", "years_old", "customer_age",
        "gender", "sex", "gender_identity",
        "income", "annual_salary", "yearly_earnings"
    ]
    
    base_definitions = [
        "Unique identifier for the customer", "Unique identifier for the client", "Unique identifier for the user",
        "Amount of the transaction in dollars", "Value of the payment in dollars", "Sum of money in dollars",
        "Category of the product", "Type of the item", "Classification of the product",
        "Time when the event occurred", "Time when the event took place", "Date and time of the transaction",
        "Email address of the customer", "Email contact information", "Electronic mail address for communication",
        "Phone number of the customer", "Phone contact information", "Telephone number for communication",
        "Physical address of the customer", "Location of the customer", "Place of residence",
        "First name of the customer", "Given name of the person", "Forename of the individual",
        "Last name of the customer", "Surname of the person", "Family name of the individual",
        "Age of the customer in years", "How old the person is in years", "Age of the customer",
        "Gender of the customer", "Sex of the person", "Gender identity of the individual",
        "Annual income of the customer", "Annual salary of the person", "Yearly earnings of the individual"
    ]
    
    # Create additional random attributes and definitions to reach the desired number
    additional_needed = max(0, num_samples - len(base_attributes))
    
    random_attributes = [f"attribute_{i}" for i in range(additional_needed)]
    random_definitions = [f"Definition for attribute_{i}" for i in range(additional_needed)]
    
    # Combine base and random attributes/definitions
    all_attributes = base_attributes + random_attributes
    all_definitions = base_definitions + random_definitions
    
    # Create DataFrame
    df = pd.DataFrame({
        'attribute_name': all_attributes[:num_samples],
        'attribute_definition': all_definitions[:num_samples]
    })
    
    # Save to Parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Generated sample data with {len(df)} attributes saved to {output_path}")
    return output_path

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "sample_attributes.parquet")
    generate_sample_data(output_file)
