"""
Cloud Cost and Usage Report (CUR) Data Simulation Script

This script generates simulated cloud billing data that mimics AWS/GCP
Cost and Usage Reports for FinOps analysis and portfolio demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_cur_data(num_rows: int = 1000, output_file: str = "simulated_cur_report.csv") -> pd.DataFrame:
    """
    Generate simulated Cloud Cost and Usage Report data.
    
    Args:
        num_rows: Number of billing records to generate (default: 1000)
        output_file: Path to save the CSV file (default: simulated_cur_report.csv)
    
    Returns:
        DataFrame containing the simulated billing data
    """
    np.random.seed(42)  # For reproducibility
    
    # Define cloud services with realistic cost profiles
    # Format: (service_name, base_cost_per_hour, cost_variance, resource_prefix)
    services = [
        ("EC2", 0.50, 2.0, "i-"),           # Compute instances - expensive
        ("RDS", 0.40, 1.5, "db-"),          # Database instances - moderately expensive
        ("Lambda", 0.01, 0.05, "func-"),    # Serverless - cheap per invocation
        ("S3", 0.02, 0.03, "bucket-"),      # Storage - cheap
        ("EBS", 0.10, 0.15, "vol-"),        # Block storage - moderate
        ("CloudFront", 0.05, 0.10, "dist-"), # CDN - moderate
        ("EKS", 0.35, 0.50, "cluster-"),    # Kubernetes - expensive
        ("ElastiCache", 0.25, 0.30, "cache-"), # Caching - moderately expensive
    ]
    
    # Generate resource IDs for each service (simulating multiple resources per service)
    resource_pool = {}
    for service_name, _, _, prefix in services:
        num_resources = np.random.randint(5, 15)
        resource_pool[service_name] = [
            f"{prefix}{np.random.randint(10000, 99999):05x}" 
            for _ in range(num_resources)
        ]
    
    # Generate timestamps spanning the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Create the billing records
    records = []
    
    for _ in range(num_rows):
        # Select a random service
        service_idx = np.random.randint(0, len(services))
        service_name, base_cost, cost_variance, _ = services[service_idx]
        
        # Select a random resource from that service
        resource_id = np.random.choice(resource_pool[service_name])
        
        # Generate random timestamp within the 30-day period
        random_days = np.random.uniform(0, 30)
        timestamp = start_date + timedelta(days=random_days)
        
        # Generate usage hours (0-24 for a day's worth of usage)
        usage_hours = np.random.uniform(0.5, 24.0)
        
        # Calculate cost with realistic variance
        # Some resources are misconfigured (high usage) to simulate zombie resources
        is_zombie = np.random.random() < 0.05  # 5% chance of being a "zombie" resource
        
        if is_zombie:
            # Zombie resources have unusually high costs
            cost_multiplier = np.random.uniform(5.0, 15.0)
        else:
            cost_multiplier = np.random.uniform(0.5, 2.0)
        
        unblended_cost = round(
            base_cost * usage_hours * cost_multiplier + 
            np.random.uniform(0, cost_variance),
            4
        )
        
        records.append({
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Service": service_name,
            "Resource_ID": resource_id,
            "Usage_Hours": round(usage_hours, 2),
            "Unblended_Cost": unblended_cost
        })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Sort by timestamp
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Generated {num_rows} billing records and saved to '{output_file}'")
    print(f"\nData Summary:")
    print(f"  - Date Range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"  - Services: {df['Service'].nunique()}")
    print(f"  - Unique Resources: {df['Resource_ID'].nunique()}")
    print(f"  - Total Cost: ${df['Unblended_Cost'].sum():,.2f}")
    print(f"\nCost by Service:")
    print(df.groupby("Service")["Unblended_Cost"].sum().sort_values(ascending=False).to_string())
    
    return df


if __name__ == "__main__":
    generate_cur_data()
