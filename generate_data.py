import pandas as pd
import numpy as np

np.random.seed(42)

n = 800

data = pd.DataFrame({
    "date": pd.date_range(start="2023-01-01", periods=n, freq="D"),
    "client": np.random.choice(["Bank", "Hospital", "E-commerce", "Gov"], n),
    "region": np.random.choice(["US", "Europe", "Asia"], n),
    "product": np.random.choice(["Firewall", "Antivirus", "SIEM", "Endpoint Security"], n),
    "sales_amount": np.random.randint(2000, 50000, n),
    "licenses_sold": np.random.randint(1, 50, n),
    "employee": np.random.choice(["Alice", "Bob", "Charlie", "David", "Emma"], n),
    "threat_type": np.random.choice(["Phishing", "Malware", "DDoS", "Ransomware"], n),
    "attack_count": np.random.randint(10, 300, n),
})

# anomalies
data.loc[np.random.choice(n, 20), "sales_amount"] *= 3
data.loc[np.random.choice(n, 20), "attack_count"] *= 4

data.to_csv("cyber_data.csv", index=False)

print("Dataset created successfully")
