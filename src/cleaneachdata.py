import pandas as pd

df = pd.read_excel("./data/newdata/Table 53_SocialInfrastructure.xlsx", skiprows=4, header=None)

df = df.iloc[:, [3,4,5,6,7,9,10,11]]  



df.columns = [
    "Area",
    "Total_Wards",
    "higher education in 30 mins",
    "access to library in 30 mins",
    "access to doctor in 30 mins",
    "access to firebrigade in 30 mins",
    "access to entertainment in 30 mins",
    "access to shopping mall in 30 mins"
]

# Drop rows where Area is empty or NaN
df = df[df["Area"].notna()]
df["Area"] = df["Area"].astype(str).str.strip()
df = df[df["Area"] != ""]

# Save to CSV
df.to_csv("cleaned_Social_Infrastructure.csv", index=False)
