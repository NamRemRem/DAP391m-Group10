import pandas as pd
import json

df = pd.read_csv("Data/e-shop_clothing_2008.csv", sep=";")

analysis = {
    "columns": df.columns.tolist(),
    "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
    "n_rows": len(df),
    "sample": df.head(3).to_dict(orient="records"),
    "page1_counts": df["page 1 (main category)"].value_counts().to_dict(),
}

with open("data_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

print("Analysis saved to data_analysis.json")
