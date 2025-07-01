import csv

COMPANY_DATA = []
with open("nasdaq_companies.csv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        COMPANY_DATA.append({
            "name": row["Name"],
            "symbol": row["Symbol"]
        })

print(f"Loaded {len(COMPANY_DATA)} companies from CSV.")