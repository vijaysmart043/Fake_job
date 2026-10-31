
from src.preprocessing import load_dataset, prepare_features

df = load_dataset()
df = prepare_features(df)

print("Preprocessing completed successfully!")
print("Dataset shape:", df.shape)

print("\nSample cleaned text:")
print(df["combined_text"].iloc[0][:500])

print("\nMissing values in important features:")
print(df[["combined_text", "fraudulent"]].isnull().sum())

print("\nTarget distribution:")
print(df["fraudulent"].value_counts())
