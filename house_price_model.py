import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 57)
print("   HOUSE PRICE PREDICTION - LINEAR REGRESSION MODEL")
print("=" * 57)

try:
    df_raw = pd.read_csv('kc_house_data.csv')
    print(f"\n✅ Dataset loaded: {df_raw.shape[0]:,} rows")
except FileNotFoundError:
    print("\n⚠️  Generating demo data...")
    np.random.seed(42)
    n = 5000
    sqft = np.random.randint(500, 5000, n)
    beds = np.random.randint(1, 7, n)
    baths = np.random.randint(1, 5, n)
    price = (100*sqft + 20000*beds + 15000*baths + 80000
             + np.random.normal(0, 30000, n)).astype(int)
    df_raw = pd.DataFrame({'price': price,
                           'sqft_living': sqft,
                           'bedrooms': beds,
                           'bathrooms': baths})

col_map = {}
for c in df_raw.columns:
    cl = c.lower()
    if 'price' in cl: col_map[c] = 'price'
    elif 'sqft' in cl and 'living' in cl: col_map[c] = 'sqft_living'
    elif 'bed' in cl: col_map[c] = 'bedrooms'
    elif 'bath' in cl: col_map[c] = 'bathrooms'
df_raw = df_raw.rename(columns=col_map)
df = df_raw[['price','sqft_living','bedrooms','bathrooms']].copy()

df.dropna(inplace=True)
df = df[(df['price']>0)&(df['sqft_living']>0)
        &(df['bedrooms']>0)&(df['bathrooms']>0)]
for col in ['price','sqft_living']:
    Q1,Q3 = df[col].quantile(0.25),df[col].quantile(0.75)
    IQR = Q3-Q1
    df = df[(df[col]>=Q1-1.5*IQR)&(df[col]<=Q3+1.5*IQR)]

print(f"Clean dataset: {len(df):,} rows")

X = df[['sqft_living','bedrooms','bathrooms']]
y = df['price']
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n📈 RESULTS")
print(f"  MAE  : ${mae:,.2f}")
print(f"  RMSE : ${rmse:,.2f}")
print(f"  R²   : {r2:.4f} ({r2*100:.1f}%)")

print("\n🏠 SAMPLE PREDICTIONS")
samples=[(800,1,1),(1500,3,2),(2500,4,3),(3500,5,3)]
for s,b,bt in samples:
    p = model.predict(scaler.transform([[s,b,bt]]))[0]
    print(f"  {s} sqft | {b} bed | {bt} bath → ${p:,.0f}")

fig,axes = plt.subplots(1,2,figsize=(12,5))
fig.suptitle("House Price Prediction - Linear Regression",
             fontweight='bold')
axes[0].scatter(y_test,y_pred,alpha=0.4,color='steelblue')
axes[0].plot([y_test.min(),y_test.max()],
             [y_test.min(),y_test.max()],'r--',lw=2)
axes[0].set_xlabel("Actual Price")
axes[0].set_ylabel("Predicted Price")
axes[0].set_title("Actual vs Predicted")
axes[1].scatter(y_pred,y_test-y_pred,alpha=0.4,color='coral')
axes[1].axhline(0,color='black',linestyle='--')
axes[1].set_xlabel("Predicted Price")
axes[1].set_ylabel("Residuals")
axes[1].set_title("Residual Plot")
plt.tight_layout()
plt.savefig('model_results.png',dpi=150)
print("\n✅ Saved model_results.png")