# Features
X = df.drop('SalePrice', axis=1)

# Target
y = df['SalePrice']

X = pd.get_dummies(X, drop_first=True)

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

