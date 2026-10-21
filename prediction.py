import os
if not os.path.exists('dld_transactions.csv'):
    from google.colab import files
    files.upload()
import pandas as pd, numpy as np, matplotlib.pyplot as plt, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = pd.read_csv('dld_transactions.csv', low_memory=False)
print('Rows and columns:', data.shape)
print('\nData types:'); print(data.dtypes.value_counts()); print(data.dtypes.to_string(), '\n')
print('Duplicate rows:', data.duplicated().sum())
print('Date range:', data.instance_date.str[-4:].min(), 'to', data.instance_date.str[-4:].max())


print(data.trans_group_en.value_counts(), '\n')
print(data.property_usage_en.value_counts(), '\n')
print(data.property_type_en.value_counts(), '\n')
print(data.property_sub_type_en.value_counts(), '\n')
print(data.reg_type_en.value_counts(), '\n')
print('Missing values:'); print(data.isna().sum()[data.isna().sum() > 0])


print(data.groupby('trans_group_en').actual_worth.describe().round(0))
print('\nProcedure names by group:'); print(pd.crosstab(data.trans_group_en, data.procedure_name_en))

print('Start:', len(data))
data = data.drop_duplicates();
print('1 Remove duplicates:', len(data))
data = data[data.trans_group_en == 'Sales'];
print('2 Sales only:', len(data))
data = data[data.property_usage_en == 'Residential'];
print('3 Residential only:', len(data))
data = data[data.property_sub_type_en.isin(['Flat', 'Villa', 'Hotel Apartment'])];  print('4 Flat/Villa/Hotel Apt:', len(data))
data = data[data.actual_worth > 0];
print('5 Price above 0:', len(data))

print(data.procedure_area.describe().round(2))
print('\nZero or negative areas:', (data.procedure_area <= 0).sum())
print('Areas between 0 and 15 m2:', ((data.procedure_area > 0) & (data.procedure_area < 15)).sum())
print('Smallest area of 15 m2 or more:', round(data.procedure_area[data.procedure_area >= 15].min(), 1), 'm2')
print('\nHow many areas fall in each range:')
print(pd.cut(data.procedure_area, [-1, 0, 5, 15, 30, 40, 50, 100]).value_counts().sort_index())

data = data[data.procedure_area >= 15];
print('6 Area at least 15 m2:', len(data))

data['area_name_en'] = data.area_name_en.str.strip().str.title()    
data['size_sqft'] = data.procedure_area * 10.7639                    
data['log_size'] = np.log(data.size_sqft)
data['rooms'] = data.rooms_en.fillna('Unknown')                      
data['offplan'] = (data.reg_type_en == 'Off-Plan Properties') * 1   
data['sale_date'] = pd.to_datetime(data.instance_date, format='%d-%m-%Y')
data['year'] = data.sale_date.dt.year
data['month'] = data.sale_date.dt.month
data['quarter'] = data.sale_date.dt.quarter
data['price_per_sqft'] = data.actual_worth / data.size_sqft        

print('Communities:', data.area_name_en.nunique())
print('Size range (sq ft):', round(data.size_sqft.min()), 'to', round(data.size_sqft.max()))

# Leakage check: meter_sale_price is just price / area
print('Biggest gap between price/area and meter_sale_price:', (data.actual_worth / data.procedure_area - data.meter_sale_price).abs().max())

# Extreme prices - errors or real luxury?
show = ['area_name_en', 'property_sub_type_en', 'rooms', 'size_sqft', 'actual_worth', 'price_per_sqft']
print('\nMost expensive sales'); print(data.nlargest(5, 'actual_worth')[show].round(0))
print('\nHighest price per sq ft'); print(data.nlargest(5, 'price_per_sqft')[show].round(0))
print('\nLowest price per sq ft'); print(data.nsmallest(5, 'price_per_sqft')[show].round(0))

# Rank communities by price and by price per sq ft
rank = data.groupby('area_name_en').agg(sales=('actual_worth', 'size'), median_price=('actual_worth', 'median'),
average_price=('actual_worth', 'mean'), median_ppsf=('price_per_sqft', 'median'),
median_size=('size_sqft', 'median'))
rank['rank_by_price'] = rank.median_price.rank(ascending=False)
rank['rank_by_ppsf'] = rank.median_ppsf.rank(ascending=False)
print(rank.sort_values('rank_by_price').round(0))

# Save the cleaned data
final = data[['sale_date', 'area_name_en', 'property_sub_type_en', 'rooms', 'size_sqft', 'log_size',
'has_parking', 'offplan', 'year', 'actual_worth', 'price_per_sqft']]
final.to_csv('cleaned_modelling_dataset.csv', index=False)
print(final.shape)

plt.hist(final.actual_worth / 1e6, bins=range(0, 21))
plt.title('Chart 1: Price distribution'); plt.xlabel('Price (AED millions)'); plt.ylabel('Number of sales'); plt.show()
print('Mean:', round(final.actual_worth.mean()), ' Median:', round(final.actual_worth.median()))

plt.hist(np.log(final.actual_worth), bins=50)
plt.title('Chart 2: Log of price'); plt.xlabel('ln(price)'); plt.ylabel('Number of sales'); plt.show()

by_area = final.groupby('area_name_en')
(by_area.actual_worth.median() / 1e6).sort_values().plot(kind='barh', figsize=(7, 6))
plt.title('Chart 3: Median price by community'); plt.xlabel('Median price (AED millions)'); plt.ylabel('Community'); plt.show()

by_area.price_per_sqft.median().sort_values().plot(kind='barh', figsize=(7, 6), color='orange')
plt.title('Chart 4: Median price per sq ft by community'); plt.xlabel('AED per sq ft'); plt.ylabel('Community'); plt.show()

final.groupby('year').price_per_sqft.median().plot(marker='o')
plt.title('Chart 5: Price per sq ft by year'); plt.xlabel('Year'); plt.ylabel('AED per sq ft'); plt.show()

sample = final.sample(10000, random_state=42)
plt.scatter(sample.size_sqft, sample.actual_worth, s=3, alpha=0.3)
plt.xscale('log'); plt.yscale('log')
plt.title('Chart 6: Size vs price'); plt.xlabel('Size (sq ft)'); plt.ylabel('Price (AED)'); plt.show()

rooms_order = ['Studio', '1 B/R', '2 B/R', '3 B/R', '4 B/R', '5 B/R', '6 B/R', 'PENTHOUSE']
(final.groupby('rooms').actual_worth.median().loc[rooms_order] / 1e6).plot(kind='bar')
plt.title('Chart 7: Median price by room type'); plt.xlabel('Room type'); plt.ylabel('Median price (AED millions)'); plt.show()

final.groupby(['year', 'offplan']).price_per_sqft.median().unstack().plot(marker='o')
plt.title('Chart 8: Off-plan (1) vs ready (0) price per sq ft'); plt.xlabel('Year'); plt.ylabel('AED per sq ft'); plt.show()

numbers = ['size_sqft', 'log_size', 'has_parking', 'offplan', 'year']
words = ['area_name_en', 'property_sub_type_en', 'rooms']

X = final[numbers + words]
y = final.actual_worth

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('Train rows:', len(X_train), ' Test rows:', len(X_test))

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
for extra in [None, 'month', 'quarter']:
    A, B, cols = X_tr.copy(), X_val.copy(), list(words)
    if extra:
        A[extra] = data.loc[A.index, extra].astype(str)
        B[extra] = data.loc[B.index, extra].astype(str)
        cols = words + [extra]
    enc = ColumnTransformer([('w', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cols)], remainder='passthrough')
    m = Pipeline([('e', enc), ('lr', LinearRegression())]).fit(A, np.log(y_tr))
    p = np.exp(m.predict(B))
    print('Extra column:', extra, '| validation MAPE:', round((abs(y_val - p) / y_val).mean() * 100, 2), '%')

    def new_model():
      encode = ColumnTransformer([('words', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), words)], remainder='passthrough')
      return Pipeline([('encode', encode), ('lr', LinearRegression())])

model_A = new_model().fit(X_train, y_train)              
model_B = new_model().fit(X_train, np.log(y_train))      

pred_A = model_A.predict(X_test)
pred_B = np.exp(model_B.predict(X_test))                  

print('Model A negative predictions:', (pred_A < 0).sum())
print('Model B negative predictions:', (pred_B < 0).sum())

def scores(pred):
    ape = abs(y_test - pred) / y_test
    return {'R2': r2_score(y_test, pred), 'MAE (AED)': mean_absolute_error(y_test, pred),
            'RMSE (AED)': mean_squared_error(y_test, pred) ** 0.5, 'MAPE (%)': ape.mean() * 100,
            'Within 20% (%)': (ape <= 0.2).mean() * 100}

table = pd.DataFrame({'Model A (normal price)': scores(pred_A), 'Model B (log price)': scores(pred_B)}).T
print(table.round(3).to_string())

result = X_test.copy()
result['actual'] = y_test
result['predicted'] = pred_B
result['error_%'] = abs(result.actual - result.predicted) / result.actual * 100
result['bias_%'] = (result.predicted - result.actual) / result.actual * 100
result['price_band'] = pd.cut(result.actual, [0, 0.75e6, 1.5e6, 3e6, 6e6, 1e9], labels=['<0.75M', '0.75-1.5M', '1.5-3M', '3-6M', '>6M'])

print('Error by price band'); print(result.groupby('price_band')[['error_%', 'bias_%']].mean().round(1), '\n')
print('Error by property type'); print(result.groupby('property_sub_type_en')[['error_%', 'bias_%']].mean().round(1), '\n')
print('Error by community'); print(result.groupby('area_name_en')['error_%'].mean().sort_values().round(1))

plt.scatter(result.actual / 1e6, result.predicted / 1e6, s=3, alpha=0.3)
plt.plot([0, 20], [0, 20], 'r--')
plt.xlim(0, 20); plt.ylim(0, 20)
plt.title('Chart 9: Actual vs predicted (Model B)'); plt.xlabel('Actual price (AED millions)'); plt.ylabel('Predicted price (AED millions)'); plt.show()


low_err, high_err = result['bias_%'].quantile([0.15, 0.85])
LOW = 1 / (1 + high_err / 100)
HIGH = 1 / (1 + low_err / 100)
print('Valuation range = estimate x', round(LOW, 2), 'to estimate x', round(HIGH, 2))

names = model_B.named_steps['encode'].get_feature_names_out()
coef = pd.Series(model_B.named_steps['lr'].coef_, index=names)
effect = (np.exp(coef) - 1) * 100

print('Community effect (% vs Al Barsha First, same size/type/rooms/year):')
print(effect.filter(like='area_name_en').sort_values().round(0).to_string())
print('\nOther effects (%):')
print(effect.filter(like='rooms_').round(1).to_string())
print('Villa vs flat:', round(effect['words__property_sub_type_en_Villa'], 1), '%')
print('Off-plan vs ready:', round(effect['remainder__offplan'], 1), '%')
print('Each extra year:', round(effect['remainder__year'], 1), '%')
print('Size: coefficient on log size =', round(coef['remainder__log_size'], 2), '(close to 1 = price grows in proportion to size)')

info = {'features': numbers + words,
        'communities': sorted(final.area_name_en.unique()),
        'sub_types': sorted(final.property_sub_type_en.unique()),
        'rooms': sorted(final.rooms.unique()),
        'year_min': int(final.year.min()), 'year_max': int(final.year.max()),
        'range_low_factor': float(LOW), 'range_high_factor': float(HIGH),
        'test_metrics': {'MAPE (%)': float((abs(y_test - pred_B) / y_test).mean() * 100)},
        'size_range_by_subtype': final.groupby('property_sub_type_en').size_sqft.quantile([0.005, 0.995]).unstack().to_dict('index')}

joblib.dump(model_B, 'model_B_log_linear_regression.joblib')
joblib.dump(model_A, 'model_A_raw_linear_regression.joblib')
joblib.dump(info, 'model_metadata.joblib')
print('Saved')