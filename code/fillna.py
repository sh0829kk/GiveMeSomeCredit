train.loc[train.MonthlyIncome.isnull(), 'SeriousDlqin2yrs'].sum()

#Fill none value with mean value
train['MonthlyIncome'] = train['MonthlyIncome'].fillna(train['MonthlyIncome'].mean())