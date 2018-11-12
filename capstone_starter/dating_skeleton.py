import pandas as pd
import numpy as np
import re
from collections import Counter
from matplotlib import pyplot as plt
from sklearn import preprocessing

#Create your df here:

df = pd.read_csv("profiles.csv")

#print(df.job.head())

plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

all_data = df

#print(all_data.drinks.unique())
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
all_data['drinks_code'] = all_data.drinks.map(drink_mapping)
#print(all_data[['drinks','drinks_code']].head())

#print(all_data.drugs.unique())
smokes_mapping = {"no": 0, "trying to quit": 1, "when drinking": 2, "sometimes": 3, "yes": 4}
all_data['smokes_code'] = all_data.smokes.map(smokes_mapping)
#print(all_data[['smokes','smokes_code']].head())

#print(all_data.drugs.unique())
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
all_data['drugs_code'] = all_data.drugs.map(drugs_mapping)
#print(all_data[['drugs','drugs_code']].head())


essay_cols = ['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9']

# Removing the NaNs
all_essays = all_data[essay_cols].replace(np.nan, '', regex=True)

# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)

all_data['essay_len'] = all_essays.apply(lambda x: len(x))

#print(all_data['essay_len'].head())

all_essays = all_essays.apply(lambda words: re.sub(r"&amp", " and ", words))
all_essays = all_essays.apply(lambda words: re.sub(r"<.*?>", "", words))
all_essays = all_essays.apply(lambda words: re.sub(r"'", "", words))
all_essays = all_essays.apply(lambda words: re.findall(r"\w+", words.lower()))


all_data['avg_word_length'] = all_essays.apply(lambda words: sum(len(word) for word in words) / (len(words) + 0.0000001))

print(all_data['avg_word_length'])

#all_data['word_i_or_me'] = all_essays.apply(lambda words: Counter(words))
#print(all_data['word_i_or_me'][0])

all_data['word_i_or_me'] = all_essays.apply(lambda words: Counter(words)['i'] + Counter(words)['me'])

print(all_data['word_i_or_me'][0])



feature_data = all_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]


print(feature_data.isna().any())

feature_data.fillna({'smokes_code':5, 'drinks_code':6, 'drugs_code':3}, inplace=True)

print(feature_data.isna().any())

print(feature_data)

x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

print(feature_data)