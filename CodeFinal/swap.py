import pandas as pd

# load both
malbehav = pd.read_csv("MalBehavD-V1-processed-clean.csv")
malapi = pd.read_csv("MalAPI2019-processed.csv")

print(f"MalBehavD: {len(malbehav)}")
print(f"Mal-API: {len(malapi)}")

# get malware only from malbehav
malbehav_mal = malbehav[malbehav['labels'] == 1]
malbehav_ben = malbehav[malbehav['labels'] == 0]
print(f"MalBehavD - malware: {len(malbehav_mal)}, benign: {len(malbehav_ben)}")

# pick 100 from each
swap_from_malbehav = malbehav_mal.sample(n=200, random_state=72) #change numbet to test differnt swap sizes
swap_from_malapi = malapi.sample(n=200, random_state=72) #change number to test differnt swap sizes

# remove from originals
malbehav_remaining = malbehav[~malbehav.index.isin(swap_from_malbehav.index)]
malapi_remaining = malapi[~malapi.index.isin(swap_from_malapi.index)]

# prep swap from malapi for training split 80/10/10
swap_from_malapi = swap_from_malapi.copy()
n = len(swap_from_malapi)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

swap_from_malapi = swap_from_malapi.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
swap_from_malapi.loc[:n_train-1, 'split'] = 'train'
swap_from_malapi.loc[n_train:n_train+n_val-1, 'split'] = 'val'
swap_from_malapi.loc[n_train+n_val:, 'split'] = 'test'

print(f"Mal-API samples split: train={n_train}, val={n_val}, test={n - n_train - n_val}")

# add mal-api samples to malbehav
malbehav_mixed = pd.concat([malbehav_remaining, swap_from_malapi], ignore_index=True)

# prep swap_from_malbehav for testing drop sha256 and split cols to match malapi format
swap_for_test = swap_from_malbehav[['labels', 'raw_api_sequence', 'bigram_sequence', 'trigram_sequence']].copy()

# add malbehav samples to malapi test set
malapi_test = pd.concat([malapi_remaining, swap_for_test], ignore_index=True)

print(f"\nAfter swap:")
print(f"MalBehavD mixed: {len(malbehav_mixed)} (training)")
print(f"Mal-API test: {len(malapi_test)} (testing)")

# check balance
print(f"\nMalBehavD balance: {malbehav_mixed['labels'].value_counts().to_dict()}")

# save
malbehav_mixed.to_csv("MalBehavD-V1-mixed.csv", index=False)
malapi_test.to_csv("MalAPI2019-test.csv", index=False)

print("\ndone")