import pandas as pd
import numpy as np

# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

usernames = [f'user_{i}' for i in range(n_samples)]
labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])

data = []
for label in labels:
    if label == 0:  # Real account
        followers = max(0, int(np.random.normal(500, 100)))
        posts = max(0, int(np.random.normal(100, 20)))
        age = max(1, int(np.random.normal(24, 6)))
        profile_pic = np.random.choice(['Yes', 'No'], p=[0.9, 0.1])
    else:  # Fake account
        followers = max(0, int(np.random.normal(10, 5)))
        posts = max(0, int(np.random.normal(5, 3)))
        age = max(1, int(np.random.normal(3, 2)))
        profile_pic = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
    
    data.append([usernames[len(data)], followers, posts, age, profile_pic, label])

df = pd.DataFrame(data, columns=[
    'Username',
    'Number of Followers',
    'Number of Posts',
    'Account Age (months)',
    'Profile Picture Present (Yes/No)',
    'Label'
])

df.to_csv('fake_accounts.csv', index=False)