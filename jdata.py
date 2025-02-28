import json
import random
from faker import Faker

fake = Faker()

def generate_account(is_fake=True):
    return {
        "username": fake.user_name() if is_fake else fake.first_name() + str(random.randint(100, 999)),
        "email": fake.email(),
        "profile_picture": fake.image_url(),
        "followers_count": random.randint(0, 100000) if is_fake else random.randint(500, 1000000),
        "following_count": random.randint(0, 5000) if is_fake else random.randint(50, 5000),
        "posts_count": random.randint(0, 500) if is_fake else random.randint(100, 10000),
        "is_active": fake.boolean(),
        "account_type": "fake" if is_fake else "real"
    }

def generate_dataset(real_count=500, fake_count=500):
    dataset = [generate_account(is_fake=False) for _ in range(real_count)] + \
              [generate_account(is_fake=True) for _ in range(fake_count)]
    random.shuffle(dataset)
    
    with open("account_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)
    
    print(f"Dataset saved with {real_count} real and {fake_count} fake accounts.")

# Generate dataset
generate_dataset(500, 500)
