from pymongo import MongoClient

c = MongoClient('mongodb://localhost:27017/')
db = c['smc']

users = list(db.users.find({}, {'password': 0}))
history = list(db.history.find({}, {'_id': 0}))

print("=" * 50)
print(f"DATABASE: smc | Total Users: {len(users)}")
print("=" * 50)
for i, u in enumerate(users, 1):
    print(f"\nUser {i}:")
    print(f"  Full Name : {u.get('full_name')}")
    print(f"  Username  : {u.get('username')}")
    print(f"  Email     : {u.get('email')}")
    print(f"  Phone     : {u.get('phone')}")
    print(f"  ID        : {u.get('_id')}")

print("\n" + "=" * 50)
print(f"Total Predictions Stored: {len(history)}")
print("=" * 50)
for h in history[:5]:
    print(f"  [{h.get('prediction')}] {h.get('username')} - {str(h.get('message',''))[:50]}")
