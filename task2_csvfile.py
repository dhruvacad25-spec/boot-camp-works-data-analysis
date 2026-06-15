import pandas as pd

url = "https://raw.githubusercontent.com/namithar99/task-2-dataset/refs/heads/main/movies_info%20(1).csv"
df = pd.read_csv(url)

# 1. Count Total Movies
print("1. Total Movies:")
print(len(df))

# 2. List Unique Genres
print("\n2. Unique Genres:")
genres = set()

for g in df.iloc[:, 2].dropna():
    for genre in str(g).split(','):
        genres.add(genre.strip())

for genre in sorted(genres):
    print(genre)

# 3. Count Movies Per Genre
print("\n3. Movies Per Genre:")
genre_count = {}

for g in df.iloc[:, 2].dropna():
    for genre in str(g).split(','):
        genre = genre.strip()
        genre_count[genre] = genre_count.get(genre, 0) + 1

for genre, count in sorted(genre_count.items()):
    print(f"{genre}: {count}")

# 4. Find the Longest Movie Description
print("\n4. Longest Movie Description:")
df['description_length'] = df.iloc[:, 1].astype(str).apply(len)

longest = df.loc[df['description_length'].idxmax()]

print("Movie Title:")
print(longest.iloc[0])

print("\nDescription:")
print(longest.iloc[1])

print("\nDescription Length:")
print(longest['description_length'])

# 5. Search Movies by Keyword
print("\n5. Search Movies by Keyword:")
keyword = input("Enter keyword: ")

result = df[df.iloc[:, 1].astype(str).str.contains(keyword, case=False, na=False)]

print("\nMovies containing the keyword:")
for movie in result.iloc[:, 0]:
    print(movie)
