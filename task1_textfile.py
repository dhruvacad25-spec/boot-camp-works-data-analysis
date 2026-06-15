from urllib.request import urlopen

url = "https://www.gutenberg.org/files/11/11-0.txt"

with urlopen(url) as file:
    text = file.read().decode("utf-8")

# 1. Total number of words
words = text.split()
print("Total number of words:", len(words))

# 2. Count occurrence of "Alice"
alice_count = words.count("Alice")
print("Occurrence of 'Alice':", alice_count)

# 3. Lines mentioning "Queen"
lines = text.splitlines()
queen_count = 0

for line in lines:
    if "Queen" in line:
        queen_count += 1

print("Lines mentioning 'Queen':", queen_count)

# 4. First 500 characters
print("First 500 characters:")
print(text[:500])

# 5. Unique words and their count
unique_words = {}

for word in words:
    if word not in unique_words:
        unique_words[word] = 1
    else:
        unique_words[word] += 1

print("Total unique words:", len(unique_words))

# 6. Longest word
longest_word = ""

for word in words:
    if len(word) > len(longest_word):
        longest_word = word

print("Longest word:", longest_word)
