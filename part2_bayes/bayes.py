import csv
import re

def load_data(path):
    data = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)                          # skip the header row
        for row in reader:
            data.append((row[0], row[1]))     # (review_text, sentiment)
    return data


def tokenize(text):
    text = text.lower()                       # 1. lowercase first
    text = text.replace("<br />", " ")        # 2. remove <br /> tags (space, not empty)
    words = re.findall(r"[a-z']+", text)      # 3. grab runs of letters/apostrophes
    words = [w.strip("'") for w in words]     # 4. remove quote marks from the ends
    return set(w for w in words if w)


def prepare(data):
    # turn each (text, sentiment) into (set_of_words, sentiment), done once
    return [(tokenize(text), sentiment) for text, sentiment in data]


def compute_posterior(prepared, keyword):
    total = len(prepared)
    negative = 0          # reviews labelled "negative"
    with_keyword = 0      # reviews containing the keyword
    both = 0              # negative AND containing the keyword

    for words, sentiment in prepared:
        if sentiment == "negative":
            negative += 1
        if keyword in words:
            with_keyword += 1
        if sentiment == "negative" and keyword in words:
            both += 1

    prior = negative / total
    likelihood = both / negative if negative > 0 else 0
    marginal = with_keyword / total if total > 0 else 0
    posterior = likelihood * prior / marginal if marginal > 0 else None

    return prior, likelihood, marginal, posterior

prepared = prepare(load_data("IMDB Dataset.csv"))
positive_keywords = ["great", "excellent", "wonderful", "perfect"]
negative_keywords = ["worst", "awful", "terrible", "waste"]

for kw in positive_keywords + negative_keywords:
    prior, likelihood, marginal, posterior = compute_posterior(prepared, kw)
    print(kw, round(prior, 4), round(likelihood, 4), round(marginal, 4), round(posterior, 4))