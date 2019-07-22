import re
def normalize_text(s):
    s = s.lower()
    # remove punctuation that is not word-internal (e.g., hyphens, apostrophes)
    s = re.sub('\s\W', ' ', s)
    s = re.sub('\W\s', ' ', s)
    # make sure we didn't introduce any double spaces
    s = re.sub('\s+', ' ', s)
    return s
