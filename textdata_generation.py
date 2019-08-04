import os
import codecs
from nltk.tokenize import sent_tokenize

# glob text files except modified, original and  textdataset.txt files
path = "./textdataset"
all_text_list = os.listdir(path)
text_list = [s for s in all_text_list if "modified" not in s]
text_list = [s for s in text_list if "org" not in s]
text_list.remove("textdataset.txt")

all_sent_tokenize_list = []

with open("./textdata/textdata.txt", mode='w') as f:
    f.write("")

for filename in text_list:
    textdata_path = "./textdata/" + filename
    with codecs.open(textdata_path, "r", encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f
                 if len(line) != 0]
        text = " ".join(lines)
    sent_tokenize_list = sent_tokenize(text)

    # extract textfile -> "Title\tsentence\n"
    modified_textdata_path = textdata_path.rstrip(".txt") + "_modified.txt"
    with codecs.open(modified_textdata_path, "w", encoding="utf-8") as g:
        for text in sent_tokenize_list:
            g.write(filename.rstrip(".txt") + "\t" + text + "\n")

    # write text to textdataset -> "Title\tsentence\n"
    with codecs.open("./textdata/textdataset.txt", mode='a', encoding="utf-8") as f:
        for text in sent_tokenize_list:
            f.write(filename.rstrip(".txt") + "\t" + text + "\n")