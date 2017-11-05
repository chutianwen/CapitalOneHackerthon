from collections import Counter
with open("./Dataset/merchant_list.txt") as f:
    text = f.read()

merchant_names = text.split(",")
text = text.lower()
text = text.replace('.', ' <PERIOD> ')
text = text.replace(',', ' <COMMA> ')
text = text.replace('"', ' <QUOTATION_MARK> ')
text = text.replace(';', ' <SEMICOLON> ')
text = text.replace('!', ' <EXCLAMATION_MARK> ')
text = text.replace('?', ' <QUESTION_MARK> ')
text = text.replace('(', ' <LEFT_PAREN> ')
text = text.replace(')', ' <RIGHT_PAREN> ')
text = text.replace('--', ' <HYPHENS> ')
text = text.replace('?', ' <QUESTION_MARK> ')
# text = text.replace('\n', ' <NEW_LINE> ')
text = text.replace(':', ' <COLON> ')
text = text.replace('&', ' <AND> ')
text = text.replace('-', ' <DASH> ')

words = text.split()
word_cnt = Counter(words)
print(len(word_cnt))
print(word_cnt)

# trim out unrelated words
unrelated_words = {'<AND>', '<DASH>', 'of', 'the', 'and', 'pa'}
word_cnt_trimmed = {word: word_cnt[word] for word in word_cnt
                    if word not in unrelated_words and 3 <= word_cnt[word] < 35}

print("Size of trimmed word_cnt:{}".format(len(word_cnt_trimmed)))
print(word_cnt_trimmed)

top_words = sorted(word_cnt_trimmed, key=word_cnt_trimmed.get, reverse=True)
print(top_words)

merchant_names_category = []
for merchant_name in merchant_names:
    merchant_name_ori = merchant_name
    merchant_name = merchant_name.replace("\"", "")
    merchant_name = merchant_name.replace(".", " ")
    merchant_name_words = merchant_name.lower().split()
    category = "other"
    for word in top_words:
        merchant_name_words = merchant_name.split()
        if word in merchant_name_words:
            category = word
            break
    merchant_names_category.append([merchant_name_ori, category])

merchant_names_category.sort(key=lambda x: x[1])
categories = set(map(lambda x:x[1], merchant_names_category))
print("Categories:", categories)

with open("./Dataset/MerchantName_Category.txt", 'w') as f2:
    f2.writelines("{}\t{}\n".format("Merchant Name", "Category"))
    for item in merchant_names_category:
        f2.writelines("{}\t{}\n".format(item[0], item[1]))

condense_category = {'inn': 'travel', }