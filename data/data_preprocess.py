import pandas as pd
from sklearn.model_selection import train_test_split

dpath = "./sentences_with_sentiment.xlsx"
df = pd.read_excel(dpath, sheet_name=0)
df_melt = pd.melt(df, id_vars=['ID'],var_name='target', value_vars=['Positive','Negative','Neutral'])
df_melt = df_melt[df_melt['value']==1]
df_melt.groupby('target')['value'].sum()
data = df_melt.sort_values(by='ID').reset_index(drop=True).merge(df[['ID','Sentence']], on='ID')
label_map = {
    "Positive":0,
    "Negative":1,
    "Neutral":2}
data['label'] = data['target'].map(label_map)

X = data['Sentence']
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=23, test_size=0.2,stratify=y)

# with open("./data/train.txt", mode='w', encoding='utf-8') as f:
train_file = []
for sent, label in zip(X_train.values, y_train):
    if label == 0:
        lb = 'Positive'
    elif label == 1:
        lb = 'Negative'
    elif label == 2:
        lb = 'Neutral'
    else:
        raise ValueError
    text = sent + '\t' + lb
    train_file.append(text)

test_file = []
for sent, label in zip(X_test.values, y_test):
    if label == 0:
        lb = 'Positive'
    elif label == 1:
        lb = 'Negative'
    elif label == 2:
        lb = 'Neutral'
    else:
        raise ValueError
    text = sent + '\t' + lb
    test_file.append(text)

def write_txt(file,lines):
    with open(file, 'w',encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")
    return 'Saved'


write_txt("./data/train.txt", train_file)
write_txt("./data/dev.txt", test_file)