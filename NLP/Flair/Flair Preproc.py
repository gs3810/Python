import pandas as pd

inputDF = pd.ExcelFile("News_Headlines.xlsx")
tabnames = inputDF.sheet_names
data = inputDF.parse(tabnames[0])

data = data[['Relevance', 'Headline']].rename(columns={"Relevance":"label", "Headline":"text"})
 
data = '__label__' + data['label'].astype(str) +' ' + data['text']

data.iloc[0:int(len(data)*0.8)].to_csv('train.csv', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.csv', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('dev.csv', index = False, header = False);