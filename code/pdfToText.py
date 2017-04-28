import slate

data_file_path = '../data/'

pdf = data_file_path + 'sf.pdf'

with open(pdf) as f:
    doc = slate.PDF(f)

text_file = open(data_file_path + 'sf.txt', 'w')

for page in doc:
    text_file.write(page)
