import PyPDF2

text_file = open('../data/sf.txt', 'w')

pdfFileObj = open('../data/sf.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader.numPages
pageObj = pdfReader.getPage(1)
text_file.write(pageObj.extractText())
print pageObj.extractText()
