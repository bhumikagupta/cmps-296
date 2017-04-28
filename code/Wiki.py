from wikiapi import WikiApi
wiki = WikiApi()
wiki = WikiApi({'locale' : 'en'})
results = wiki.find('San Francisco')
article = wiki.get_article(results[0])
#print article.heading
print article.content.encode('utf8')