from pattern.web import Google, SEARCH, download, plaintext
import textacy

query = "things to do during deep space travel"
engine = Google(license="AIzaSyB4f-UO51_qDWXIwSwR92aejZso6hHJEY4", throttle=1.0, language="en")

for result in engine.search(query, start=1, count=10, type=SEARCH, cached=False):
	text = plaintext(download(result.url))
	print textacy.TextDoc(text).key_terms(algorithm='textrank', n=5)