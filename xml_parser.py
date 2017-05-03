import numpy as np
import xml.etree.ElementTree as ET
import json
import urllib2

class ParseXML(object):

    def __init__(self, filename, columns):
        self.filename = filename
        self.body = ""
        self.title = ""
        self.count100 = dict()

        self.columns = columns
        self.body_words = [x.replace("text:", "") for x in columns if "text:" in x]
        self.title_words = [x.replace("title:", "") for x in columns if "title:" in x]

        self.text_length = 0
        self.title_length = 0

        
    def parsexml(self):
        tree = ET.parse(self.filename)
        root = tree.getroot()

        body = ""
        title = ""
        
        for paragraph in root.iter("body"):
            body += " " + paragraph.text.rstrip()
        self.body = body.replace(",", "")

        for paragraph in root.iter("title"):
            title += " " + paragraph.text.rstrip()
        self.title = title.replace(",", "")

        for word in self.title_words:
            self.count100["title:" + word] = self.title.count(word)

        self.count100["textNUM_TOKENS"] = len(self.body.split())
        self.count100["titleNUM_TOKENS"] = len(self.title.split())
            
        for word in self.body_words:
            self.count100["text:" + word] = self.body.count(word)


    def getjson(self):
        response = urllib2.urlopen("http://boilerpipe-web.appspot.com/extract?url=" + self.filename + "&output=json")
        webContent = response.read()
        jfile = json.loads(webContent)

        return (jfile["response"]["content"], jfile["response"]["title"])
            
    def parse(self):
        (body, title) = self.getjson()        
        
        self.body = body.encode('utf-8').replace(",", "")
        self.title = title.encode('utf-8').replace(",", "")

        for word in self.title_words:
            self.count100["title:" + word] = self.title.count(word)

        self.count100["textNUM_TOKENS"] = len(self.body.split())
        self.count100["titleNUM_TOKENS"] = len(self.title.split())
            
        for word in self.body_words:
            self.count100["text:" + word] = self.body.count(word)

            
            
    def make_vector(self):
        def get_count(x):
            if x in self.count100:
                return self.count100[x]
            else:
                return 0
        return np.array([get_count(x) for x in self.columns])
            

    
p = ParseXML("http://www.theonion.com/article/trump-promises-government-will-continue-fund-all-e-55839", ["text:stuff", "text:Trump", "title:Trump"])

p.getjson()
