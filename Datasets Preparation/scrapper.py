import numpy as np
import pandas as pd
import re
import sys
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.request import urlopen
import nltk
import requests
import urllib.parse
from html.parser import HTMLParser
import csv

#https://www.snopes.com/fact-check/category/business/page/NUMBER/
#From 1-38
#Choose h5 with class title
linkss = []
df = pd.DataFrame(columns=['Link','Claim','Status','Date'])
seq = 0
for i in range (10,38):
    Page1 = urlopen(url='https://www.snopes.com/fact-check/category/business/page/' + str(i), data=None)
    Page1 = BeautifulSoup(Page1, features="html5lib")
    articles = Page1.find_all(lambda tag: tag and tag.name.startswith("article"))
    #print(articles)
    for a in articles:
        if a == None:
            continue
        link = a.find_all(lambda tag: tag and tag.name == "a" ,href=True)
        try:
            link = link[0]['href']
            print('LINK:',link)
            # Enter the link
            PageInside = urlopen(url=link, data=None)
            PageInside = BeautifulSoup(PageInside, features="html5lib")
            # Find p with "Claim:"
            claim = PageInside.find('div',{'class': 'content'}).find_all('p')[0]
            print('Claim:', claim)
            '''
            #=================================================2==============================

            claim = ''
            claim = PageInside.find_all(lambda tag: tag and tag.name == "span" and tag.text == "Claim")[0].nextSibling
            #spanIn = claim.find("span")
            #tt = spanIn.extract()
            #claim = str(claim.next).replace(':   ','')
            print('Claim:',claim)
            if claim.strip() == '':
                continue
            statusTag = PageInside.find('noindex').find("span").find("span").next
            print(statusTag)
            publishDate = PageInside.find_all(lambda tag: tag and tag.name == "span" and tag.text == "Originally published:")[0].nextSibling
            print('DATE: ',publishDate)
            '''
            '''
            #=================================================3==============================
            claimTag = PageInside.find("div", {"class": "claim"}).find('p')
            claim = claimTag.next
            print('Claim:',claim)
            statusTag = PageInside.find("div", {"class": "rating-wrapper card"}).find("div", {"class": "media-list"}).find("div", {"class":"media rating"}).find("div",{"class":"media-body"}).find("h5").next
            # Find p with "Status:"
            # Find p with "Last updated:"
            print('Status:', statusTag)
            publishDate = PageInside.find("div",{"class":"footer-wrapper"}).find("footer").find("ul").find("li",{"class":"date date-published"}).find_all("span")[1].next
            print("Date: ", publishDate)
            '''
            #df.loc[seq] = [link, claim, str(statusTag), str(publishDate)]
            seq += 1
            #df.to_csv('snopes_dataset3.csv')
        except Exception as e:
            print(e)
print(linkss)

'''
def ArchivePage(url1):
    Page1 = urlopen(url=url1, data=None)
    Page1 = BeautifulSoup(Page1, features="html5lib")
    # print(elements)
    headers = []
    headersAll = []
    paragraphs = []
    hdrs = Page1.find_all(lambda tag: tag and tag.name.startswith("h1"))
    with open('OmarAlTalebSheet.csv', 'a', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['cleanString', 'tokens', 'numbers', 'IsDeathYear', 'content', 'sentences', 'events'])
        for header in hdrs:
            print(header.next_element.name)
            foundNext = False
            content = ''
            cleanString = re.sub('[\n0-9-]', '', strip_tags(str(header)))
            stringwithnumbers = re.sub('[\n]', '', strip_tags(str(header)))
            if cleanString.strip() != '':
                content = getNextAdjacent(20, header, 'p')
                content = content.replace(u'\xa0',u'')
                headersAll.append(header)
                headers.append(cleanString)
                IsDeathYear = False
                if 'ت' in cleanString.split(' '):
                    IsDeathYear = True
                tokens = [s for s in cleanString.split(' ') if s != '' and s != 'ت']
                sentences = re.split(r"\.|\:|\n|\,", content)
                events = [re.split(r'([0-9]+)',s) for s in sentences if s != '']
                numbers = re.findall(r'\d+', stringwithnumbers)
                print('NUMBERS',numbers)
                print('HEADER', cleanString)
                print('TOKENIZED', tokens)
                print('DEAD', IsDeathYear)
                print('CONTENT', content)
                print('SENTENCES', sentences)
                print('EVENTS', events)
                row = [cleanString, tokens, numbers, IsDeathYear, content, sentences, events]
                writer.writerow(row)
    csvFile.close()
links  = getAllLinks()
for link in links:
    ArchivePage(link)'''