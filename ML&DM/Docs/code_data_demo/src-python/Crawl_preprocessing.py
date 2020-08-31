#!/usr/bin/env python
# coding: utf-8

# # Các thư viện cần sử dụng 
# 
# - **bs4**: parse cấu trúc html 
# - **hashlib**: hash md5 của 1 string, sử dụng như id của văn bản 
# - **re**: xử lý các lệnh liên quan đến  regex 
# - **requests**: gửi các gói tin get / post đến  server .
# - **pyvi**: cho nhiệm vụ tách từ tiếng việt 
# - **dateutil**: cho việc parsing ngày tháng năm 

# In[1]:



import requests
import re
import traceback
import sys
from pyvi import ViTokenizer

import hashlib
from dateutil import parser as dateutil_parser  # install: pip install python-dateutil

from bs4 import BeautifulSoup 
from pprint import pprint

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets.base import load_files

# class **NewsSimple** lưu thông tin của một đối tượng bài báo: 
# - image_url: link ảnh 
# - url: link bài báo 
# - title: title bài viết 
# - code: id bài viết (hash md5 của url)
# - domain: trang domain chứa bài báo (vnexpress, ...)
# - date: thời gian của bài viết được khởi tạo 

# In[2]:




class NewsSimple:
    """
    Thong tin co ban cua 1 trang web
    """

    def __init__(self, url, image_url, title, date):
        self.image_url = image_url
        self.url = url
        self.title = title
        self.code = hashlib.sha1(self.url.encode("utf-8")).hexdigest()
        domain = re.search('(?<=http://)[^/]+', url)
        domain_in_https = re.search('(?<=https://)[^/]+', url)
        if domain is not None:
            self.domain = domain.group(0)
        elif domain_in_https is not None:
            self.domain = domain_in_https.group(0)
        else:
            self.domain = "Unknow"

        date_tmp = ""
        if date is not None and len(date) > 0:
            try:
                date_x = dateutil_parser.parse(date)
                date_tmp = date_x.date().__str__() + ", " + date_x.timetz().__str__()
            except:
                traceback.print_exc(file=sys.stdout)
                date_tmp = ""
        self.date = date_tmp


    def get_date_obj(self):
        return dateutil_parser.parse(self.date)


# __Chuẩn hoá nhãn của bài báo__
# 
# Mỗi trang báo có một nhãn riêng cho từng bài báo, chuẩn hoá lại nhãn của các bài báo. 
# 
# Tính **Đồng nhất** dữ liệu 

# In[3]:


#!/usr/local/bin/python
# coding: utf-8

# g_map_label_domain
# Dinh nghia bien toan cuc anh xa thong tin cac nhan cua cac trang bao
# Chuyen hoa ve dang dictionary
g_map_label_domain = {
    "Khoa học - Công nghệ": [
        "vietbao/Khoa Học",
        "Dân trí/Khoa học - Công nghệ",
        "Dân trí/Sức mạnh số",
        "vietbao/Công Nghệ",
        "techtalk/Cong nghe",
        "vtv/Công nghệ",
        "vnexpress/Khoa học",
        "vnexpress/Số hóa"
    ],
    "Kinh tế": [
        "vietbao/Kinh Tế",
        "vtv/Kinh tế",
        "vnexpress/Kinh doanh",
        "Dân trí/Kinh doanh"
    ],
    "Giáo dục": [
        "Dân trí/Giáo dục - Khuyến học",
        "vtv/Giáo dục",
        "vietbao/Giáo Dục",
        "vnexpress/Giáo dục"
    ],
    "Du lịch": [
        "vietbao/Du Lịch",
        "vietbao/Khám Phá Việt Nam",
        "vnexpress/Du lịch"
    ],
    "Thời sự": [
        "vietbao/Phóng Sự",
        "vietbao/Thế Giới",
        "vnexpress/Thế giới",
        "vtv/Truyền hình",
        "vtv/Trong nước",
        "vtv/Thế giới",
        "vtv/Việt Nam và thế giới",
        "vnexpress/Thời sự",
        "Dân trí/Thế giới"
    ],
    "Việc làm": [
        "Dân trí/Việc làm"
    ],
    "Xe": [
        "vietbao/Xe",
        "Dân trí/Ô tô - Xe máy",
        "vnexpress/Xe"
    ],
    "Pháp luật": [
        "vietbao/An Ninh - Pháp Luật",
        "Dân trí/Pháp luật",
        "vnexpress/Pháp luật"
    ],
    "Độc giả": [
        "Dân trí/Bạn đọc",
        "Dân trí/Diễn đàn",
        "vtv/Góc khán giả"
    ],
    "Thể thao": [
        "Dân trí/Thể thao",
        "vietbao/Bóng Đá",
        "vnexpress/Thể thao"
    ],
    "Sức khỏe": [
        "vietbao/Sức Khỏe",
        "Dân trí/Sức khỏe",
        "vtv/Sức khỏe",
        "vnexpress/Sức khỏe"
    ],
    "Giải trí": [
        "vietbao/Game",
        "vietbao/Media",
        "vietbao/Thế Giới Giải Trí",
        "vietbao/Cười",
        "vnexpress/Giải trí",
        "Dân trí/Giải trí",
        "vnexpress/Cười",
        "vtv/Văn hóa - Giải trí",
        "Dân trí/Chuyện lạ"

    ],
    "Đời sống - Xã hội": [
        "vietbao/Nhà Đất",
        "vietbao/Blog Hay",
        "vietbao/Đời Sống - Gia Đình",
        "vietbao/Xã Hội",
        "vietbao/Văn Hóa",
        "vietbao/Sống Đẹp",
        "vietbao/Nhịp Sống Trẻ",
        "vietbao/Người Việt Bốn Phương",
        "Dân trí/Tình yêu - Giới tính",
        "Dân trí/Xã hội",
        "Dân trí/Nhịp sống trẻ",
        "vnexpress/Tâm sự",
        "vnexpress/Gia đình",
        "vnexpress/Cộng đồng",
        "Dân trí/Tấm lòng nhân ái",
        "Dân trí/Văn hóa",
        "Dân trí/Đời sống",
        "vtv/Đời sống"
    ],
    "Tin khác": [
        "vnexpress",
        "Dân trí",
        "vietbao",
        "techtalk",
        "vtv"
    ]
}

# g_map_domain_label
# Dinh nghia bien toan cuc nhan cua cac trang bao -> nhan chung
# Chuyen hoa ve dang dictionary
g_map_domain_label = {}
for v in g_map_label_domain:
    for key in g_map_label_domain[v]:
        g_map_domain_label[key] = v


# Lop nhan - xac dinh nhan chung cho cac trang bao
class LabelGeneral:
    """Dinh nghia tap nhan chung cho cac trang web"""

    label_general = "Tin khác"

    def __init__(self, label):
        label = label
        if label in g_map_domain_label:
            self.label_general = g_map_domain_label[label]

    def get_label_general(self):
        """ Lay thong tin nhan chung """
        return self.label_general


# class **News** lưu thông tin của một đối tượng bài báo, kế thừa lớp NewsSimple: 
# 
# - Lớp này đóng vai trò ngữ nghĩa cao hơn, gần với dữ liệu mà ta cần. 
# - content: thuộc tính chứa dữ liệu của bài báo 
# - labels: thuọc tính chứa nhãn của bài báo (nhãn gốc của từng trang báo)
# 
# => các phương thức kế thừa từ NewsSimple và LabelGeneral cho phép lớp này lấy thông tin nhãn đã được chuẩn hoá và nội dung của dữ liệu 

# In[4]:



class News(NewsSimple, LabelGeneral):
    """
        Doi tuong thao tac chinh voi 1 trang web 
    """

    def __init__(self, url, image_url, title, content, labels, date=None):
        NewsSimple.__init__(self, url, image_url, title, date)
        self.content = content
        self.labels = labels

        LabelGeneral.__init__(self, self.get_labels(1))

    def get_content(self):
        return self.content

    def get_labels(self, level=-1):
        if level == 1:
            regex = "([^/]+/[^/]+)"
            sublabel = re.search(regex, self.labels)
            if (sublabel != None):
                return sublabel.group(0)

        return self.labels


rss_link = ["http://vnexpress.net/rss"]

def get_content(url):
    """
    Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8
    Accept-Encoding:gzip, deflate, sdch
    Accept-Language:en-US,en;q=0.8,vi;q=0.6
    Connection:keep-alive
    Cookie:__ltmc=225808911; __ltmb=225808911.202893004; __ltma=225808911.202893004.204252493; _gat=1; __RC=4; __R=1; _ga=GA1.3.938565844.1476219934; __IP=20217561; __UF=-1; __uif=__ui%3A-1%7C__uid%3A877575904920217840%7C__create%3A1475759049; __tb=0; _a3rd1467367343=0-9
    Host:dantri.com.vn
    Referer:http://dantri.com.vn/su-kien.htm
    Upgrade-Insecure-Requests:1
    User-Agent:Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36
    """
    domain = None
    domains = url.split('/')
    if (domains.__len__() >= 3): domain = domains[2]

    headers = dict()
    headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    headers['Accept-Encoding'] = 'gzip, deflate, sdch'
    headers['Accept-Language'] = 'en-US,en;q=0.8,vi;q=0.6'
    headers['Connection'] = 'keep-alive'
    headers['Host'] = domain
    headers['Referer'] = url
    headers['Upgrade-Insecure-Requests'] = '1'
    headers[
        'User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.82 Safari/537.36'

    r = requests.get(url, headers=headers, timeout=10)
    r.encoding = 'utf-8'
    r.close()
    return str(r.text)


# Lay tat ca cac rss title web server cung cap 
def get_links_rss(rss_parent):

    # Lay noi dung neu chua co
    raw_content = get_content(rss_parent)
    soup = BeautifulSoup(raw_content.encode('utf-8', 'inorge'), 'html.parser')

    links_rss = {}
    
    # Lay khoi div rss
    domain = "vnexpress.net"
    div_rss_tags = soup.find_all('section', class_=['page_rss'])

    if (div_rss_tags is None or div_rss_tags.__len__() == 0):
        return links_rss
    else:
        divrss_tag = div_rss_tags[0]

    # Duyet qua cac the a -> link
    for a_tag in divrss_tag.find_all('a', href=True):
        if (a_tag["href"].find('http://') == -1):
            links_rss["http://" + domain + a_tag["href"]] = True
        else:
            links_rss[a_tag["href"]] = True
    return links_rss 


# In[8]:


def get_content_news(url):
    content_paper = {}
    content_paper["labels"] = labels = None
    content_paper["content_text"] = content_text = None

    try:
        raw_content = get_content(url)
        soup = BeautifulSoup(raw_content, 'html.parser')

        # lay thong tin nhan bai bao
        labels = "vnexpress"
        divs_headermain = soup.find_all("section", class_="cat_header")
        if (divs_headermain != None and divs_headermain.__len__() > 0):
            div_headermain = divs_headermain[0]
            for labels_element in div_headermain.find_all('ul', 'breadcrumb'):
                for label in labels_element.find_all('a', href=True):
                    labels += ("/" + label.text)

                if len(labels) > len("vnexpress"):
                    break

        # lay noi dung bai bao
        content_text = ""
        for div_bodymain in soup.find_all("article", class_="content_detail"):
            for p_element in div_bodymain.find_all('p', 'Normal'):
                content_text += p_element.text

        content_paper["labels"] = labels
        content_paper["content_text"] = content_text

    except:
        print(url)
        traceback.print_exc(file=sys.stdout)

    return content_paper

rss_parent = rss_link[0]
rss_link_raw = get_links_rss(rss_parent)
pprint(rss_link_raw)

content = get_content(list(rss_link_raw.keys())[0])
print(content)


# In[10]:


# check struct of content
soup = BeautifulSoup(content, 'html.parser')
print(soup)


# In[11]:


list_webs = []
items = soup.find_all('item')
print(items[0])


# In[12]:


# Lay tin bai dac biet
for item in items:
    url1 = ""
    try:
        # Lay thong tin
        url1 = item.link.text
        print(url1)
        
        if url1 == "":
            url1 = item.guid.text
            print(url1)
            
#         if url1.find("http:") == -1:
#             description_elements = BeautifulSoup(item.description.text, 'html.parser')
#             a_tag = description_elements.find_all('a')[0]
#             url1 = a_tag['href']
        title = item.title.text
        date = item.pubdate.text
        content_item = BeautifulSoup(item.description.text, 'html.parser')
        if content_item.img is not None:
            image_url = content_item.img['src']
        else:
            image_url = ""


        # tai va boc tach noi dung bai viet
        content_paper = get_content_news(url1)
        print(content_paper)
        

    except:
        traceback.print_exc(file=sys.stdout)
        continue

    # Tao doi tuong
    if url1 is not None and title is not None and content_paper['content_text'] is not None             and content_paper['labels'] is not None:
        element = News(url1, image_url, title, content_paper['content_text'],
                       content_paper['labels'], date)
        list_webs.append(element)


# ## Tiền xử lý dữ liệu 
# 
# - Chuẩn hoá nhãn của văn bản / bài báo 
#     - Văn bản / bài báo thu thập từ nhiều nguồn trang báo khác nhau, cần chuẩn hoá lại các nhãn tương đồng. 
#         - ví dụ "vietbao/Khoa Học" và "Dân trí/Khoa học - Công nghệ" => Khoa học - Công nghệ 
# - Tách từ. Biểu diễn văn bản thành tập hợp các từ. 
#     - Tách từ giúp cho ý nghĩa văn bản được biểu diễn rõ ràng hơn. 
# - Chuyển hoá văn bản và nhãn thành dạng vector. 
#     - Vd: thu thập 10 nhãn ['Giải trí', 'Khoa học - Công nghệ', 'Kinh tế', 'Pháp luật', 'Sức khỏe', 'Thể thao', 'Thời sự', 'Tin khác', 'Đời sống - Xã hội', 'Độc giả'] và chuyển về dạng id onehot tương ứng từ 0 - 9. 
#     - Chuyển hoá văn bản về dạng [vector tfidf](https://vi.wikipedia.org/wiki/Tf%E2%80%93idf) 
#     - Loại bỏ từ dừng. Các từ không mang ý nghĩa phân loại (anh, anh ấy, ba, ba ba, ...)
#     

# In[13]:


print(ViTokenizer.tokenize(list_webs[0].get_content()))
print(list_webs[0].get_labels())
print(list_webs[0].get_label_general())


# #### đưa dữ liệu text về dạng ma trận 

# In[18]:


# load tập dữ liệu 
data_train = load_files(container_path="../data_sample/news_1135/", encoding="utf-8")

# load các từ dừng từ file định nghĩa trước cho tiếng việt 
with open("../vietnamese-stopwords.txt") as f:
    stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords] 
print(stopwords[:10])

# build module pipeline cho các bước tiền xử lý 
# - CountVectorizer() => sinh vector tần số cho từng văn bản 
# - TfidfTransformer() => chuyển vector tần số về dạng vector tfidf 
model_rf_preprocess = Pipeline([('vect', CountVectorizer(stop_words=stopwords)),
                    ('tfidf', TfidfTransformer()),
                    ])
data_preprocessed = model_rf_preprocess.fit_transform(data_train.data, data_train.target)


# print dữ liệu kiểm thử 
print("data = \n", data_preprocessed[0])
print("target = \n", data_train.target[0])


# In[20]:


# tiền xử lý với dữ liệu mới 
sentence = ViTokenizer.tokenize("""
USD điều chỉnh trái chiều, vàng SJC quay đầu tăng
Mỗi lượng vàng SJC sáng nay tăng 20.000 đồng sau khi vừa có phiên đi xuống; tỷ giá trung tâm được Ngân hàng Nhà nước tăng 10 đồng nhưng các nhà băng lại giảm.
Mở cửa đầu ngày 23/11, Tập đoàn DOJI niêm yết giá vàng miếng SJC cao hơn 20.000 đồng một lượng so với hôm qua với giá mua bán 36,44 - 36,54 triệu đồng. Tương tự, các doanh nghiệp vàng khác trong nước cũng tăng vài chục nghìn đồng một lượng và dao động quanh mốc 36,5 triệu đồng. 

Giao dịch vàng tại doanh nghiệp PNJ. Ảnh: Lệ Chi.
Giao dịch vàng tại doanh nghiệp PNJ. Ảnh: Lệ Chi.

Sự đi lên nhẹ của giá vàng trong nước cũng chung xu hướng với thế giới. Theo đó, chốt phiên Mỹ tối qua, mỗi ounce tăng gần 2 USD, lên 1.227 USD một ounce. Sang phiên châu Á sáng nay, kim loại quý vẫn dao động quanh mốc này. 

Quy đổi sang tiền Việt, giá thế giới hiện tương đương 34,6 triệu đồng một lượng (chưa thuế, phí, gia công). Chênh lệch với thị trường trong nước vẫn quanh 1,9 triệu đồng.

Tập đoàn DOJI cho biết, trên thị trường ghi nhận lượng khách bán vàng ra chiếm chủ đạo, với tỷ lệ khoảng 65% trên tổng lượng giao dịch tại doanh nghiệp.

Trên thị trường ngoại hối, tỷ giá trung tâm do Ngân hàng Nhà nước công bố sáng nay là 22.743 đồng một USD, tăng 10 đồng so với hôm qua. Trong khi đó, giá USD tại các ngân hàng thương mại lại đi xuống. Cụ thể, giá bán đôla Mỹ tại Vietcombank hiện quanh 23.295 - 23.385 đồng, giảm 10 đồng so với hôm qua.


""")
sentence_vector = model_rf_preprocess.transform([sentence])
print(sentence)
print(sentence_vector[0] )


# In[ ]:




