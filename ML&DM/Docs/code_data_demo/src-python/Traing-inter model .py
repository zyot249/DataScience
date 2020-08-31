#!/usr/bin/env python
# coding: utf-8

# # Các thư viện cần sử dụng
# - matplotlib: vẽ ảnh lên giao diện 
# - numpy: hỗ trợ các phép toán với ma trận 
# - sklearn: thư viện hỗ trợ các mô hình học máy - random forest, svm, các hàm hỗ trợ quá trình training cross-validate, ... 
# - pyvi: thư viện hỗ trợ tách từ 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

from sklearn.datasets.base import load_files
from pyvi import ViTokenizer


from sklearn import svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.datasets.base import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load dữ liệu từ thư mục đã crappy từ trước 
# 
# Cấu trúc thư mục như sau 
# 
# - data_sample/news_1135/
# 
#     - Kinh tế: 
#         - bài báo 1.txt 
#         - bài báo 2.txt 
#     - Pháp luật
#         - bài báo 3.txt 
#         - bài báo 4.txt 

# In[2]:


data_train = load_files(container_path="../data_sample/news_1135/", encoding="utf-8")

print(data_train.filenames)
print()

print("Tong so file: {}" .format( len(data_train.filenames)))



# ## Mô tả nội dung file 
# 
# - Nội dung file là text nội dung các bài báo đã qua module tách từ - sử dụng Vitokenizer thư viện pivy 
# - Nhãn của văn bản là tên của folder mà bài báo đang nằm trong (Nhãn được chuyển hoá thành id dạng số)
# - Danh sách 10 nhãn của tập dữ liệu ở dưới đây 
#     - ['Giải trí', 'Khoa học - Công nghệ', 'Kinh tế', 'Pháp luật', 'Sức khỏe', 'Thể thao', 'Thời sự', 'Tin khác', 'Đời sống - Xã hội', 'Độc giả']

# In[3]:


print(data_train.target_names)
print()

print(str(data_train.data[0]))
print(data_train.target[0])
print()

print(str(data_train.data[1]))
print(data_train.target[1])


# ## Thử nghiệm training dữ liệu với mô hình Random forest
# 
# - Dựng module tiền xử lý dữ liệu text, chuyển hoá text về dạng vector 
#     - Sử dụng TfIdf vector để làm biểu diễn cho 1 văn bản 
#     - Sử dụng training mô hình Doc2Vect - biểu diễn văn bản dưới dạng vector 
# - Training dữ liệu với hàm fit. 
#     - Dữ liệu input là vector văn bản / bài báo  
#     - Dữ liệu nhãn là id của vector văn bản / bài báo 
# - Save mô hình sau khi đã train. Và tái sử dụng cho việc infer sau này 

# In[4]:


# load dữ liệu các stopwords 
with open("../vietnamese-stopwords.txt") as f:
    stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords] 
print(stopwords[:10])

# 
# Transforming data 
# Chuyển hoá dữ liệu text về dạng vector tfidf 
#     - loại bỏ từ dừng
#     - sinh từ điển
module_count_vector = CountVectorizer(stop_words=stopwords)
model_rf_preprocess = Pipeline([('vect', module_count_vector),
                    ('tfidf', TfidfTransformer()),
                    ])
data_preprocessed = model_rf_preprocess.fit_transform(data_train.data, data_train.target)
print(module_count_vector.vocabulary_)


# Build model 
# Thử nghiệm với mô hình random forest classifier, với 50 cây quyết định (n_estimators), sử dụng hàm lỗi gini.  
model_rf =  RandomForestClassifier(criterion='gini', n_estimators=50)


# Training data 
# Gọi hàm fit với tham số 
#   - X: dữ liệu văn bản dạng vector 
#   - y: dữ liệu nhãn của văn bản (identify)
model_rf.fit(X=data_preprocessed, y=data_train.target)



# ## Thử nghiệm infer dữ liệu thực tế với mô hình Random forest sau khi đã training 
# 
# - Chuyển hoá dữ liệu text qua module tiền xử lý 
#     - Do hệ thống cần truyền vào một chuỗi văn bản, nên với 1 văn bản cần infer thì ta truyền mảng có 1 phần tử 
# - Dữ liệu thu được từ module tiền xử lý là vector đại diện của văn bản 
# - Sử dụng mô hình đã được training để predict 

# In[27]:


sentence = ViTokenizer.tokenize("""
HLV tuyển Việt Nam hài lòng với kết quả thắng 3-0 và ngôi đầu sau trận tiếp Campuchia tối 24/11.
HLV Felix Dalmas: 'Campuchia cố gắng, nhưng Việt Nam quá mạnh' / Việt Nam vào bán kết AFF Cup với ngôi đầu
"Hôm nay, chúng tôi thắng 3-0, các cầu thủ của tôi thi đấu tốt và tôi hài lòng. Nhưng nếu chỉ nhìn vào kết quả, sẽ rất dễ hiểu nhầm, bởi thực tế đây không hề là trận đấu dễ dàng", ông Park Hang-seo mở lời trong cuộc họp báo sau trận đấu.

Trên sân Hàng Đẫy tối 24/11, tuyển Việt Nam được đánh giá mạnh hơn hẳn so với Campuchia - đội đã bị loại sau ba lượt trận đầu. Tuy nhiên, trước lối đá phòng ngự dày đặc của đối phương, dàn cầu thủ dưới trướng HLV Park Hang-seo cũng phải chờ tới phút 39 mới có thể mở tỷ số, nhờ công tiền đạo Tiến Linh. Sau bàn khai thông, tuyển Việt Nam đá thoải mái hơn, nhưng cũng chỉ có thể ghi thêm hai bàn trong thời gian còn lại, với các pha lập công của Quang Hải ở phút 41, rồi Văn Đức ở phút 61.

Việt Nam 3-0 Campuchia    
Nhờ thắng Campuchia 3-0, cộng thêm việc Myanmar thua 0-3 trên sân Malaysia ở trận đấu cùng giờ, tuyển Việt Nam vượt lên chiếm ngôi nhất bảng A. Vị trị này được cho là sẽ mang lại lợi thế nhất định cho thầy trò HLV Park Hang-seo ở vòng knock-out, vì đội đứng thứ nhất ở vòng bảng sẽ đá trận bán kết lượt về trên sân nhà. Tuỳ kết quả các trận cuối bảng B ngày mai, tuyển Việt Nam sẽ biết Thái Lan, Philippines hay Singapore là đối thủ ở bán kết. Tuy nhiên, HLV Park Hang-seo không quá bận tâm về việc Việt Nam sẽ tranh vé vào chung kết với ai.

"Điều quan trọng nhất là tuyển Việt Nam cán đích ở vị trí dẫn đầu bảng. Ngay lúc này, chúng ta vẫn chưa biết sẽ đấu bán kết với đối thủ nào. Ngày mai, hai trợ lý của tôi sẽ chia nhau đi xem hai trận đấu ở Bangkok và Jakarta để tìm hiểu kỹ hơn về các đội có liên quan. Tuy nhiên, chúng tôi có nhiều video về cả ba đội có khả năng đi tiếp ở bảng B. Tuyển Việt Nam cũng có tới bảy ngày chuẩn bị cho trận bán kết lượt đi. Tôi tin mọi chuyện sẽ ổn", nhà cầm quân người Hàn Quốc cho biết.

HLV Park Hang-seo rất thoải mái khi nhắc đến vòng bán kết. 
HLV Park Hang-seo rất thoải mái khi nhắc đến vòng bán kết. 

Đang thoải mái và vui vẻ khi nói về trận đấu và vòng bán kết, HLV Park Hang-seo chợt chùng giọng xuống khi được hỏi về học trò Nguyễn Văn Toàn. Cầu thủ thuộc biên chế HAGL dính chấn thương trong buổi tập hôm qua 23/11, và không có tên trong danh sách đăng ký đấu Campuchia hôm nay. Khi xuất hiện trên sân Hàng Đẫy tối 24/11, Văn Toàn cũng phải nhờ các đồng đội đỡ, dìu hoặc cõng đi.  

"Văn Toàn có phần suy sụp vì chấn thương hôm qua. Theo tính toán ban đầu của tôi, lẽ ra cậu ấy sẽ đá chính trước Campuchia hôm nay. Qua đánh giá ban đầu, tôi nghĩ chấn thương của Văn Toàn không quá nghiêm trọng. Nhưng chúng ta vẫn phải chờ kết luận bác sỹ. Đến giờ, tôi vẫn chưa biết liệu cậu ấy có thể trở lại hay không. Tôi chỉ hy vọng Văn Toàn sẽ bình phục càng sớm càng tốt", HLV trưởng tuyển Việt Nam nói. 


""")
sentence_vector = model_rf_preprocess.transform([sentence])
print(sentence)
print(sentence_vector[0] )

out_label = model_rf.predict(sentence_vector)
print(data_train.target_names)
print(out_label[0])

print(data_train.target_names[out_label[0]])

print(data_preprocessed[0][:30])
print(data_train.target_names[out_label[0]])



# In[ ]:




