#numpy وهى مكتبة يتم استعمالها لانها تحتوى على mathimatics functions
import numpy as np
#pandas ي أداة تحليل بيانات مفتوحة المصدر ومعالجتها سريعة وقوية ومرنة وسهلة الاستخدام
import pandas as pd
#تقسيم المصفوفات أو المصفوفات إلى مجموعه  عشوائي من البيانات واختبار مجموعات فرعية
from sklearn.model_selection import train_test_split
#تحويل مجموعة من المستندات الأولية إلى مصفوفة من ميزات TF-IDF.
#TF-IDF  تردد المستند المعكوس للتردد ، ووزن tf-idf هو وزن يستخدم غالبًا في استرجاع المعلومات واستخراج النص
#بمهني تاني بيقيس الكلمه اتكررت في النص كام مره و اهميتها 
from sklearn.feature_extraction.text import TfidfVectorizer
#هي خوارزميه لا تحتاج الي large-scale learning للتعليم 
from sklearn.linear_model import PassiveAggressiveClassifier 
#خوارزميه تحسب دقه المجموغه الفرعيه للموديول
#بمعني بتحسب دقه test 
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv(r"D:\PYTHON\GITHUB code\done\Detecting Fake News\news.csv")

#هنا بنعرض اول خمس صفوف في الداتا بيز 
data.head()

#هنا بيعرفنا كام عمود و كام صف 
data.shape

#خزنا في المتغير الي اسمه lables عمود label
lables=data.label

#و اظهرنا شكل الداتا الي متخزنه في عمود lables
lables.head()

data.head()

#هنا قسمنا الداتا عشان نبدا نتدرب الداتا
#و محطتش random_state عشان انا بحب ادرب الداتا و اعرف ايه الي ببيعلي الاكيروسي  
x_train,x_test,y_train,y_test=train_test_split(data['text'],lables,test_size=0.2,random_state=9)

#بنستخرج feather بتاعه كل كلمه اذا بتاعه الكلمه ترددها اكبر من 0.7 بنتجاهلها
#و هنحط القيم في مصفوفه و النتيجه من كل صف 
#https://www.youtube.com/watch?v=ib34L_r-5zs
#https://www.youtube.com/watch?v=4vT4fzjkGCQ
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#هندرب الموديول علي داتا بتاعه التدريب و داته Test
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#بنستخدمها لما يكون في تدفق كبير من البيانات
#بنستخدمها مع ملفات txt
pac=PassiveAggressiveClassifier(max_iter=50)
#بندرب classifier بتاعنا
pac.fit(tfidf_train,y_train)

#بنحسب الاكيورسي بتاعه الموديل بتاعنا 
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#بنطبع مصفوفه  confusion عشان نشوف الصح من الاخطاء 
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])