import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
ps=PorterStemmer()
def transform_test(text):
  text=text.lower()
  # becomes a list after this
  text=nltk.word_tokenize(text)
  text2=[]
  for i in text:
    # only letters and words and no's are filtered and appended
    if i.isalnum():
      text2.append(i)
  text=text2[:]
  text2.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      text2.append(i)
  text=text2[:]
  text2.clear()
  for i in text:
    text2.append(ps.stem(i))
# to convert list to txt sentence again
  return " ".join(text2)
tfidf=pickle.load(open("vectorize.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))
st.title("Email & SMS Spam detector")
inp_msg=st.text_area("Enter the message")

if st.button("Predict"):
    transformed_sms=transform_test(inp_msg)
    vectr_inp=tfidf.transform([transformed_sms])
    result=model.predict(vectr_inp)
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

# here we hv to preprocces then vectorize,predict,display
# se we took pickle files from our ipynb notebook and added here
# to perform all these
