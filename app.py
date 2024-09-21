import streamlit as st
import pickle
import re
import spacy
import numpy as np
import sklearn
from sklearn.preprocessing import minmax_scale

# loading models
clf = pickle.load(open('clf1_rf.pkl','rb'))

def cleantxt(txt):
  clntxt = re.sub('http\S+\s',' ',txt)
  clntxt = re.sub('RT|CC',' ',clntxt)
  clntxt = re.sub('#\S+\s',' ',clntxt)
  clntxt = re.sub(' @\S+',' ',clntxt)
  clntxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""),' ',clntxt)
  clntxt = re.sub(r'[^\x00-|x7f]',' ',clntxt)
  clntxt = re.sub('\s+',' ',clntxt)

  return clntxt

#preprocess
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
  doc = nlp(text)
  txt = [i.lemma_ for i in doc if not i.is_punct and not i.is_stop]
  txt = [i.lower() for i in txt]

  return ' '.join(txt)

nlp1 = spacy.load('en_core_web_lg')

# web app

def main():
    st.title('Resume Field Detector App')
    uploaded_file = st.file_uploader('Upload Your Resume', type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            #if utf-8 decoding fails, try decoding with latin-1
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleantxt(resume_text)
        cleaned_resume = preprocess(cleaned_resume)
        cleaned_resume = nlp1(cleaned_resume).vector
        cleaned_resume = np.stack(cleaned_resume)
        cleaned_resume = minmax_scale(cleaned_resume)
        cleaned_resume_scaled = cleaned_resume.reshape(1, -1)
        prediction_id = clf.predict(cleaned_resume_scaled)[0]

        st.write(f'Predicted Id: {prediction_id}')

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)


# python main
if __name__ == '__main__':
    main()
