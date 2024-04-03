from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
import speech_recognition as sr
from gtts import gTTS 
import pyttsx3 
import time
import os

OPENAI_API_KEY=""
PINECONE_API_KEY=""
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# Function to convert text to
# speech
class SpeaktoText:
    def SpeakText(self,command):
        
        # Initialize the engine
        engine = pyttsx3.init()
        engine.say(command) 
        engine.runAndWait()
        
    def getTextFromSpeech(self):
        r = sr.Recognizer() 
        while True:
            try:
                with sr.Microphone() as source:
                    print("Listening...")
                    r.adjust_for_ambient_noise(source, duration=0.2)
                    audio = r.listen(source)
                    print("Recognizing...")
                    text = r.recognize_google(audio)
                    text = text.lower()
                    return text
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
                
    def getSpeechFromText(self,textspeech:str):
        language = 'en'
        myobj = gTTS(text=textspeech, lang=language, slow=False) 
        myobj.save("farmerdata.mp3")
        time.sleep(5)
        return os.system("mpg321 farmerdata.mp3")

# load the pdf
file = "data/beta.pdf"
loader = UnstructuredPDFLoader(file)
documents = loader.load()
print(f'You have {len(documents)} document(s) in your data')
print(f'There are {len(documents[0].page_content)} characters in your document')

# The pdf is splitted in small chucks of documents
spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
data = spliter.split_documents(documents)
print(f'Now you have {len(data)} documents')


embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
index_name="langchain3"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)
index.delete(delete_all=True)

docsearch = PineconeVectorStore.from_documents(data, embeddings, index_name=index_name)
time.sleep(5)   
# initialise an object
speaktotext=SpeaktoText()
llm=OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
chain=load_qa_chain(llm,chain_type="stuff")
query = speaktotext.getTextFromSpeech()
docs = docsearch.similarity_search(query)

outputText=chain.invoke({"question": query, "input_documents": docs})['output_text']
print(outputText)
speaktotext.getSpeechFromText(outputText)


