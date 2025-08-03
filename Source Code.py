# Step 1: Import required libraries
import pandas as pd
import joblib
from tkinter import *
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import nltk
import os


# Step 2: Exception Handling for NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Step 3 - Create the Main Window
root = Tk()
root.geometry('800x720')
root.configure(bg='light blue')
root.title('Chatbot')
root.resizable(0, 0)


# Step 4: Read Training Data
file_name = 'data.csv'
data = pd.read_csv(file_name, encoding='latin1')
data = pd.DataFrame(data)

# Step 5: Defining Functions
def get_parameters(df):
    par = {}
    for col in df.columns[df.isnull().any()]:
        if df[col].dtype in ['float64', 'int64', 'int32']:
            strategy = 'mean'
        else:
            strategy = 'most frequent'
        missing_values = df[col][df[col].isnull()].values[0]
        par[col] = {'missing_values': missing_values, 'strategy': strategy}
    return par

def impute_missing_values(df, par):
    for col in par.keys():
        missing_values = par[col]['missing_values']
        strategy = par[col]['strategy']
        imp = SimpleImputer(missing_values=missing_values, strategy=strategy)
        df[col] = imp.fit_transform(df[[col]])
    return df

def ask_bot():
    user_message = user_input.get()
    chatbot.insert(END, 'User: '+ user_message)
    chatbot.insert(END, "")
    try:
        if user_message.lower() in ["hi", "hello", "hey", "good morning", "good evening"]:
            bot_response = "Hello! How can I assist you today?"
        else:
            bot_response = process_message(user_message)
        chatbot.insert(END, 'Bot: '+ bot_response)
        chatbot.insert(END, "")
    except Exception:
        chatbot.insert(END, "Bot: Error in processing your message.")
    user_input.delete(0, END)

def enter(event):
    send_button.invoke()
root.bind('<Return>', enter)


# Step 6: Data Preprocessing
parameters = get_parameters(data)
data = impute_missing_values(data, parameters)
label_encoder = preprocessing.LabelEncoder()
data['flags'] = label_encoder.fit_transform(data['flags'])
data['category'] = label_encoder.fit_transform(data['category'])
data['intent'] = label_encoder.fit_transform(data['intent'])


# Step 7: Split Data into Features (X) and Target (y)
X = data.drop(columns=['intent'])
y = data['intent']


# Step 8: Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2022, stratify=y)


# Step 9: Create and Train the Model
clt_tfid = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('LogisticRegress', LogisticRegression(C=1.0, penalty='l2', max_iter=100))
])
clt_tfid.fit(X_train['utterance'], y_train)


# Step 10: Save the Model using Joblib (Optional)
joblib.dump(clt_tfid, 'chatbot_model.pkl')


# Step 11: Define Function to Process User Message and Generate Response
def process_message(message):
    try:
        response_index = clt_tfid.predict([message])
        response = label_encoder.inverse_transform(response_index)
        return response[0]
    except Exception as e:
        return f"Error processing the message: {str(e)}"

# Step 12: Creating a Chatbot GUI

# Step 12.1 - Loading the images
if os.path.exists('bot avatar.png'):
    bot_avatar_image = PhotoImage(file='C:\\Users\\meabh\\OneDrive\\Documents\\Program\\Python\\Intenship\\Project\\Chatbot\\bot avatar.png')
else:
    raise FileNotFoundError("Bot avatar image 'bot avatar.png' not found.")
if os.path.exists('send.png'):
    send_image = PhotoImage(file='send.png')
else:
    raise FileNotFoundError("Send image 'send.png' not found.")


# Step 12.2 - Create Labels, Input Fields, and Buttons

# Step 12.2.1 - Text Labels
label1 = Label(root, text='Chat with', bg='light blue', fg='navy', font='Helvetica: 20 bold italic')
label2 = Label(root, text='Chatbot', bg='light blue', fg='navy', font='Helvetica: 30 bold italic')
label3 = Label(root, text='We typically reply in a few minutes', bg='light blue', fg='navy', font='Helvetica: 12 italic')

# Step 12.2.2 - Input Box and Chat Frame
user_input = Entry(root, font='Helvetica: 15 bold italic', bd=2, bg='white', fg='black')
frame = Frame(root)
sc = Scrollbar(frame)
chatbot = Listbox(frame, width=57, height=15, bg="white", bd=2, font="Helvetica: 16 bold italic")
chatbot.config(yscrollcommand=sc.set)
sc.config(command=chatbot.yview)
chatbot.insert(END,' '*30+'Hello How can I help you')

# Step 12.2.3 - Buttons
send_button = Button(root, image=send_image, activebackground='light blue', command=ask_bot, bd=0, bg='light blue', fg='navy', height=48, width=60)

# Step 12.2.4 - Avatar Images
bot_avatar = Label(root, image=bot_avatar_image, bd=2, bg='light blue')

# Step 12.3 - Position Elements
bot_avatar.place(x=50, y=30)
label1.place(x=180, y=50)
label2.place(x=180, y=80)
label3.place(x=50, y=170)
frame.place(x=50, y=200)
sc.pack(side=RIGHT, fill=Y)
chatbot.pack(side=LEFT, fill=BOTH, pady=10)
user_input.place(x=50, y=620, width=650, height=40)
send_button.place(x=700, y=615)

# Step 12.4 - Mainloop
root.mainloop()