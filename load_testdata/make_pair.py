import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "test.csv")

def get_questions():
    fp = open(csv_path,'r',encoding='utf-8')
    title = fp.readline()

    questions = []

    while True:
        data = fp.readline()
        if not data :break
        data = data.strip().split(",")
        id = data[0].strip()
        file_path = data[1].strip()

        question = ""
        for i in range(3,len(data)):
            question += data[i]

        question = question.lstrip("\"").rstrip("\"").strip()
        questions.append([id,file_path,question])

    fp.close()
    return questions 
