from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        return 'Arquivo {} enviado com sucesso!'.format(file.filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()