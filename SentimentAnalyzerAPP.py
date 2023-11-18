from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
import os
import tensorflow as tf
from RUNNER.RunBiLSTM import RNNBILSTM
from RUNNER.RunGRU import RNNGRU
from RUNNER.RunLSTM import RNNLSTM

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
UPLOAD_FOLDER = 'uploadTxt'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
risultato = "lorem ipsum"

def runSelectRNN(typeRnn, fileName):
    if typeRnn == "lstm":
        app.logger.info("Init rete LSTM")
        rnn = RNNLSTM(fileName)
        app.logger.info("Calocolo Sentimenti")
        percentualePos, percentualeNeg = rnn.analyze_sentiments_Percentage()
        ris = f"Percentuale positivi '{percentualePos}%'<br>Percentuale negativi '{percentualeNeg}%'"
        print(ris)
        return ris
    elif typeRnn == "bilstm":
        app.logger.info("Init rete BiLSTM")
        rnn = RNNBILSTM(fileName)
        app.logger.info("Calocolo Sentimenti")
        percentualePos, percentualeNeg = rnn.analyze_sentiments_Percentage()
        ris = f"Percentuale positivi '{percentualePos}%'<br>Percentuale negativi '{percentualeNeg}%'"
        print(ris)
        return ris
    else:
        app.logger.info("Init rete GRU")
        rnn = RNNGRU(fileName)
        app.logger.info("Calocolo Sentimenti")
        percentualePos, percentualeNeg = rnn.analyze_sentiments_Percentage()
        ris = f"Percentuale positivi '{percentualePos}%'<br>Percentuale negativi '{percentualeNeg}%'"
        print(ris)
        return ris


@app.route('/')
def index():
    return render_template('index.html', title='Home', username='Genny', risultato=risultato)


@app.route('/ricevi-dati', methods=['POST'])
def ricevi_dati():
    global risultato

    try:
        scelta = request.form.get('scelta')
        file = request.files.get('file')

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            if os.path.exists(file_path):
                os.remove(file_path)
                app.logger.info(f"File '{file.filename}' già esistente. Vecchio file cancellato.")

            file.save(file_path)
            app.logger.info(f"Tipo rete '{scelta}' e file '{file.filename}' inviati con successo")

        risultato = runSelectRNN(scelta, file.filename)
        print(f"Risultato: {risultato}")

        return risultato
    except Exception as e:
        return f'Errore durante la gestione della richiesta: {str(e)}'

if __name__ == '__main__':
    app.run(port=8080)
