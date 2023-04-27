from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/check-sleep', methods=['POST'])
def check_sleep():
    image = request.form['image']
    # Hier könntest du eine Funktion aufrufen, die das Bild analysiert
    # und das Ergebnis zurückgibt. Zum Beispiel:
    result = {'sleepy': True}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
