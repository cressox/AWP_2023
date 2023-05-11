import time, requests

def measure_api_response_time():
    def get_time_by_api_call(uri):
        # Lese Bild von einer Datei
        with open('test-image.jpg', 'rb') as f:
            image_data = f.read()
        start_time = time.time()
        # Sende POST-Anfrage an die Flask-API
        response = requests.post(uri, files={'image': image_data})
        # Überprüfe die Antwort der Flask-API
        if response.status_code != 200:
            print(f"API call failed with status code {response.status_code}"+'\n'+uri)
            return None
        if not response.content:
            print("API response is empty")
            return None
        try:
            # Empfange die Antwort der Flask-API
            response_data = response.json()
        except ValueError as e:
            print(f"Failed to decode API response: {e}")
            return None
        end_time = time.time()
        # Berechne die Zeit, die benötigt wurde, um die Antwort der Flask-API zu erhalten
        response_time = end_time - start_time
        return response_time
    
    rt1 = get_time_by_api_call('http://localhost:5000/check-face-slow')

    # Schreibe die Zeit in eine externe txt-Datei
    with open('response_time.txt', 'w') as f:
        f.write(str(rt1))

if __name__=='__main__':
    measure_api_response_time()
