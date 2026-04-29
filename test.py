import requests

API_KEY = "1c59ec266bb1bac83b588a4184d56b2d"

# Probar búsqueda
search_url = "https://api.themoviedb.org/3/search/movie"
params = {
    'api_key': API_KEY,
    'query': 'Inception',
    'year': 2010
}

response = requests.get(search_url, params=params)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    if data['results']:
        poster_path = data['results'][0].get('poster_path')
        print(f"Poster path: {poster_path}")
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
            print(f"URL: {poster_url}")
            
            # Probar que la imagen se puede descargar
            img_response = requests.get(poster_url)
            print(f"Imagen status: {img_response.status_code}")
            if img_response.status_code == 200:
                print("✅ La imagen es accesible")
            else:
                print("❌ La imagen NO es accesible")
        else:
            print("No tiene póster")
    else:
        print("No se encontró la película")
else:
    print(f"Error: {response.text}")
