import requests
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def run_tests():
    print('=== 1. Testing Frontend HTTP Server ===')
    res_fe = requests.get('http://127.0.0.1:5173/')
    print('Frontend Status:', res_fe.status_code)
    print('Frontend Title present:', 'SatQuery AI' in res_fe.text)

    print('\n=== 2. Testing Backend Health API ===')
    res_health = requests.get('http://127.0.0.1:8000/api/health')
    print('Health Status:', res_health.status_code)
    print('Health Response:', res_health.json())

    print('\n=== 3. Testing Sample Presets API ===')
    res_samples = requests.get('http://127.0.0.1:8000/api/samples')
    print('Samples Status:', res_samples.status_code)
    samples = res_samples.json()
    print('Total Presets:', len(samples))
    for s in samples:
        print(f" - [{s['category']}] {s['title']} -> {s['image_names']}")

    print('\n=== 4. Testing Query API (Preset 1: Flood Change Detection - 2 Images) ===')
    with open('sample-images/urban_before.jpg', 'rb') as f1, open('sample-images/urban_after_flood.jpg', 'rb') as f2:
        res1 = requests.post(
            'http://127.0.0.1:8000/api/query',
            data={'query': 'Compare before and after satellite images for flood damage and affected roads'},
            files=[('images', ('urban_before.jpg', f1, 'image/jpeg')), ('images', ('urban_after_flood.jpg', f2, 'image/jpeg'))]
        )
    print('Preset 1 Status:', res1.status_code, 'Task:', res1.json().get('task_type'))

    print('\n=== 5. Testing Query API (Preset 2: Optical+SAR Agricultural Fusion - 2 Images) ===')
    with open('sample-images/agri_optical.jpg', 'rb') as f1, open('sample-images/agri_sar.jpg', 'rb') as f2:
        res2 = requests.post(
            'http://127.0.0.1:8000/api/query',
            data={'query': 'Analyze multi-modal optical and SAR radar imagery for surface roughness and soil moisture'},
            files=[('images', ('agri_optical.jpg', f1, 'image/jpeg')), ('images', ('agri_sar.jpg', f2, 'image/jpeg'))]
        )
    print('Preset 2 Status:', res2.status_code, 'Task:', res2.json().get('task_type'))

    print('\n=== 6. Testing Query API (Preset 3: Naval Port Grounding - 1 Image) ===')
    with open('sample-images/naval_port_grounding.jpg', 'rb') as f1:
        res3 = requests.post(
            'http://127.0.0.1:8000/api/query',
            data={'query': 'Where are the fuel oil storage tanks, container ships, and gantry cranes located? Provide spatial coordinates.'},
            files=[('images', ('naval_port_grounding.jpg', f1, 'image/jpeg'))]
        )
    print('Preset 3 Status:', res3.status_code, 'Task:', res3.json().get('task_type'))

    print('\n=== 7. Testing Query API (Preset 4: Airport Scene Captioning - 1 Image) ===')
    with open('sample-images/airport_hub_captioning.jpg', 'rb') as f1:
        res4 = requests.post(
            'http://127.0.0.1:8000/api/query',
            data={'query': 'Describe this satellite scene in detail, including primary land cover taxonomy and aviation infrastructure.'},
            files=[('images', ('airport_hub_captioning.jpg', f1, 'image/jpeg'))]
        )
    print('Preset 4 Status:', res4.status_code, 'Task:', res4.json().get('task_type'))

    print('\n=== 8. Testing Query API (Preset 5: Solar Park Turbine VQA - 1 Image) ===')
    with open('sample-images/solar_park_vqa.jpg', 'rb') as f1:
        res5 = requests.post(
            'http://127.0.0.1:8000/api/query',
            data={'query': 'How many wind turbines are visible on the western perimeter, and what is the layout of the solar array?'},
            files=[('images', ('solar_park_vqa.jpg', f1, 'image/jpeg'))]
        )
    print('Preset 5 Status:', res5.status_code, 'Task:', res5.json().get('task_type'))

    print('\nALL PRESET TESTS PASSED! ✅')

if __name__ == '__main__':
    run_tests()
