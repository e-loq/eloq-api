import json

def to_json():
    ceiling_height = 0
    max_sat_height = 0
    layers = []
    data = {
        "unit": "m",
        "z_ceil":  ceiling_height,
        "z_sat": max_sat_height,
        "z_marker": 1,  #  NOTE this is given by TRUMPF
        "layers": layers,
        "optimize": True,
        "marker_grid": 1,
        "sat_grid": 10
    }

    return data

if __name__ == "__main__":
    json_data = to_json()
    with open('data/output.json', 'w') as f:
        json.dump(json_data, f)
    print(json_data)