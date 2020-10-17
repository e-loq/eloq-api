import json

def export(output_path: str, points: dict):
    """
    Exports the data to the desired JSON schema

    Args:
        output_path (str): destination file path
        points (dict): point data according to the specific schema
    """
    ceiling_height = 0
    max_sat_height = 0
    data = {
        "unit": "m",
        "z_ceil":  ceiling_height,
        "z_sat": max_sat_height,
        "z_marker": 1,  #  NOTE this is given by TRUMPF
        "layers": points,
        "optimize": True,
        "marker_grid": 1,
        "sat_grid": 10
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    export('data/output.json', {})