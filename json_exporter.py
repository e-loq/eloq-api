import json
import base64

def export(output_path: str, points: list, background_img_path: str, ceil_height, max_z):
    """
    Exports the data to the desired JSON schema

    Args:
        output_path (str): destination file path
        points (dict): point data according to the specific schema
    """
    ceiling_height = ceil_height
    max_sat_height = max_z
    data = {
        "unit": "m",
        "z_ceil":  ceiling_height,
        "z_sat": max_sat_height,
        "z_marker": 1,  #  NOTE this is given by TRUMPF
        "allShapes": [],
        "layers": points,
        "optimize": True,
        "marker_grid": 1,
        "sat_grid": 10,
        "img": base64.b64encode(open(background_img_path, "rb").read()).decode('ascii')
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    export('data/output.json', {})