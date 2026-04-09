from gui.render_utils import build_candidate_scene


def test_t_stiffener_preview_places_flange_next_to_skin() -> None:
    candidate = {
        "candidate_id": "TMP_1",
        "geometry": {
            "panel_length_mm": 700,
            "panel_width_mm": 600,
            "skin_thickness_mm": 2.4,
            "pitch_mm": 120,
            "stiffener_height_mm": 28,
            "web_thickness_mm": 2.0,
            "flange_width_mm": 16,
            "flange_thickness_mm": 2.0,
        },
    }

    scene = build_candidate_scene(candidate)
    assert scene is not None
    meshes, _ = scene

    skin_bounds = meshes[0][0].bounds
    flange_bounds = meshes[1][0].bounds
    web_bounds = meshes[2][0].bounds

    assert flange_bounds[4] >= skin_bounds[5]
    assert web_bounds[4] >= flange_bounds[5]
