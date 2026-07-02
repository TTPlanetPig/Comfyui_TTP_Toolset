import json
from pathlib import Path
import sys
import types

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def install_comfy_stubs():
    for name in ["cv2", "node_helpers", "latent_preview"]:
        sys.modules[name] = types.ModuleType(name)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: "."
    folder_paths.filter_files_content_types = lambda files, _types: files
    folder_paths.get_annotated_filepath = lambda image: image
    folder_paths.exists_annotated_filepath = lambda image: True
    sys.modules["folder_paths"] = folder_paths

    torch = types.ModuleType("torch")

    class Tensor:
        pass

    class FakeTensor:
        def __init__(self, array):
            self.array = array
            self.shape = getattr(array, "shape", ())

        def unsqueeze(self, _dim):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    torch.Tensor = Tensor
    torch.from_numpy = lambda array: FakeTensor(array)
    torch.cat = lambda tensors, dim=0: tensors
    torch.unsqueeze = lambda tensor, dim: tensor
    sys.modules["torch"] = torch

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    sys.modules["comfy"] = comfy
    for name in ["model_management", "samplers", "sample", "utils"]:
        module = types.ModuleType(f"comfy.{name}")
        sys.modules[f"comfy.{name}"] = module
        setattr(comfy, name, module)


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


install_comfy_stubs()

import TTP_toolsets as ttp  # noqa: E402


# This mirrors JSON.stringify output from the browser: exact 0/1 edges are
# serialized as integers, while inner grid lines stay fractional.
browser_grid_layout = {
    "tiles": [
        {
            "name": f"tile_{row}_{col}",
            "x0": 0 if col == 0 else col / 3,
            "y0": 0 if row == 0 else row / 3,
            "x1": 1 if col == 2 else (col + 1) / 3,
            "y1": 1 if row == 2 else (row + 1) / 3,
        }
        for row in range(3)
        for col in range(3)
    ]
}

layout_json = json.dumps(browser_grid_layout, separators=(",", ":"))
normalized_layout = ttp._ttp_interactive_layout_with_defaults(layout_json, 64, 32, False)
tiles_meta = ttp._ttp_parse_smart_tile_layout(normalized_layout, 900, 600)

expected_core_boxes = [
    [0, 0, 300, 200],
    [300, 0, 300, 200],
    [600, 0, 300, 200],
    [0, 200, 300, 200],
    [300, 200, 300, 200],
    [600, 200, 300, 200],
    [0, 400, 300, 200],
    [300, 400, 300, 200],
    [600, 400, 300, 200],
]

assert_equal([tile["core_box"] for tile in tiles_meta], expected_core_boxes, "3x3 core boxes")

expected_sample_boxes = [
    [0, 0, 364, 264],
    [236, 0, 428, 264],
    [536, 0, 364, 264],
    [0, 136, 364, 328],
    [236, 136, 428, 328],
    [536, 136, 364, 328],
    [0, 336, 364, 264],
    [236, 336, 428, 264],
    [536, 336, 364, 264],
]

assert_equal([tile["sample_box"] for tile in tiles_meta], expected_sample_boxes, "3x3 overlap sample boxes")

_tiles, crop_meta, _positions, _preview = ttp._ttp_crop_smart_tiles_from_meta(
    Image.new("RGB", (900, 600), "white"),
    [dict(tile) for tile in tiles_meta],
    8,
)
expected_batched_sample_boxes = [
    [0, 0, 432, 328],
    [236, 0, 432, 328],
    [468, 0, 432, 328],
    [0, 136, 432, 328],
    [236, 136, 432, 328],
    [468, 136, 432, 328],
    [0, 272, 432, 328],
    [236, 272, 432, 328],
    [468, 272, 432, 328],
]
assert_equal(
    [tile["sample_box"] for tile in crop_meta["tiles"]],
    expected_batched_sample_boxes,
    "3x3 batch crops should expand inward with real image pixels instead of transport padding",
)
assert_equal(
    crop_meta["tiles"][0]["tile_canvas_box"],
    [0, 0, 432, 328],
    "top-left crop should fill the batch canvas with real pixels",
)
assert_equal(
    crop_meta["tiles"][0]["tile_canvas_size"],
    [432, 328],
    "top-left batch canvas should match largest crop",
)
assert_equal(
    crop_meta["tiles"][8]["tile_canvas_box"],
    [0, 0, 432, 328],
    "bottom-right crop should fill the batch canvas with real pixels",
)
assert_equal(
    crop_meta["tiles"][0]["overlap_edges_px_source"],
    {"left": 0, "right": 132, "top": 0, "bottom": 128},
    "top-left batch crop should grow only inward",
)
assert_equal(
    crop_meta["tiles"][8]["overlap_edges_px_source"],
    {"left": 132, "right": 0, "top": 128, "bottom": 0},
    "bottom-right batch crop should grow only inward",
)

assert_equal(
    tiles_meta[0]["overlap_edges_px_source"],
    {"left": 0, "right": 64, "top": 0, "bottom": 64},
    "top-left grid overlap edges",
)
assert_equal(
    tiles_meta[4]["overlap_edges_px_source"],
    {"left": 64, "right": 64, "top": 64, "bottom": 64},
    "center grid overlap edges",
)
assert_equal(
    tiles_meta[8]["overlap_edges_px_source"],
    {"left": 64, "right": 0, "top": 64, "bottom": 0},
    "bottom-right grid overlap edges",
)

gappy_layout = json.dumps({
    "tiles": [
        {"name": "left", "x0": 0, "y0": 0, "x1": 0.25, "y1": 1},
        {"name": "right", "x0": 0.5, "y0": 0, "x1": 1, "y1": 1},
    ]
}, separators=(",", ":"))
gappy_normalized = ttp._ttp_interactive_layout_with_defaults(gappy_layout, 64, 32, False)
gappy_meta = ttp._ttp_parse_smart_tile_layout(gappy_normalized, 800, 600)
assert_equal(
    [tile["sample_box"] for tile in gappy_meta],
    [[0, 0, 200, 600], [400, 0, 400, 600]],
    "non-adjacent manual tiles must not overlap across a gap",
)
assert_equal(
    [tile["overlap_edges_px_source"] for tile in gappy_meta],
    [
        {"left": 0, "right": 0, "top": 0, "bottom": 0},
        {"left": 0, "right": 0, "top": 0, "bottom": 0},
    ],
    "non-adjacent manual overlap edges",
)

touching_layout = json.dumps({
    "tiles": [
        {"name": "left", "x0": 0, "y0": 0, "x1": 0.5, "y1": 1},
        {"name": "right", "x0": 0.5, "y0": 0, "x1": 1, "y1": 1},
    ]
}, separators=(",", ":"))
touching_normalized = ttp._ttp_interactive_layout_with_defaults(touching_layout, 64, 32, False)
touching_meta = ttp._ttp_parse_smart_tile_layout(touching_normalized, 800, 600)
assert_equal(
    [tile["sample_box"] for tile in touching_meta],
    [[0, 0, 464, 600], [336, 0, 464, 600]],
    "adjacent manual tiles should overlap only on the shared edge",
)
assert_equal(
    [tile["overlap_edges_px_source"] for tile in touching_meta],
    [
        {"left": 0, "right": 64, "top": 0, "bottom": 0},
        {"left": 64, "right": 0, "top": 0, "bottom": 0},
    ],
    "adjacent manual overlap edges",
)

print("smoke_smart_tile_grid_crop: ok")
