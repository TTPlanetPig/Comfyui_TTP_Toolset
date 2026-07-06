import json
import base64
from io import BytesIO
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def install_comfy_stubs():
    for name in ["cv2", "node_helpers", "latent_preview"]:
        sys.modules[name] = types.ModuleType(name)

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_input_directory = lambda: "."
    folder_paths.get_output_directory = lambda: "."
    folder_paths.filter_files_content_types = lambda files, _types: files
    folder_paths.get_annotated_filepath = lambda image: image
    folder_paths.exists_annotated_filepath = lambda image: True
    folder_paths.get_save_image_path = lambda filename_prefix, output_dir, image_width=0, image_height=0: (
        ".",
        filename_prefix,
        1,
        "",
        filename_prefix,
    )
    sys.modules["folder_paths"] = folder_paths

    torch = types.ModuleType("torch")

    class Tensor:
        pass

    class FakeTensor:
        def __init__(self, array):
            self.array = array
            self.shape = getattr(array, "shape", ())

        def unsqueeze(self, dim):
            if dim == 0:
                return FakeTensor(self.array[None, ...])
            return self

        def __getitem__(self, item):
            return FakeTensor(self.array[item])

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
    for name in ["model_management", "sd", "samplers", "sample", "utils"]:
        module = types.ModuleType(f"comfy.{name}")
        sys.modules[f"comfy.{name}"] = module
        setattr(comfy, name, module)
    cli_args = types.ModuleType("comfy.cli_args")
    cli_args.args = types.SimpleNamespace(disable_metadata=False)
    sys.modules["comfy.cli_args"] = cli_args


def assert_equal(actual, expected, message):
    if actual != expected:
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


def assert_image_close(actual, expected, message, tolerance=1e-6):
    actual_array = getattr(actual, "array", actual)
    expected_array = getattr(expected, "array", expected)
    if not np.allclose(actual_array, expected_array, atol=tolerance, rtol=0.0):
        max_delta = float(np.max(np.abs(actual_array - expected_array)))
        raise AssertionError(f"{message}: max delta {max_delta} > {tolerance}")


def call_assemble_like_comfy(node, widget_values, **linked_inputs):
    kwargs = {}
    required = node.INPUT_TYPES()["required"]
    for index, (name, spec) in enumerate(required.items()):
        if index < len(widget_values):
            kwargs[name] = widget_values[index]
            continue
        if isinstance(spec, tuple) and len(spec) > 1 and isinstance(spec[1], dict) and "default" in spec[1]:
            kwargs[name] = spec[1]["default"]
    kwargs.update(linked_inputs)
    return node.assemble_tiles(**kwargs)


install_comfy_stubs()

import TTP_toolsets as ttp  # noqa: E402


assert_equal(
    "TTP_Smart_Tile_Layout_Experimental" in ttp.NODE_CLASS_MAPPINGS,
    False,
    "old Smart Tile layout step node should not be registered",
)
assert_equal(
    "TTP_Smart_Tile_Crop_Experimental" in ttp.NODE_CLASS_MAPPINGS,
    False,
    "old Smart Tile crop step node should not be registered",
)
assert_equal(
    "TTP_Smart_Tile_Visual_Crop_Experimental" in ttp.NODE_CLASS_MAPPINGS,
    False,
    "old Smart Tile param crop step node should not be registered",
)
assert_equal(
    "(Experimental)" in ttp.NODE_DISPLAY_NAME_MAPPINGS["TTP_Smart_Tile_Interactive_Crop_Experimental"],
    False,
    "interactive crop display name should not include Experimental",
)
assert_equal(
    ttp.NODE_DISPLAY_NAME_MAPPINGS["TTP_Smart_Tile_Loop_Source_Experimental"],
    "TTP Smart Tile Loop Source",
    "loop workflow display names should be stable and non-experimental",
)
assert_equal(
    ttp.TTP_Smart_Tile_Interactive_Crop_Experimental.CATEGORY,
    "TTP/Smart Tile",
    "Smart Tile nodes should use the non-experimental menu category",
)

interactive_inputs = ttp.TTP_Smart_Tile_Interactive_Crop_Experimental.INPUT_TYPES()
required_inputs = interactive_inputs["required"]
optional_inputs = interactive_inputs["optional"]
hidden_inputs = interactive_inputs["hidden"]
assert_equal(
    ttp.TTP_Smart_Tile_Interactive_Crop_Experimental.RETURN_TYPES,
    ("IMAGE", "IMAGE", "TTP_SMART_TILE_SET", "TTP_SMART_TILE_META", "LIST", "IMAGE", "STRING"),
    "interactive crop should expose a tile_set output before metadata",
)
assert_equal(
    required_inputs["auto_detect_mode"][0],
    ["none", "sam3.1", "qwenvl3"],
    "interactive crop should expose the requested auto detect modes",
)
assert_equal("auto_detect_request" in required_inputs, True, "interactive crop should expose an inference request trigger")
assert_equal("auto_prompt" in required_inputs, True, "interactive crop should expose a visual prompt")
assert_equal("allow_object_overlap" in required_inputs, True, "interactive crop should expose object overlap control")
assert_equal("auto_object_padding" in required_inputs, True, "interactive crop should expose object padding control")
assert_equal("auto_max_tiles" in required_inputs, True, "interactive crop should expose auto max tiles control")
assert_equal("auto_paint_mask" in required_inputs, True, "interactive crop should expose hidden painted mask input")
assert_equal("vision_model" in optional_inputs, True, "interactive crop should expose a vision model input")
assert_equal("vision_conditioning" in optional_inputs, True, "interactive crop should expose official SAM3 conditioning input")
assert_equal("clip" in optional_inputs, True, "interactive crop should optionally encode SAM3 prompts with CLIP")
assert_equal("qwen_vl_model" in optional_inputs, True, "interactive crop should accept local QwenVL model for bbox auto tile")
assert_equal(hidden_inputs["unique_id"], "UNIQUE_ID", "interactive crop should receive its ComfyUI node id")

assemble_inputs = ttp.TTP_Smart_Tile_Assemble_Experimental.INPUT_TYPES()
expected_assemble_required_order = [
    "blend_multiplier",
    "output_scale",
    "use_priority",
    "tile_alignment",
    "edge_crop_px",
    "color_correction",
    "color_strength",
    "mask_blend_mode",
    "pixel_alignment",
    "pixel_alignment_radius",
    "pixel_alignment_device",
    "large_tile_policy",
    "large_tile_area_threshold",
    "min_tile_scale_ratio",
    "context_tile_weight",
    "assemble_device",
    "assemble_mode",
    "base_canvas_mode",
    "weight_preview_mode",
    "small_tile_on_top",
    "auto_composite_policy",
]
assert_equal(
    list(assemble_inputs["required"].keys()),
    expected_assemble_required_order,
    "assemble required widgets should keep the same order ComfyUI stores in workflow widget values",
)
assert_equal("sampled_tiles" in assemble_inputs["optional"], True, "assemble should keep batch tiles as an optional compatibility input")
assert_equal("tile_meta" in assemble_inputs["optional"], True, "assemble should keep batch metadata as an optional compatibility input")
assert_equal("tile_set" in assemble_inputs["optional"], True, "assemble should accept true variable-size tile sets")
assert_equal("local_mean_std" in assemble_inputs["required"]["color_correction"][0], True, "assemble should expose local mean/std color correction")
assert_equal("tile_alignment" in assemble_inputs["required"], True, "assemble should expose tile alignment")
assert_equal("edge_crop_px" in assemble_inputs["required"], True, "assemble should expose edge crop control")
assert_equal("mask_blend_mode" in assemble_inputs["required"], True, "assemble should expose object mask blending")
assert_equal("pixel_alignment" in assemble_inputs["required"], True, "assemble should expose pixel alignment")
assert_equal("pixel_alignment_radius" in assemble_inputs["required"], True, "assemble should expose pixel alignment radius")
assert_equal(assemble_inputs["required"]["pixel_alignment_device"][0], ["auto", "cpu", "gpu"], "assemble should expose CPU/GPU pixel alignment selection")
assert_equal(assemble_inputs["required"]["assemble_device"][0], ["auto", "cpu", "gpu"], "assemble should expose CPU/GPU paste selection")
assert_equal(assemble_inputs["required"]["assemble_mode"][0], ["final_only", "always"], "assemble should expose final-only mode first")
assert_equal(assemble_inputs["required"]["assemble_mode"][1]["default"], "final_only", "assemble should default to final-only loop compositing")
assert_equal(assemble_inputs["required"]["base_canvas_mode"][0], ["auto", "black", "base_image", "source_image"], "assemble should expose base canvas source selection")
assert_equal(assemble_inputs["required"]["weight_preview_mode"][0], ["raw_weight", "coverage"], "assemble should expose raw weight and coverage preview modes")
assert_equal(assemble_inputs["required"]["weight_preview_mode"][1]["default"], "raw_weight", "assemble should keep raw overlap weight preview as the default")
assert_equal("small_tile_on_top" in assemble_inputs["required"], True, "assemble should expose small tile top stacking")
assert_equal(assemble_inputs["required"]["auto_composite_policy"][0], ["safe_auto", "strict_layer", "soft_detail", "replace_object"], "assemble should expose automatic compositing policy")
assert_equal(assemble_inputs["required"]["auto_composite_policy"][1]["default"], "safe_auto", "assemble should default to safe auto compositing")
assert_equal(assemble_inputs["optional"]["done"][0], "BOOLEAN", "assemble should accept loop done gate")
preview_inputs = ttp.TTP_Smart_Tile_Set_Preview_Experimental.INPUT_TYPES()
assert_equal(preview_inputs["required"]["tile_set"][0], "TTP_SMART_TILE_SET", "tile set preview should accept Smart Tile Set")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Set_Preview_Experimental"],
    ttp.TTP_Smart_Tile_Set_Preview_Experimental,
    "tile set preview should be registered",
)
prompt_builder_inputs = ttp.TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental.INPUT_TYPES()
assert_equal("tile_set" in prompt_builder_inputs["required"], False, "prompt builder should not put forceInput tile_set before visible widgets")
assert_equal(prompt_builder_inputs["optional"]["tile_set"][1].get("forceInput"), True, "prompt builder tile_set should be an optional forceInput connection")
assert_equal(prompt_builder_inputs["required"]["reference_image_mode"][0], ["none", "first_message", "every_tile", "contact_sheet"], "prompt builder should support reference image and contact sheet strategies")
assert_equal("mode" in prompt_builder_inputs["required"], False, "prompt builder should auto-select template or local QwenVL mode from the model input")
assert_equal("prompt_preset" in prompt_builder_inputs["required"], True, "prompt builder should expose preset prompt selection")
assert_equal("qwen_max_side" in prompt_builder_inputs["required"], True, "prompt builder should expose Qwen image resize control")
assert_equal("use_tile_cache" in prompt_builder_inputs["required"], True, "prompt builder should expose tile-level cache control")
assert_equal("endpoint_url" in prompt_builder_inputs["required"], False, "prompt builder should not expose API endpoint fields")
assert_equal("model_name" in prompt_builder_inputs["required"], False, "prompt builder should use the connected local QwenVL model instead of a model-name widget")
prompt_required_order = list(prompt_builder_inputs["required"].keys())
assert_equal(
    prompt_required_order.index("prompt_preset") > prompt_required_order.index("temperature"),
    True,
    "new prompt builder widgets should be appended after legacy prompt fields to avoid saved-widget value shifts",
)
assert_equal(
    prompt_required_order.index("global_negative") > prompt_required_order.index("use_tile_cache"),
    True,
    "global negative should stay at the end to avoid shifting core prompt widgets",
)
assert_equal(prompt_builder_inputs["required"]["system_prompt"][1]["dynamicPrompts"], False, "system prompt should not use ComfyUI dynamic prompt parsing")
assert_equal(prompt_builder_inputs["required"]["tile_instruction"][1]["dynamicPrompts"], False, "tile instruction should not use ComfyUI dynamic prompt parsing")
assert_equal("seed" in prompt_builder_inputs["required"], False, "prompt builder should avoid ComfyUI's special seed widget name")
assert_equal("seed" in prompt_builder_inputs["optional"], False, "prompt builder should not expose the legacy seed widget name")
assert_equal(prompt_builder_inputs["required"]["qwen_seed"][0], "INT", "prompt builder should expose qwen_seed for local QwenVL sampling")
assert_equal(prompt_builder_inputs["optional"]["qwen_vl_model"][0], "TTP_QWENVL3_MODEL", "prompt builder should accept local QwenVL loader output")
qwen_loader_inputs = ttp.TTP_QwenVL3_Local_Loader_Experimental.INPUT_TYPES()
assert_equal("model_file" in qwen_loader_inputs["required"], True, "QwenVL local loader should expose a safetensors file picker")
assert_equal(qwen_loader_inputs["required"]["model_family"][0], ["auto", "qwen_vl"], "QwenVL local loader should focus on visual tagging model family")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_QwenVL3_Local_Loader_Experimental"],
    ttp.TTP_QwenVL3_Local_Loader_Experimental,
    "QwenVL local loader should be registered",
)
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental"],
    ttp.TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental,
    "prompt builder should be registered",
)
prompt_override_inputs = ttp.TTP_Smart_Tile_Prompt_Override_Experimental.INPUT_TYPES()
assert_equal(prompt_override_inputs["required"]["tile_set"][0], "TTP_SMART_TILE_SET", "prompt override should accept Smart Tile Set")
assert_equal(prompt_override_inputs["required"]["tile_set"][1].get("forceInput"), True, "prompt override tile_set should be a forceInput connection")
assert_equal("regex" in prompt_override_inputs["required"]["selector_type"][0], True, "prompt override should support regex selectors")
assert_equal(prompt_override_inputs["required"]["unmatched_mode"][0], ["keep", "drop_unmatched"], "prompt override should allow dropping unmatched tiles")
assert_equal(prompt_override_inputs["required"]["prompt_text"][1]["dynamicPrompts"], False, "prompt override text should not use dynamic prompt parsing")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Prompt_Override_Experimental"],
    ttp.TTP_Smart_Tile_Prompt_Override_Experimental,
    "prompt override should be registered",
)

override_tile_set = {
    "version": 1,
    "type": "ttp_smart_tile_set",
    "original_size": [100, 100],
    "tile_meta": {
        "version": 3,
        "type": "ttp_smart_tile",
        "storage": "tile_set",
        "original_size": [100, 100],
        "tiles": [
            {
                "id": 0,
                "name": "tile_1",
                "label": "face",
                "source": "paint_mask",
                "core_box": [0, 0, 50, 50],
                "sample_box": [0, 0, 50, 50],
                "paste_box": [0, 0, 50, 50],
                "prompt": "old face prompt",
                "negative": "old face negative",
            },
            {
                "id": 1,
                "name": "tile_2",
                "label": "background",
                "source": "manual",
                "core_box": [50, 0, 50, 50],
                "sample_box": [50, 0, 50, 50],
                "paste_box": [50, 0, 50, 50],
                "prompt": "old background prompt",
                "negative": "old background negative",
            },
        ],
    },
    "tile_images": ["face_image", "background_image"],
    "positions": [(0, 0, 50, 50), (50, 0, 100, 50)],
}
override_node = ttp.TTP_Smart_Tile_Prompt_Override_Experimental()
overridden_set, override_json, override_summary = override_node.override_prompts(
    override_tile_set,
    selector_type="label",
    selector="face",
    unmatched_mode="keep",
    prompt_mode="replace",
    prompt_text="edit only the face",
    negative_mode="append",
    negative_text="avoid changing identity",
)
assert_equal(overridden_set["tile_meta"]["tiles"][0]["prompt"], "edit only the face", "prompt override should replace matched tile prompt")
assert_equal(overridden_set["tile_meta"]["tiles"][0]["negative"], "old face negative\navoid changing identity", "prompt override should append matched tile negative")
assert_equal(overridden_set["tile_meta"]["tiles"][1]["prompt"], "old background prompt", "prompt override should keep unmatched tile prompt")
assert_equal(json.loads(override_json)["matched"], 1, "prompt override JSON should report matched tile count")
assert_equal("matched=1" in override_summary, True, "prompt override summary should report matched count")
dropped_set, _dropped_json, _dropped_summary = override_node.override_prompts(
    override_tile_set,
    selector_type="label",
    selector="face",
    unmatched_mode="drop_unmatched",
    prompt_mode="replace",
    prompt_text="edit only the face",
)
assert_equal(len(dropped_set["tile_meta"]["tiles"]), 1, "prompt override should drop unmatched tile metadata")
assert_equal(len(dropped_set["tile_images"]), 1, "prompt override should drop unmatched tile images")
assert_equal(dropped_set["tile_meta"]["tiles"][0]["id"], 0, "prompt override should reindex kept tile ids")
loop_source_inputs = ttp.TTP_Smart_Tile_Loop_Source_Experimental.INPUT_TYPES()
assert_equal(loop_source_inputs["required"]["tile_set"][0], "TTP_SMART_TILE_SET", "loop source should accept Smart Tile Set")
assert_equal(loop_source_inputs["optional"]["clip"][0], "CLIP", "loop source should optionally encode tile prompts with CLIP")
assert_equal(loop_source_inputs["hidden"]["unique_id"], "UNIQUE_ID", "loop source should know its node id for auto queue events")
assert_equal("prompt" in ttp.TTP_Smart_Tile_Loop_Source_Experimental.RETURN_NAMES, True, "loop source should output tile prompts")
assert_equal("negative" in ttp.TTP_Smart_Tile_Loop_Source_Experimental.RETURN_NAMES, True, "loop source should output tile negatives")
assert_equal("positive_conditioning" in ttp.TTP_Smart_Tile_Loop_Source_Experimental.RETURN_NAMES, True, "loop source should output encoded positive conditioning")
assert_equal("negative_conditioning" in ttp.TTP_Smart_Tile_Loop_Source_Experimental.RETURN_NAMES, True, "loop source should output encoded negative conditioning")
loop_collect_inputs = ttp.TTP_Smart_Tile_Loop_Collect_Experimental.INPUT_TYPES()
assert_equal(loop_collect_inputs["required"]["tile_task"][0], "TTP_SMART_TILE_TASK", "loop collect should accept tile tasks")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Loop_Source_Experimental"],
    ttp.TTP_Smart_Tile_Loop_Source_Experimental,
    "loop source should be registered",
)
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Loop_Collect_Experimental"],
    ttp.TTP_Smart_Tile_Loop_Collect_Experimental,
    "loop collect should be registered",
)
upscale_inputs = ttp.TTP_Smart_Tile_Image_Upscale_Prep_Experimental.INPUT_TYPES()
assert_equal(upscale_inputs["required"]["image"][0], "IMAGE", "upscale prep should accept one tile image")
assert_equal("max_megapixels" in upscale_inputs["required"], True, "upscale prep should expose a megapixel cap")
assert_equal("use_upscale_model" in upscale_inputs["required"], True, "upscale prep should allow disabling a connected upscale model")
assert_equal(upscale_inputs["optional"]["upscale_model"][0], "UPSCALE_MODEL", "upscale prep should accept ComfyUI upscale models")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Image_Upscale_Prep_Experimental"],
    ttp.TTP_Smart_Tile_Image_Upscale_Prep_Experimental,
    "upscale prep should be registered",
)
output_size_inputs = ttp.TTP_Smart_Tile_Output_Size_Estimate_Experimental.INPUT_TYPES()
assert_equal(output_size_inputs["required"]["tile_set"][0], "TTP_SMART_TILE_SET", "output size estimate should accept tile sets")
assert_equal(output_size_inputs["required"]["scale_strategy"][0], ["median", "mean", "min", "max", "focus_weighted"], "output size estimate should expose scale strategies")
assert_equal(output_size_inputs["optional"]["done"][0], "BOOLEAN", "output size estimate should accept a loop done gate")
semantic_rank_inputs = ttp.TTP_Smart_Tile_Semantic_Rank_Experimental.INPUT_TYPES()
assert_equal(semantic_rank_inputs["required"]["tile_set"][0], "TTP_SMART_TILE_SET", "semantic rank should accept Smart Tile Set")
assert_equal(semantic_rank_inputs["required"]["rank_policy"][0], ["portrait", "balanced", "product", "text"], "semantic rank should expose rank policies")
assert_equal("apply_composite_rank" in semantic_rank_inputs["required"], True, "semantic rank should optionally write composite rank metadata")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Semantic_Rank_Experimental"],
    ttp.TTP_Smart_Tile_Semantic_Rank_Experimental,
    "semantic rank should be registered",
)
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Output_Size_Estimate_Experimental"],
    ttp.TTP_Smart_Tile_Output_Size_Estimate_Experimental,
    "output size estimate should be registered",
)
save_final_inputs = ttp.TTP_Smart_Tile_Save_Final_Image_Experimental.INPUT_TYPES()
assert_equal(save_final_inputs["required"]["images"][0], "IMAGE", "save final image should accept images")
assert_equal("done" in save_final_inputs["required"], True, "save final image should expose a done gate")
assert_equal(save_final_inputs["hidden"]["extra_pnginfo"], "EXTRA_PNGINFO", "save final image should embed workflow metadata")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Save_Final_Image_Experimental"],
    ttp.TTP_Smart_Tile_Save_Final_Image_Experimental,
    "save final image should be registered",
)


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

tile_set = ttp._ttp_crop_smart_tile_set_from_meta(
    Image.new("RGB", (900, 600), "white"),
    [dict(tile) for tile in tiles_meta],
    8,
)
expected_tile_set_sample_boxes = [
    [0, 0, 368, 264],
    [236, 0, 432, 264],
    [532, 0, 368, 264],
    [0, 136, 368, 328],
    [236, 136, 432, 328],
    [532, 136, 368, 328],
    [0, 336, 368, 264],
    [236, 336, 432, 264],
    [532, 336, 368, 264],
]
assert_equal(tile_set["type"], "ttp_smart_tile_set", "tile_set should use the custom Smart Tile Set type")
assert_equal(
    [tile["sample_box"] for tile in tile_set["tile_meta"]["tiles"]],
    expected_tile_set_sample_boxes,
    "tile_set crops should keep per-tile sample boxes after round_to alignment",
)
assert_equal(
    [tile["tile_canvas_size"] for tile in tile_set["tile_meta"]["tiles"]],
    [[box[2], box[3]] for box in expected_tile_set_sample_boxes],
    "tile_set canvas sizes should stay per tile instead of batch-sized",
)
assert_equal(
    [list(image.shape[1:3]) for image in tile_set["tile_images"]],
    [[box[3], box[2]] for box in expected_tile_set_sample_boxes],
    "tile_set image tensors should preserve different tile dimensions",
)
preview_node = ttp.TTP_Smart_Tile_Set_Preview_Experimental()
contact_sheet, sheet_info = preview_node.preview_tile_set(
    tile_set,
    mode="contact_sheet",
    thumbnail_size=128,
    columns=3,
)
assert_equal(contact_sheet.shape[0], 1, "tile set contact sheet should output a regular IMAGE tensor")
assert_equal("368x264" in sheet_info and "432x328" in sheet_info, True, "tile set preview info should report real tile sizes")
selected_tile, selected_info = preview_node.preview_tile_set(
    tile_set,
    mode="selected_tile",
    selected_index=4,
)
assert_equal(list(selected_tile.shape[1:3]), [328, 432], "selected tile preview should preserve the selected tile size")
assert_equal("4:" in selected_info, True, "selected tile preview should include tile index info")

grid_assemble_node = ttp.TTP_Smart_Tile_Assemble_Experimental()
_grid_output_raw, grid_raw_weight_preview = grid_assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    tile_set=tile_set,
)
assert_equal(float(grid_raw_weight_preview.array.max()), 1.0, "raw 3x3 weight preview should normalize the highest overlap to white")
assert_equal(float(grid_raw_weight_preview.array.min()) < 1.0, True, "raw 3x3 weight preview should show lower-weight single tile areas darker than overlap zones")
_grid_output, grid_coverage_preview = grid_assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    weight_preview_mode="coverage",
    mask_blend_mode="mask_feather",
    tile_set=tile_set,
)
assert_equal(float(grid_coverage_preview.array.min()), 1.0, "black canvas 3x3 coverage preview should be uniformly covered")
assert_equal(float(grid_coverage_preview.array.max()), 1.0, "black canvas 3x3 coverage preview should not show raw overlap weight grid lines")

comfy_assemble_widget_values = [
    1.0,
    0.0,
    True,
    "resize",
    0,
    "off",
    0.35,
    "mask_feather",
    "off",
    0,
    "cpu",
    "use_if_higher_resolution",
    0.55,
    0.95,
    0.25,
    "cpu",
    "final_only",
    "black",
    "raw_weight",
    False,
    "safe_auto",
]
direct_comfy_output, direct_comfy_weights = grid_assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    tile_alignment="resize",
    edge_crop_px=0,
    color_correction="off",
    color_strength=0.35,
    mask_blend_mode="mask_feather",
    pixel_alignment="off",
    pixel_alignment_radius=0,
    pixel_alignment_device="cpu",
    large_tile_policy="use_if_higher_resolution",
    large_tile_area_threshold=0.55,
    min_tile_scale_ratio=0.95,
    context_tile_weight=0.25,
    assemble_device="cpu",
    assemble_mode="final_only",
    base_canvas_mode="black",
    weight_preview_mode="raw_weight",
    small_tile_on_top=False,
    auto_composite_policy="safe_auto",
    tile_set=tile_set,
    done=True,
)
replayed_comfy_output, replayed_comfy_weights = call_assemble_like_comfy(
    grid_assemble_node,
    comfy_assemble_widget_values,
    output_scale=1.0,
    tile_set=tile_set,
    done=True,
)
assert_image_close(replayed_comfy_output, direct_comfy_output, "ComfyUI widget-order assemble replay should match direct assemble output")
assert_image_close(replayed_comfy_weights, direct_comfy_weights, "ComfyUI widget-order assemble replay should match direct assemble weight preview")

ranked_grid_meta = dict(tile_set["tile_meta"])
ranked_grid_tiles = []
for tile_index, tile in enumerate(tile_set["tile_meta"]["tiles"]):
    ranked_tile = dict(tile)
    ranked_tile["layer"] = tile_index
    ranked_tile["occlusion_priority"] = tile_index
    ranked_grid_tiles.append(ranked_tile)
ranked_grid_meta["tiles"] = ranked_grid_tiles
ranked_grid_tile_set = dict(tile_set)
ranked_grid_tile_set["tile_meta"] = ranked_grid_meta
_ranked_grid_output, ranked_grid_weight_preview = grid_assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    auto_composite_policy="safe_auto",
    tile_set=ranked_grid_tile_set,
)
ranked_width, ranked_height = ranked_grid_meta["original_size"]
internal_x = sorted({tile["core_box"][0] for tile in ranked_grid_tiles if tile["core_box"][0] > 0})
internal_y = sorted({tile["core_box"][1] for tile in ranked_grid_tiles if tile["core_box"][1] > 0})
column_centers = sorted({min(ranked_width - 1, tile["core_box"][0] + tile["core_box"][2] // 2) for tile in ranked_grid_tiles[:3]})
row_centers = sorted({min(ranked_height - 1, ranked_grid_tiles[row * 3]["core_box"][1] + ranked_grid_tiles[row * 3]["core_box"][3] // 2) for row in range(3)})
ranked_seam_values = []
for seam_x in internal_x:
    for center_y in row_centers:
        ranked_seam_values.append(float(ranked_grid_weight_preview.array[0, center_y, seam_x, 0]))
for seam_y in internal_y:
    for center_x in column_centers:
        ranked_seam_values.append(float(ranked_grid_weight_preview.array[0, seam_y, center_x, 0]))
assert_equal(min(ranked_seam_values) > 0.45, True, "safe auto ranked 3x3 grid seams should keep both tile contributions instead of showing dark split lines")

prompt_builder = ttp.TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental()
prompt_tile_set, prompt_set_json, prompt_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="template",
    reference_image_mode="every_tile",
    global_prompt="masterpiece",
    global_negative="blurry",
)
prompt_data = json.loads(prompt_set_json)
assert_equal(prompt_data["type"], "ttp_smart_tile_prompt_set", "prompt builder should output prompt set JSON")
assert_equal(len(prompt_data["tiles"]), len(tile_set["tile_images"]), "prompt builder should create one prompt per tile")
assert_equal("masterpiece" in prompt_tile_set["tile_meta"]["tiles"][0]["prompt"], True, "prompt builder should merge global prompt")
assert_equal(prompt_tile_set["tile_meta"]["tiles"][0]["negative"], "blurry", "prompt builder should write tile negative prompts")
assert_equal("0:" in prompt_summary, True, "prompt builder should summarize prompts")

shifted_tile_set, shifted_prompt_json, _shifted_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="template",
    reference_image_mode="every_tile",
    system_prompt="tile_json_strict",
    tile_instruction="custom system",
    global_prompt="custom tile instruction",
    global_negative="masterpiece",
    prompt_merge_mode="blurry",
    output_language="global_plus_caption",
    max_new_tokens="chinese",
    temperature=768,
    prompt_preset=0.2,
)
shifted_prompt_data = json.loads(shifted_prompt_json)
assert_equal(shifted_prompt_data["prompt_preset"], "tile_json_strict", "prompt builder should recover preset from shifted widget values")
assert_equal(shifted_prompt_data["tiles"][0]["negative"], "blurry", "prompt builder should recover global negative from shifted widget values")
assert_equal("masterpiece" in shifted_tile_set["tile_meta"]["tiles"][0]["prompt"], True, "prompt builder should recover global prompt from shifted widget values")

class FakeQwenClip:
    last_seed = None
    last_text = ""
    last_kwargs = {}

    def tokenize(self, text, **kwargs):
        self.last_text = text
        self.last_kwargs = kwargs
        return {"text": text, **kwargs}

    def generate(self, tokens, **kwargs):
        self.last_seed = kwargs.get("seed")
        return ["fake_ids"]

    def decode(self, output_ids, skip_special_tokens=True):
        tile_index = "unknown"
        marker = "Tile index:"
        if marker in self.last_text:
            tile_index = self.last_text.split(marker, 1)[1].split(".", 1)[0].strip()
        return '{"label":"qwen tile %s","caption":"Qwen saw tile %s.","prompt":"qwen detailed tile %s prompt","negative":"qwen negative %s"}' % (tile_index, tile_index, tile_index, tile_index)


fake_qwen_clip = FakeQwenClip()
qwen_tile_set, qwen_prompt_json, qwen_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="template",
    reference_image_mode="every_tile",
    global_prompt="masterpiece",
    global_negative="blurry",
    qwen_seed=1234,
    qwen_vl_model={
        "type": "ttp_qwenvl3_model",
        "model_file": "fake_qwen_vl.safetensors",
        "clip": fake_qwen_clip,
    },
)
qwen_prompt_data = json.loads(qwen_prompt_json)
assert_equal(qwen_prompt_data["mode"], "qwen_vl_local", "connected local QwenVL model should override template fallback")
assert_equal(qwen_prompt_data["model_file"], "fake_qwen_vl.safetensors", "QwenVL prompt set JSON should record the connected local model file")
assert_equal(qwen_prompt_data["qwen_seed"], 1234, "QwenVL prompt set JSON should record the local sampling seed")
assert_equal(qwen_prompt_data["prompt_preset"], "tile_img2img_prompt", "QwenVL prompt set JSON should record the prompt preset")
assert_equal("mode=qwen_vl_local" in qwen_summary, True, "QwenVL summary should report the effective mode")
assert_equal(qwen_tile_set["tile_meta"]["tiles"][0]["prompt"], "qwen detailed tile 0 prompt", "QwenVL prompt should come from model output")
assert_equal(qwen_tile_set["tile_meta"]["tiles"][1]["prompt"], "qwen detailed tile 1 prompt", "QwenVL should write a distinct prompt per tile")
assert_equal(qwen_tile_set["tile_meta"]["tiles"][0]["qwen_cache"], "miss", "first QwenVL prompt build should miss tile cache")
assert_equal(fake_qwen_clip.last_seed, 1234, "local QwenVL generate should receive an integer seed")
assert_equal("llama_template" in fake_qwen_clip.last_kwargs, False, "local QwenVL should use ComfyUI native Qwen image chat template")
assert_equal("images" in fake_qwen_clip.last_kwargs, True, "local QwenVL should pass images through the native tokenizer path")

cached_qwen_tile_set, _cached_qwen_prompt_json, _cached_qwen_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="qwen_vl_local",
    reference_image_mode="every_tile",
    prompt_preset="tile_img2img_prompt",
    global_prompt="masterpiece",
    global_negative="blurry",
    qwen_seed=1234,
    qwen_vl_model={
        "type": "ttp_qwenvl3_model",
        "model_file": "fake_qwen_vl.safetensors",
        "clip": fake_qwen_clip,
    },
)
assert_equal(cached_qwen_tile_set["tile_meta"]["tiles"][0]["qwen_cache"], "hit", "repeated QwenVL prompt build should hit tile cache")

legacy_seed_clip = FakeQwenClip()
_legacy_seed_tile_set, legacy_seed_prompt_json, _legacy_seed_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="qwen_vl_local",
    seed=5678,
    qwen_vl_model={
        "type": "ttp_qwenvl3_model",
        "model_file": "fake_legacy_seed_qwen_vl.safetensors",
        "clip": legacy_seed_clip,
    },
)
legacy_seed_prompt_data = json.loads(legacy_seed_prompt_json)
assert_equal(legacy_seed_prompt_data["qwen_seed"], 5678, "legacy seed kwargs should map to qwen_seed")
assert_equal(legacy_seed_clip.last_seed, 5678, "legacy seed kwargs should still reach local QwenVL generate")

small_qwen_clip = FakeQwenClip()
small_qwen_tile_set, _small_qwen_prompt_json, _small_qwen_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="qwen_vl_local",
    reference_image_mode="contact_sheet",
    prompt_preset="tile_json_strict",
    qwen_max_side=96,
    qwen_max_pixels=0,
    use_tile_cache=False,
    qwen_vl_model={
        "type": "ttp_qwenvl3_model",
        "model_file": "fake_small_qwen_vl.safetensors",
        "clip": small_qwen_clip,
    },
)
small_size = small_qwen_tile_set["tile_meta"]["tiles"][0]["qwen_input_size"]
assert_equal(max(small_size) <= 96, True, "QwenVL prompt builder should resize tile inputs for inference")

miswired_qwen_clip = FakeQwenClip()
miswired_tile_set, miswired_prompt_json, _miswired_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="template",
    reference_image= {
        "type": "ttp_qwenvl3_model",
        "model_file": "fake_miswired_qwen_vl.safetensors",
        "clip": miswired_qwen_clip,
    },
)
miswired_prompt_data = json.loads(miswired_prompt_json)
assert_equal(miswired_prompt_data["mode"], "qwen_vl_local", "miswired QwenVL model in reference_image should be recovered")
assert_equal(miswired_tile_set["tile_meta"]["tiles"][0]["prompt"], "qwen detailed tile 0 prompt", "recovered QwenVL model should still generate prompts")

class FakeQwenRetryClip(FakeQwenClip):
    attempts = 0

    def decode(self, output_ids, skip_special_tokens=True):
        self.attempts += 1
        if "Return compact JSON only" in self.last_text:
            return '{"label":"retry tile","caption":"Recovered concise tile facts.","prompt":"recovered concise tile prompt","negative":"retry negative"}'
        return "This response keeps describing the tile without JSON until it reaches the token limit."


retry_qwen_clip = FakeQwenRetryClip()
retry_tile_set, _retry_prompt_json, _retry_summary = prompt_builder.build_prompt_set(
    tile_set,
    mode="qwen_vl_local",
    qwen_vl_model={
        "type": "ttp_qwenvl3_model",
        "model_file": "fake_retry_qwen_vl.safetensors",
        "clip": retry_qwen_clip,
    },
)
assert_equal(retry_tile_set["tile_meta"]["tiles"][0]["prompt"], "recovered concise tile prompt", "QwenVL prompt builder should retry with compact JSON-only prompt")
assert_equal("qwen_raw_retry" in retry_tile_set["tile_meta"]["tiles"][0], True, "QwenVL retry raw output should be stored for debugging")

class FakeQwenEchoClip(FakeQwenClip):
    def decode(self, output_ids, skip_special_tokens=True):
        return "Refine this tile with attention to subject identity, clothing, pose, material, lighting, camera perspective, and local details. Preserve all visible visual facts without adding or altering non-visible elements."


try:
    prompt_builder.build_prompt_set(
        tile_set,
        mode="qwen_vl_local",
        qwen_vl_model={
            "type": "ttp_qwenvl3_model",
            "model_file": "fake_echo_qwen_vl.safetensors",
            "clip": FakeQwenEchoClip(),
        },
    )
    raise AssertionError("QwenVL instruction echo should not be accepted as a tile prompt")
except RuntimeError as exc:
    assert_equal("did not return JSON fields" in str(exc), True, "QwenVL instruction echo should fail with a clear JSON error")

partial_qwen = '{ "label": "object_5", "caption": "close-up portrait with warm light", "prompt": "close-up portrait with warm'
partial_record = ttp._ttp_parse_qwen_tile_record(partial_qwen, 5)
assert_equal(partial_record["label"], "object_5", "partial Qwen JSON should preserve label")
assert_equal(partial_record["caption"], "close-up portrait with warm light", "partial Qwen JSON should preserve completed caption")
assert_equal(partial_record["prompt"], "close-up portrait with warm", "partial Qwen JSON should salvage truncated prompt text")
messy_qwen = 'assistant:\n<think>hidden reasoning</think>```json\n{"label":"face","caption":"sharp eyes","prompt":"detailed face","negative":""}\n```'
messy_record = ttp._ttp_parse_qwen_tile_record(messy_qwen, 0)
assert_equal(messy_record["label"], "face", "Qwen cleaner should remove assistant prefixes, markdown fences, and thinking blocks")

bbox_items = ttp._ttp_qwen_bbox_items('```json\n[{"label":"face","bbox":[300,200,500,450],"score":0.9}]\n```', 1000, 800)
assert_equal(len(bbox_items), 1, "Qwen bbox parser should read JSON list outputs")
assert_equal(bbox_items[0]["label"], "face", "Qwen bbox parser should preserve labels")
assert_equal(round(bbox_items[0]["x"]), 300, "Qwen bbox parser should convert normalized x coordinates")
single_bbox_item = ttp._ttp_qwen_bbox_items('{"label":"face","bbox_2d":[300,200,500,450],"score":0.9}', 1000, 800)
assert_equal(len(single_bbox_item), 1, "Qwen bbox parser should accept a single JSON object output")
wrapped_bbox_items = ttp._ttp_qwen_bbox_items('{"objects":[{"label":"hand","box_2d":[100,120,260,360],"score":0.8}]}', 1000, 800)
assert_equal(wrapped_bbox_items[0]["label"], "hand", "Qwen bbox parser should accept objects-wrapped outputs")
dict_bbox_items = ttp._ttp_qwen_bbox_items('{"label":"text","rect":{"x":100,"y":120,"width":200,"height":80},"score":0.7}', 1000, 800)
assert_equal(round(dict_bbox_items[0]["width"]), 200, "Qwen bbox parser should accept x/y/width/height rectangle outputs")
xywh_bbox_items = ttp._ttp_qwen_bbox_items('[{"label":"detail","bbox":[700,100,200,300],"bbox_format":"xywh"}]', 1000, 800)
assert_equal(round(xywh_bbox_items[0]["x"] + xywh_bbox_items[0]["width"]), 900, "Qwen bbox parser should accept xywh list outputs")
large_qwen_items, large_qwen_expanded = ttp._ttp_expand_single_large_qwen_bbox(
    [{"x": 0, "y": 0, "width": 1000, "height": 800, "label": "full image", "score": 1.0}],
    1000,
    800,
    8,
)
assert_equal(large_qwen_expanded, True, "Qwen auto tile should split a single full-frame bbox into useful regions")
assert_equal(len(large_qwen_items), 4, "Qwen large bbox fallback should create a 2x2 tile set")

loop_source = ttp.TTP_Smart_Tile_Loop_Source_Experimental()
loop_collect = ttp.TTP_Smart_Tile_Loop_Collect_Experimental()

class FakeClip:
    def tokenize(self, text):
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [["conditioning", tokens["text"]]]


sam3_conditioning = ttp._ttp_encode_sam3_prompt_conditioning(FakeClip(), "person, face, hands", max_detections=2)
sam3_meta = sam3_conditioning[0][1]
assert_equal(len(sam3_meta["sam3_multi_cond"]), 3, "SAM3 internal prompt encoder should split comma-separated prompts")
assert_equal(sam3_meta["sam3_multi_cond"][0]["max_detections"], 2, "SAM3 internal prompt encoder should preserve max detections")

loop_image, loop_task, loop_index, loop_count, loop_done, loop_status, loop_prompt, loop_negative, loop_caption, loop_label, loop_prompt_tag, loop_positive, loop_negative_cond = loop_source.loop_source(
    prompt_tile_set,
    session_id="smoke_loop",
    restart_request=1,
    loop_request=1,
    clip=FakeClip(),
    unique_id="42",
)
assert_equal(loop_index, 0, "loop source should start at tile 0")
assert_equal(loop_count, len(tile_set["tile_images"]), "loop source should report tile count")
assert_equal(loop_done, False, "loop source should not be done at start")
assert_equal(loop_task["source_node_id"], "42", "loop task should carry source node id")
assert_equal(list(loop_image.shape[1:3]), [264, 368], "loop source should output a single real-size tile image")
assert_equal("masterpiece" in loop_prompt, True, "loop source should output the current tile prompt")
assert_equal(loop_negative, "blurry", "loop source should output the current tile negative prompt")
assert_equal(loop_positive, [["conditioning", loop_prompt]], "loop source should encode current tile positive prompt")
assert_equal(loop_negative_cond, [["conditioning", loop_negative]], "loop source should encode current tile negative prompt")
assert_equal(bool(loop_caption), True, "loop source should output the current tile caption")
processed_tile_set, done_after_first, next_index, collect_status = loop_collect.loop_collect(loop_task, loop_image)
assert_equal(done_after_first, False, "loop collect should continue after first tile")
assert_equal(next_index, 1, "loop collect should advance to tile 1")
assert_equal(processed_tile_set["type"], "ttp_smart_tile_set", "loop collect should output a processed tile set")
next_image, next_task, next_loop_index, _next_count, _next_done, _next_status, _next_prompt, _next_negative, _next_caption, _next_label, _next_prompt_tag, _next_positive, _next_negative_cond = loop_source.loop_source(
    prompt_tile_set,
    session_id="smoke_loop",
    restart_request=1,
    loop_request=2,
    unique_id="42",
)
assert_equal(next_loop_index, 1, "loop source should emit the next tile after collect")
assert_equal(list(next_image.shape[1:3]), [264, 432], "loop source next tile should keep its own size")
assert_equal(_next_prompt != loop_prompt, True, "loop source should emit a distinct prompt for the next tile when prompts differ")
session = ttp._TTP_SMART_TILE_LOOP_SESSIONS["smoke_loop"]
session["index"] = len(tile_set["tile_images"]) - 1
last_image, last_task, _last_index, _last_count, _last_done, _last_status, _last_prompt, _last_negative, _last_caption, _last_label, _last_prompt_tag, _last_positive, _last_negative_cond = loop_source.loop_source(
    prompt_tile_set,
    session_id="smoke_loop",
    restart_request=1,
    loop_request=3,
    unique_id="42",
)
_final_tile_set, final_done, final_next, final_status = loop_collect.loop_collect(last_task, last_image)
assert_equal(final_done, True, "loop collect should mark done after the last tile")
assert_equal("done" in final_status, True, "loop collect should report done status")

qwen_loop_image, qwen_loop_task, qwen_loop_index, _qwen_count, _qwen_done, _qwen_status, qwen_loop_prompt, _qwen_negative, _qwen_caption, _qwen_label, _qwen_prompt_tag, _qwen_positive, _qwen_negative_cond = loop_source.loop_source(
    qwen_tile_set,
    session_id="smoke_qwen_loop",
    restart_request=1,
    loop_request=1,
    unique_id="43",
)
assert_equal(qwen_loop_prompt, "qwen detailed tile 0 prompt", "Qwen prompt tile 0 should reach loop source")
_qwen_processed, _qwen_done_first, _qwen_next_index, _qwen_collect_status = loop_collect.loop_collect(qwen_loop_task, qwen_loop_image)
_qwen_next_image, _qwen_next_task, qwen_next_index, _qwen_next_count, _qwen_next_done, _qwen_next_status, qwen_next_prompt, _qwen_next_negative, _qwen_next_caption, _qwen_next_label, _qwen_next_prompt_tag, _qwen_next_positive, _qwen_next_negative_cond = loop_source.loop_source(
    qwen_tile_set,
    session_id="smoke_qwen_loop",
    restart_request=1,
    loop_request=2,
    unique_id="43",
)
assert_equal(qwen_next_index, 1, "Qwen loop should advance to tile 1")
assert_equal(qwen_next_prompt, "qwen detailed tile 1 prompt", "Qwen prompt tile 1 should reach loop source")

upscale_node = ttp.TTP_Smart_Tile_Image_Upscale_Prep_Experimental()
upscaled_tile, upscale_info = upscale_node.upscale_tile(loop_image, scale=1.5, round_to=16)
assert_equal(list(upscaled_tile.shape[1:3]), [400, 560], "upscale prep should scale and round tile dimensions")
assert_equal("368x264 -> 560x400" in upscale_info, True, "upscale prep should report size changes")
capped_tile, capped_info = upscale_node.upscale_tile(
    ttp.pil2tensor(Image.new("RGB", (100, 100), "white")),
    scale=4.0,
    round_to=16,
    max_megapixels=0.04,
    use_upscale_model=False,
)
assert_equal(list(capped_tile.shape[1:3]), [192, 192], "upscale prep should round down under the megapixel cap")
assert_equal(capped_tile.shape[1] * capped_tile.shape[2] <= 40000, True, "upscale prep should stay under max megapixels after rounding")
assert_equal("max_megapixels=0.04 capped" in capped_info, True, "upscale prep should report megapixel capping")
semantic_node = ttp.TTP_Smart_Tile_Semantic_Rank_Experimental()
semantic_input_tile_set = {
    "type": "ttp_smart_tile_set",
    "original_size": [100, 100],
    "tile_meta": {
        "type": "ttp_smart_tile",
        "original_size": [100, 100],
        "tiles": [
            {"name": "bg", "label": "background", "sample_box": [0, 0, 100, 100]},
            {"name": "face", "label": "face", "caption": "portrait face", "sample_box": [20, 20, 50, 50]},
            {"name": "eyes", "label": "eyes", "caption": "sharp eyes and eyelashes", "sample_box": [35, 35, 20, 12]},
        ],
    },
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (100, 100), "white")),
        ttp.pil2tensor(Image.new("RGB", (50, 50), "white")),
        ttp.pil2tensor(Image.new("RGB", (80, 48), "white")),
    ],
}
ranked_tile_set, semantic_report = semantic_node.rank_tiles(semantic_input_tile_set)
ranked_tiles = ranked_tile_set["tile_meta"]["tiles"]
assert_equal(ranked_tiles[0]["semantic_category"], "background", "semantic rank should detect background tiles")
assert_equal(ranked_tiles[1]["semantic_category"], "face", "semantic rank should detect face tiles")
assert_equal(ranked_tiles[2]["semantic_category"], "eyes", "semantic rank should detect eye detail tiles")
assert_equal(ranked_tiles[2]["semantic_score"] > ranked_tiles[1]["semantic_score"] > ranked_tiles[0]["semantic_score"], True, "semantic rank should score detail tiles above face and background")
assert_equal(ranked_tiles[2]["recommended_composite_mode"], "soft_overlay", "semantic rank should recommend soft overlay for eye detail tiles")
assert_equal(ranked_tiles[2]["occlusion_priority"] > ranked_tiles[1]["occlusion_priority"] > ranked_tiles[0]["occlusion_priority"], True, "semantic rank should write composite priority metadata")
assert_equal("category=eyes" in semantic_report, True, "semantic rank report should include ranked categories")
size_node = ttp.TTP_Smart_Tile_Output_Size_Estimate_Experimental()
mixed_scale_meta = {
    "type": "ttp_smart_tile",
    "original_size": [100, 50],
    "tiles": [
        {
            "name": "left",
            "sample_box": [0, 0, 50, 50],
            "tile_canvas_size": [50, 50],
            "tile_canvas_box": [0, 0, 50, 50],
        },
        {
            "name": "right",
            "sample_box": [50, 0, 50, 50],
            "tile_canvas_size": [50, 50],
            "tile_canvas_box": [0, 0, 50, 50],
        },
    ],
}
mixed_scale_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": mixed_scale_meta,
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (200, 200), "white")),
        ttp.pil2tensor(Image.new("RGB", (100, 100), "white")),
    ],
}
deferred_scale, deferred_w, deferred_h, deferred_x, deferred_y, deferred_info = size_node.estimate_output_size(mixed_scale_tile_set, done=False)
assert_equal([deferred_scale, deferred_w, deferred_h, deferred_x, deferred_y], [1.0, 100, 50, 1.0, 1.0], "output size estimate should return safe pending dimensions before loop completion")
assert_equal("deferred" in deferred_info, True, "output size estimate should report deferred status")
size_scale, size_w, size_h, size_scale_x, size_scale_y, size_info = size_node.estimate_output_size(mixed_scale_tile_set)
assert_equal(round(size_scale, 4), 3.0, "output size estimate should match assemble median scale inference")
assert_equal([size_w, size_h], [300, 150], "output size estimate should report final resolution")
assert_equal(round(size_scale_x, 4), 3.0, "output size estimate should report median x scale")
assert_equal(round(size_scale_y, 4), 3.0, "output size estimate should report median y scale")
assert_equal("warning=mixed_tile_scales" in size_info, True, "output size estimate should warn about mixed tile scales")
min_scale, min_w, min_h, _min_x, _min_y, _min_info = size_node.estimate_output_size(mixed_scale_tile_set, scale_strategy="min")
assert_equal(round(min_scale, 4), 2.0, "output size estimate should support min strategy")
assert_equal([min_w, min_h], [200, 100], "output size estimate min strategy should resize final resolution")
focus_scale_meta = {
    "type": "ttp_smart_tile",
    "original_size": [100, 50],
    "tiles": [
        {
            "label": "background",
            "sample_box": [0, 0, 100, 50],
            "tile_canvas_size": [100, 50],
            "tile_canvas_box": [0, 0, 100, 50],
        },
        {
            "label": "eyes",
            "sample_box": [25, 10, 10, 10],
            "tile_canvas_size": [10, 10],
            "tile_canvas_box": [0, 0, 10, 10],
        },
    ],
}
focus_scale_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": focus_scale_meta,
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (100, 50), "white")),
        ttp.pil2tensor(Image.new("RGB", (40, 40), "white")),
    ],
}
focus_scale, focus_w, focus_h, focus_x, focus_y, focus_info = size_node.estimate_output_size(focus_scale_tile_set, scale_strategy="focus_weighted")
assert_equal(round(focus_scale, 4), 4.0, "focus-weighted output size should prefer high-detail tile scale")
assert_equal([focus_w, focus_h], [400, 200], "focus-weighted output size should report focus-detail resolution")
assert_equal(round(focus_x, 4), 4.0, "focus-weighted x scale should prefer high-detail tile scale")
assert_equal(round(focus_y, 4), 4.0, "focus-weighted y scale should prefer high-detail tile scale")
assert_equal("weight=" in focus_info, True, "focus-weighted info should report semantic weights")
done_scale, done_w, done_h, _done_x, _done_y, _done_info = size_node.estimate_output_size(mixed_scale_tile_set, done=True)
assert_equal(round(done_scale, 4), 3.0, "output size estimate should compute when done is true")
assert_equal([done_w, done_h], [300, 150], "output size estimate done gate should preserve final resolution")
aligned = ttp._ttp_align_pil_to_aspect(Image.new("RGB", (561, 401), "white"), 560, 400, "center_crop")
assert_equal(aligned.size, (560, 400), "assemble alignment should crop/resize drifted sampler output")
resized = ttp._ttp_align_pil_to_aspect(Image.new("RGB", (500, 500), "white"), 560, 400, "resize")
assert_equal(resized.size, (560, 400), "assemble resize alignment should force expected size")
save_final_node = ttp.TTP_Smart_Tile_Save_Final_Image_Experimental()
skip_result = save_final_node.save_final_image(upscaled_tile, done=False)
assert_equal(skip_result, {"ui": {"images": []}}, "save final image should skip intermediate loop frames")

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

right_bottom_mask = ttp._ttp_create_sample_blend_mask(
    432,
    328,
    {"left": 0, "right": 132, "top": 0, "bottom": 128},
    32,
)
assert_equal(float(right_bottom_mask[100, 0, 0]), 1.0, "non-overlap left edge should stay fully weighted")
assert_equal(float(right_bottom_mask[0, 100, 0]), 1.0, "non-overlap top edge should stay fully weighted")
assert_equal(float(right_bottom_mask[100, 300, 0]), 1.0, "right overlap should start at full weight")
assert_equal(float(right_bottom_mask[100, 431, 0]), 0.0, "right overlap outer edge should fade out")
assert_equal(float(right_bottom_mask[200, 100, 0]), 1.0, "bottom overlap should start at full weight")
assert_equal(float(right_bottom_mask[327, 100, 0]), 0.0, "bottom overlap outer edge should fade out")

left_top_mask = ttp._ttp_create_sample_blend_mask(
    432,
    328,
    {"left": 64, "right": 0, "top": 64, "bottom": 0},
    32,
)
assert_equal(float(left_top_mask[100, 0, 0]), 0.0, "left overlap outer edge should fade in")
assert_equal(float(left_top_mask[100, 63, 0]), 1.0, "left overlap should reach full weight at the core")
assert_equal(float(left_top_mask[0, 100, 0]), 0.0, "top overlap outer edge should fade in")
assert_equal(float(left_top_mask[63, 100, 0]), 1.0, "top overlap should reach full weight at the core")
assert_equal(float(left_top_mask[100, 431, 0]), 1.0, "non-overlap right edge should stay fully weighted")
assert_equal(float(left_top_mask[327, 100, 0]), 1.0, "non-overlap bottom edge should stay fully weighted")

auto_layout = ttp._ttp_boxes_to_auto_layout(
    [{"x": 250, "y": 120, "width": 180, "height": 260, "label": "person", "score": 0.91}],
    900,
    600,
    default_pad=64,
    default_blend=32,
    object_padding=40,
    max_tiles=4,
    include_background=True,
    allow_object_overlap=True,
)
auto_meta = ttp._ttp_parse_smart_tile_layout(auto_layout, 900, 600)
assert_equal(len(auto_meta), 2, "auto layout should include background and one object tile")
object_tile = auto_meta[1]
ox, oy, ow, oh = object_tile["core_box"]
assert_equal(ox <= 250 and oy <= 120 and ox + ow >= 430 and oy + oh >= 380, True, "auto object tile must contain the source bbox")
assert_equal(object_tile["source"], "auto", "auto layout should preserve source metadata")
assert_equal(object_tile["label"], "person", "auto layout should preserve labels")
assert_equal(object_tile["layer"], 2, "auto object tile should be above background")
assert_equal(object_tile["occlusion_priority"] > auto_meta[0]["occlusion_priority"], True, "object tile should outrank background")

detail_layout = ttp._ttp_boxes_to_auto_layout(
    [
        {"x": 220, "y": 80, "width": 300, "height": 440, "label": "person", "score": 0.95},
        {"x": 315, "y": 120, "width": 92, "height": 86, "label": "face", "score": 0.88},
    ],
    900,
    600,
    default_pad=64,
    default_blend=32,
    object_padding=24,
    max_tiles=4,
    include_background=True,
    allow_object_overlap=True,
)
detail_meta = ttp._ttp_parse_smart_tile_layout(detail_layout, 900, 600)
person_tile = next(tile for tile in detail_meta if tile.get("label") == "person")
face_tile = next(tile for tile in detail_meta if tile.get("label") == "face")
assert_equal(face_tile["layer"] > person_tile["layer"], True, "face/detail auto tile should sit above generic object tile")
assert_equal(face_tile["occlusion_priority"] > person_tile["occlusion_priority"], True, "face/detail auto tile should outrank generic object tile")

mask_image = Image.new("L", (900, 600), 0)
for x in range(260, 420):
    for y in range(140, 360):
        mask_image.putpixel((x, y), 255)
masked_auto_layout = ttp._ttp_boxes_to_auto_layout(
    [{"x": 250, "y": 120, "width": 180, "height": 260, "label": "person", "score": 0.91}],
    900,
    600,
    default_pad=64,
    default_blend=32,
    object_padding=40,
    max_tiles=4,
    include_background=False,
    allow_object_overlap=True,
    masks=[mask_image],
)
masked_auto_meta = ttp._ttp_parse_smart_tile_layout(masked_auto_layout, 900, 600)
assert_equal("object_mask" in masked_auto_meta[0], True, "auto layout should preserve SAM object masks")
mask_array = ttp._ttp_tile_object_mask_array(masked_auto_meta[0], masked_auto_meta[0]["sample_box"][2], masked_auto_meta[0]["sample_box"][3], "mask_feather", 32)
assert_equal(mask_array.shape[2], 1, "decoded object mask should be a single-channel weight map")
assert_equal(float(mask_array.max()) > 0.5, True, "decoded object mask should keep foreground weight")

large_body_mask = Image.new("L", (900, 600), 255)
large_body_layout = ttp._ttp_boxes_to_auto_layout(
    [{"x": 0, "y": 0, "width": 900, "height": 600, "label": "person body", "score": 0.93}],
    900,
    600,
    default_pad=64,
    default_blend=32,
    object_padding=0,
    max_tiles=4,
    include_background=False,
    allow_object_overlap=True,
    masks=[large_body_mask],
)
large_body_meta = ttp._ttp_parse_smart_tile_layout(large_body_layout, 900, 600)
large_body_tile = large_body_meta[0]
assert_equal("large context" in large_body_tile["label"], True, "huge non-detail auto body tiles should be marked as large context")
assert_equal(large_body_tile["layer"], 0, "huge non-detail auto body tiles should stay on the lowest layer")
assert_equal(large_body_tile["occlusion_priority"], 0, "huge non-detail auto body tiles should not outrank face/detail masks")
assert_equal(float(large_body_tile["priority"]) <= 10.0, True, "huge non-detail auto body tiles should keep low composite priority")
assert_equal(float(large_body_tile["importance"]) <= 0.35, True, "huge non-detail auto body tiles should remain a low-weight context layer")
assert_equal("object_mask" in large_body_tile, True, "huge body context tiles should still preserve their body mask")
assert_equal(large_body_tile["recommended_composite_mode"], "context", "huge body context tiles should advertise context compositing")

semantic_mask_layout = json.dumps({
    "tiles": [{
        "name": "person_grid_1",
        "x0": 0.2,
        "y0": 0.2,
        "x1": 0.5,
        "y1": 0.6,
        "label": "person",
        "semantic_category": "subject",
        "recommended_scale_weight": 1.75,
        "recommended_composite_mode": "replace",
        "recommended_occlusion_priority": 6200,
        "object_mask": masked_auto_meta[0]["object_mask"],
        "object_mask_source": masked_auto_meta[0]["object_mask"],
    }],
}, separators=(",", ":"))
semantic_mask_meta = ttp._ttp_parse_smart_tile_layout(semantic_mask_layout, 900, 600)
assert_equal(semantic_mask_meta[0]["semantic_category"], "subject", "interactive parser should preserve inherited semantic category")
assert_equal(round(semantic_mask_meta[0]["recommended_scale_weight"], 2), 1.75, "interactive parser should preserve inherited scale weight")
assert_equal(semantic_mask_meta[0]["recommended_composite_mode"], "replace", "interactive parser should preserve inherited composite mode")
assert_equal("object_mask" in semantic_mask_meta[0], True, "interactive parser should preserve inherited cropped object mask")
assert_equal("object_mask_source" in semantic_mask_meta[0], True, "interactive parser should preserve parent object mask source for refresh")

paint_mask = Image.new("L", (64, 48), 0)
for x in range(8, 22):
    for y in range(6, 20):
        paint_mask.putpixel((x, y), 255)
for x in range(42, 56):
    for y in range(28, 42):
        paint_mask.putpixel((x, y), 255)
paint_buffer = BytesIO()
paint_mask.save(paint_buffer, format="PNG")
paint_payload = json.dumps({
    "format": "png_base64",
    "width": 64,
    "height": 48,
    "data": base64.b64encode(paint_buffer.getvalue()).decode("ascii"),
})
decoded_paint = ttp._ttp_decode_interactive_paint_mask(paint_payload, 64, 48)
paint_items, paint_masks = ttp._ttp_paint_mask_to_items(decoded_paint, 64, 48)
assert_equal(len(paint_items), 2, "paint mask should become one bbox per painted island")
paint_layout, paint_message = ttp._ttp_run_paint_mask_auto_layout(
    Image.new("RGB", (64, 48), "black"),
    paint_payload,
    default_pad=8,
    default_blend=4,
    object_padding=2,
    max_tiles=8,
    allow_object_overlap=True,
)
paint_meta = ttp._ttp_parse_smart_tile_layout(paint_layout, 64, 48)
assert_equal(len(paint_meta), 2, "paint-only auto tile should create tiles from painted regions")
assert_equal(all(tile.get("object_mask") for tile in paint_meta), True, "paint-created tiles should carry object masks")
assert_equal("paint mask" in paint_message.lower(), True, "paint-only auto tile should report paint mask inference")

half_mask = Image.new("L", (8, 8), 128)
half_mask_data = ttp._ttp_encode_object_mask_data([half_mask], [0], [0, 0, 8, 8])
assemble_node = ttp.TTP_Smart_Tile_Assemble_Experimental()
soft_tile_meta = {
    "type": "ttp_smart_tile",
    "original_size": [8, 8],
    "tiles": [{
        "name": "soft_object",
        "core_box": [0, 0, 8, 8],
        "sample_box": [0, 0, 8, 8],
        "tile_canvas_size": [8, 8],
        "tile_canvas_box": [0, 0, 8, 8],
        "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "blend": 0,
        "importance": 1.0,
        "priority": 0.0,
        "layer": 2,
        "occlusion_priority": 10,
        "object_mask": half_mask_data,
    }],
}
soft_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": soft_tile_meta,
    "tile_images": [ttp.pil2tensor(Image.new("RGB", (8, 8), "white"))[0]],
}
soft_output, _soft_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    mask_blend_mode="mask_only",
    tile_set=soft_tile_set,
    base_image=ttp.pil2tensor(Image.new("RGB", (8, 8), "black")),
)
soft_value = float(soft_output.array[0, 4, 4, 0])
assert_equal(0.45 < soft_value < 0.55, True, "occlusion mask feather should alpha blend with lower pixels instead of hard replacing them")

blue_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": soft_tile_meta,
    "tile_images": [ttp.pil2tensor(Image.new("RGB", (8, 8), (32, 64, 192)))[0]],
}
red_reference = ttp.pil2tensor(Image.new("RGB", (8, 8), (192, 64, 32)))
color_matched, _color_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    color_correction="local_mean_std",
    color_strength=1.0,
    mask_blend_mode="mask_only",
    tile_set=blue_tile_set,
    base_image=red_reference,
    color_reference_image=red_reference,
)
matched_pixel = color_matched.array[0, 4, 4]
assert_equal(float(matched_pixel[0]) > float(matched_pixel[2]), True, "local mean/std color correction should match tile color toward the local reference")

black_canvas, _black_canvas_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    mask_blend_mode="mask_only",
    base_canvas_mode="black",
    tile_set=blue_tile_set,
    base_image=red_reference,
)
black_canvas_pixel = black_canvas.array[0, 4, 4]
assert_equal(float(black_canvas_pixel[2]) > float(black_canvas_pixel[0]), True, "black canvas mode should not mix the connected base image into semi-transparent tiles")

black_canvas_color_matched, _black_canvas_color_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    color_correction="local_mean_std",
    color_strength=1.0,
    mask_blend_mode="mask_only",
    base_canvas_mode="black",
    tile_set=blue_tile_set,
    base_image=red_reference,
)
black_canvas_color_pixel = black_canvas_color_matched.array[0, 4, 4]
assert_equal(float(black_canvas_color_pixel[0]) > float(black_canvas_color_pixel[2]), True, "black canvas mode should still allow base_image color reference")

stack_meta = {
    "type": "ttp_smart_tile",
    "original_size": [8, 8],
    "tiles": [
        {
            "name": "large_body",
            "label": "person body",
            "core_box": [0, 0, 8, 8],
            "sample_box": [0, 0, 8, 8],
            "tile_canvas_size": [8, 8],
            "tile_canvas_box": [0, 0, 8, 8],
            "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "blend": 0,
            "importance": 1.0,
            "priority": 100.0,
            "layer": 2,
            "occlusion_priority": 1000,
        },
        {
            "name": "small_face",
            "label": "local detail",
            "core_box": [2, 2, 4, 4],
            "sample_box": [2, 2, 4, 4],
            "tile_canvas_size": [4, 4],
            "tile_canvas_box": [0, 0, 4, 4],
            "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "blend": 0,
            "importance": 1.0,
            "priority": 0.0,
            "layer": 0,
            "occlusion_priority": 0,
        },
    ],
}
stack_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": stack_meta,
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (8, 8), (220, 20, 20)))[0],
        ttp.pil2tensor(Image.new("RGB", (4, 4), (20, 20, 220)))[0],
    ],
}
large_above, _large_above_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    auto_composite_policy="strict_layer",
    tile_set=stack_tile_set,
)
large_above_pixel = large_above.array[0, 4, 4]
assert_equal(float(large_above_pixel[0]) > float(large_above_pixel[2]), True, "explicit large tile priority should cover a small tile when small_tile_on_top is disabled")
small_above, _small_above_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    small_tile_on_top=True,
    auto_composite_policy="strict_layer",
    tile_set=stack_tile_set,
)
small_above_pixel = small_above.array[0, 4, 4]
assert_equal(float(small_above_pixel[2]) > float(small_above_pixel[0]), True, "small_tile_on_top should stack smaller tiles above larger context tiles")

legacy_bad_bool, _legacy_bad_bool_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    small_tile_on_top="safe_auto",
    auto_composite_policy="strict_layer",
    tile_set=stack_tile_set,
)
legacy_bad_bool_pixel = legacy_bad_bool.array[0, 4, 4]
assert_equal(float(legacy_bad_bool_pixel[0]) > float(legacy_bad_bool_pixel[2]), True, "legacy string values should not be treated as small_tile_on_top=true")

fingerprint_mask_a = Image.new("L", (8, 8), 0)
ImageDraw.Draw(fingerprint_mask_a).rectangle((0, 0, 3, 7), fill=255)
fingerprint_mask_b = Image.new("L", (8, 8), 0)
ImageDraw.Draw(fingerprint_mask_b).rectangle((4, 0, 7, 7), fill=255)
fingerprint_source_a = ttp._ttp_encode_object_mask_data([fingerprint_mask_a], [0], [0, 0, 8, 8])
fingerprint_source_b = ttp._ttp_encode_object_mask_data([fingerprint_mask_b], [0], [0, 0, 8, 8])
fingerprint_tile = {
    "name": "context_grid_1",
    "label": "large context",
    "core_box": [0, 0, 8, 8],
    "sample_box": [0, 0, 8, 8],
    "object_mask": fingerprint_source_a,
    "object_mask_source": fingerprint_source_a,
    "layer": 0,
    "occlusion_priority": 0,
}
fingerprint_tile_set_a = {
    "type": "ttp_smart_tile_set",
    "original_size": [8, 8],
    "tile_meta": {"type": "ttp_smart_tile", "original_size": [8, 8], "tiles": [fingerprint_tile]},
    "tile_images": [None],
}
fingerprint_tile_b = dict(fingerprint_tile)
fingerprint_tile_b["object_mask_source"] = fingerprint_source_b
fingerprint_tile_set_b = {
    "type": "ttp_smart_tile_set",
    "original_size": [8, 8],
    "tile_meta": {"type": "ttp_smart_tile", "original_size": [8, 8], "tiles": [fingerprint_tile_b]},
    "tile_images": [None],
}
assert_equal(
    ttp._ttp_tile_set_fingerprint(fingerprint_tile_set_a) != ttp._ttp_tile_set_fingerprint(fingerprint_tile_set_b),
    True,
    "loop fingerprint should change when object_mask_source changes",
)

feather_stack_meta = {
    "type": "ttp_smart_tile",
    "original_size": [16, 16],
    "tiles": [
        {
            "name": "background",
            "label": "background",
            "core_box": [0, 0, 16, 16],
            "sample_box": [0, 0, 16, 16],
            "tile_canvas_size": [16, 16],
            "tile_canvas_box": [0, 0, 16, 16],
            "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "blend": 0,
            "importance": 1.0,
            "priority": 0.0,
            "layer": 0,
            "occlusion_priority": 0,
        },
        {
            "name": "face_detail",
            "label": "face detail",
            "core_box": [4, 4, 8, 8],
            "sample_box": [4, 4, 8, 8],
            "tile_canvas_size": [8, 8],
            "tile_canvas_box": [0, 0, 8, 8],
            "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "blend": 4,
            "importance": 1.0,
            "priority": 0.0,
            "layer": 2,
            "occlusion_priority": 1000,
        },
    ],
}
feather_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": feather_stack_meta,
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (16, 16), (220, 20, 20)))[0],
        ttp.pil2tensor(Image.new("RGB", (8, 8), (20, 20, 220)))[0],
    ],
}
feathered_stack, _feathered_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    tile_set=feather_tile_set,
)
feather_edge_pixel = feathered_stack.array[0, 4, 8]
feather_center_pixel = feathered_stack.array[0, 8, 8]
assert_equal(float(feather_edge_pixel[0]) > float(feather_edge_pixel[2]), True, "focus tile fallback feather should keep lower pixels visible at the rectangle edge")
assert_equal(float(feather_center_pixel[2]) > float(feather_center_pixel[0]), True, "focus tile fallback feather should keep the focus tile strong in the center")

eye_mask_full = Image.new("L", (16, 16), 0)
eye_mask_tile = Image.new("L", (8, 8), 128)
ImageDraw.Draw(eye_mask_tile).rectangle((2, 2, 5, 5), fill=255)
eye_mask_full.paste(eye_mask_tile, (4, 4))
eye_mask_data = ttp._ttp_encode_object_mask_data([eye_mask_full], [0], [4, 4, 12, 12])
eye_stack_meta = {
    "type": "ttp_smart_tile",
    "original_size": [16, 16],
    "tiles": [
        {
            "name": "face_base",
            "label": "face",
            "core_box": [0, 0, 16, 16],
            "sample_box": [0, 0, 16, 16],
            "tile_canvas_size": [16, 16],
            "tile_canvas_box": [0, 0, 16, 16],
            "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "blend": 0,
            "importance": 1.0,
            "priority": 0.0,
            "layer": 1,
            "occlusion_priority": 100,
        },
        {
            "name": "eyes_detail",
            "label": "eyes glasses",
            "core_box": [4, 4, 8, 8],
            "sample_box": [4, 4, 8, 8],
            "tile_canvas_size": [8, 8],
            "tile_canvas_box": [0, 0, 8, 8],
            "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "blend": 0,
            "importance": 1.0,
            "priority": 0.0,
            "layer": 4,
            "occlusion_priority": 2000,
            "object_mask": eye_mask_data,
        },
    ],
}
eye_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": eye_stack_meta,
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (16, 16), (220, 20, 20)))[0],
        ttp.pil2tensor(Image.new("RGB", (8, 8), (20, 20, 220)))[0],
    ],
}
strict_eye, _strict_eye_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    auto_composite_policy="strict_layer",
    tile_set=eye_tile_set,
)
safe_eye, _safe_eye_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    auto_composite_policy="safe_auto",
    tile_set=eye_tile_set,
)
strict_eye_edge = strict_eye.array[0, 4, 8]
safe_eye_edge = safe_eye.array[0, 4, 8]
safe_eye_center = safe_eye.array[0, 8, 8]
assert_equal(float(safe_eye_edge[0]) > float(strict_eye_edge[0]), True, "safe auto detail overlay should preserve more face color at eye mask edges")
assert_equal(float(safe_eye_center[2]) > float(safe_eye_center[0]), True, "safe auto detail overlay should keep eye details stronger at the mask center")

subdivided_mask_full = Image.new("L", (64, 64), 0)
ImageDraw.Draw(subdivided_mask_full).rectangle((24, 24, 39, 39), fill=255)
subdivided_mask_data = ttp._ttp_encode_object_mask_data([subdivided_mask_full], [0], [24, 24, 40, 40])
subdivided_detail_tile = {
    "name": "object_5_second_grid_child",
    "label": "object_5",
    "core_box": [24, 24, 16, 16],
    "sample_box": [8, 8, 48, 48],
    "tile_canvas_size": [48, 48],
    "tile_canvas_box": [0, 0, 48, 48],
    "overlap_edges_px_source": {"left": 16, "right": 16, "top": 16, "bottom": 16},
    "blend": 0,
    "importance": 1.0,
    "priority": 0.0,
    "layer": 4,
    "occlusion_priority": 2000,
    "object_mask": subdivided_mask_data,
    "object_mask_source": subdivided_mask_data,
}
subdivided_meta = {
    "type": "ttp_smart_tile",
    "original_size": [64, 64],
    "tiles": [
        {
            "name": "face_base",
            "label": "face",
            "core_box": [0, 0, 64, 64],
            "sample_box": [0, 0, 64, 64],
            "tile_canvas_size": [64, 64],
            "tile_canvas_box": [0, 0, 64, 64],
            "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
            "blend": 0,
            "importance": 1.0,
            "priority": 0.0,
            "layer": 1,
            "occlusion_priority": 100,
        },
        subdivided_detail_tile,
    ],
}
subdivided_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": subdivided_meta,
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (64, 64), (220, 20, 20)))[0],
        ttp.pil2tensor(Image.new("RGB", (48, 48), (20, 20, 220)))[0],
    ],
}
assert_equal(ttp._ttp_tile_area_ratio(subdivided_detail_tile, 64 * 64) < 0.07, True, "subdivided detail ranking should use core area instead of padded sample area")
subdivided_mask_array = ttp._ttp_tile_object_mask_array(subdivided_detail_tile, 48, 48, "mask_feather", 0)
assert_equal(0.0 < float(subdivided_mask_array[16, 15, 0]) < 1.0, True, "mask_feather should soften subdivided object masks even when blend is zero")
strict_subdivided, _strict_subdivided_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    auto_composite_policy="strict_layer",
    tile_set=subdivided_tile_set,
)
safe_subdivided, _safe_subdivided_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    auto_composite_policy="safe_auto",
    tile_set=subdivided_tile_set,
)
strict_subdivided_edge = strict_subdivided.array[0, 32, 24]
safe_subdivided_edge = safe_subdivided.array[0, 32, 24]
safe_subdivided_center = safe_subdivided.array[0, 32, 32]
assert_equal(float(safe_subdivided_edge[0]) > float(strict_subdivided_edge[0]), True, "safe auto should not let a repeatedly subdivided masked tile cut a hard hole in the lower face tile")
assert_equal(float(safe_subdivided_center[2]) > float(safe_subdivided_center[0]), True, "safe auto should still keep the subdivided detail strong at the mask center")

split_body_mask = Image.new("L", (64, 32), 255)
split_body_source_mask = ttp._ttp_encode_object_mask_data([split_body_mask], [0], [0, 0, 64, 32])
split_body_left_mask = ttp._ttp_encode_object_mask_data([split_body_mask], [0], [0, 0, 32, 32])
split_body_right_mask = ttp._ttp_encode_object_mask_data([split_body_mask], [0], [32, 0, 64, 32])
split_body_left_tile = {
    "name": "body_context_grid_1",
    "label": "person body large context",
    "source": "auto_context",
    "core_box": [0, 0, 32, 32],
    "sample_box": [0, 0, 40, 32],
    "tile_canvas_size": [40, 32],
    "tile_canvas_box": [0, 0, 40, 32],
    "overlap_edges_px_source": {"left": 0, "right": 8, "top": 0, "bottom": 0},
    "blend": 8,
    "importance": 0.35,
    "priority": 10.0,
    "layer": 0,
    "occlusion_priority": 0,
    "recommended_composite_mode": "context",
    "object_mask": split_body_left_mask,
    "object_mask_source": split_body_source_mask,
}
split_body_right_tile = {
    "name": "body_context_grid_2",
    "label": "person body large context",
    "source": "auto_context",
    "core_box": [32, 0, 32, 32],
    "sample_box": [24, 0, 40, 32],
    "tile_canvas_size": [40, 32],
    "tile_canvas_box": [0, 0, 40, 32],
    "overlap_edges_px_source": {"left": 8, "right": 0, "top": 0, "bottom": 0},
    "blend": 8,
    "importance": 0.35,
    "priority": 10.0,
    "layer": 0,
    "occlusion_priority": 0,
    "recommended_composite_mode": "context",
    "object_mask": split_body_right_mask,
    "object_mask_source": split_body_source_mask,
}
split_source_mask_array = ttp._ttp_tile_object_mask_array(split_body_left_tile, 40, 32, "mask_feather", 8)
assert_equal(float(split_source_mask_array[16, 36, 0]) > 0.5, True, "subdivided context masks should use source masks so overlap regions stay available")
split_body_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": {
        "type": "ttp_smart_tile",
        "original_size": [64, 32],
        "tiles": [split_body_left_tile, split_body_right_tile],
    },
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (40, 32), (90, 90, 90)))[0],
        ttp.pil2tensor(Image.new("RGB", (40, 32), (90, 90, 90)))[0],
    ],
}
_split_body_output, split_body_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    base_canvas_mode="black",
    mask_blend_mode="mask_feather",
    auto_composite_policy="safe_auto",
    tile_set=split_body_tile_set,
)
split_body_single_weight = float(split_body_weights.array[0, 16, 16, 0])
split_body_seam_weight = float(split_body_weights.array[0, 16, 32, 0])
assert_equal(split_body_seam_weight > split_body_single_weight + 0.2, True, "subdivided context mask seams should show real overlap blending")

fractional_scale_tile_set = {
    "type": "ttp_smart_tile_set",
    "tile_meta": {
        "type": "ttp_smart_tile",
        "original_size": [3, 1],
        "tiles": [
            {
                "name": f"column_{index}",
                "label": "grid",
                "core_box": [index, 0, 1, 1],
                "sample_box": [index, 0, 1, 1],
                "tile_canvas_size": [1, 1],
                "tile_canvas_box": [0, 0, 1, 1],
                "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
                "blend": 0,
                "importance": 1.0,
                "priority": 0.0,
            }
            for index in range(3)
        ],
    },
    "tile_images": [
        ttp.pil2tensor(Image.new("RGB", (1, 1), (index * 80, index * 80, index * 80)))[0]
        for index in range(3)
    ],
}
_fractional_output, fractional_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.4,
    use_priority=False,
    base_canvas_mode="black",
    weight_preview_mode="coverage",
    tile_set=fractional_scale_tile_set,
)
assert_equal(list(fractional_weights.shape), [1, 1, 4, 3], "fractional output scale should round the full canvas size")
assert_equal(float(fractional_weights.array.min()), 1.0, "fractional output scale should not leave rounded tile boundary gaps")
fractional_edges = ttp._ttp_scaled_overlap_edges([1, 0, 3, 1], {"left": 1, "right": 0, "top": 0, "bottom": 0}, 1.4, 5, 1)
assert_equal(fractional_edges["left"], 2, "fractional overlap edges should use source endpoints instead of independent length rounding")

deferred_output, deferred_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    assemble_mode="final_only",
    done=False,
    tile_set=blue_tile_set,
    base_image=red_reference,
)
deferred_pixel = deferred_output.array[0, 4, 4]
assert_equal(float(deferred_pixel[0]) > float(deferred_pixel[2]), True, "final-only assemble should return the base preview before loop completion")
assert_equal(float(deferred_weights.array.max()), 0.0, "final-only assemble should skip weight drawing before loop completion")

forced_final_output, forced_final_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=1.0,
    use_priority=True,
    assemble_mode="always",
    pixel_alignment="edge_match",
    done=False,
    tile_set=blue_tile_set,
    base_image=red_reference,
)
forced_pixel = forced_final_output.array[0, 4, 4]
assert_equal(float(forced_pixel[0]) > float(forced_pixel[2]), True, "pixel alignment should force final-only preview during unfinished loops")
assert_equal(float(forced_final_weights.array.max()), 0.0, "pixel alignment forced final-only mode should skip intermediate weights")

base_scale_meta = {
    "type": "ttp_smart_tile",
    "original_size": [8, 8],
    "tiles": [{
        "name": "large_context",
        "label": "large context tile",
        "core_box": [0, 0, 8, 8],
        "sample_box": [0, 0, 8, 8],
        "tile_canvas_size": [8, 8],
        "tile_canvas_box": [0, 0, 8, 8],
        "overlap_edges_px_source": {"left": 0, "right": 0, "top": 0, "bottom": 0},
        "blend": 0,
        "importance": 1.0,
        "priority": 100.0,
        "layer": 2,
        "occlusion_priority": 100,
    }],
}
base_canvas = ttp.pil2tensor(Image.new("RGB", (16, 16), (200, 20, 20)))
low_res_context = {
    "type": "ttp_smart_tile_set",
    "tile_meta": base_scale_meta,
    "tile_images": [ttp.pil2tensor(Image.new("RGB", (8, 8), (20, 20, 220)))[0]],
}
kept_base, _kept_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=0.0,
    use_priority=True,
    tile_set=low_res_context,
    base_image=base_canvas,
    large_tile_policy="use_if_higher_resolution",
    context_tile_weight=0.0,
)
assert_equal(list(kept_base.shape), [1, 16, 16, 3], "base_image should set the inferred assemble output size")
kept_pixel = kept_base.array[0, 8, 8]
assert_equal(float(kept_pixel[0]) > 0.75 and float(kept_pixel[2]) < 0.1, True, "low-resolution large tiles should not replace a higher-resolution base canvas")

high_res_context = {
    "type": "ttp_smart_tile_set",
    "tile_meta": base_scale_meta,
    "tile_images": [ttp.pil2tensor(Image.new("RGB", (16, 16), (20, 20, 220)))[0]],
}
used_high_res, _used_weights = assemble_node.assemble_tiles(
    blend_multiplier=1.0,
    output_scale=0.0,
    use_priority=True,
    tile_set=high_res_context,
    base_image=base_canvas,
    large_tile_policy="use_if_higher_resolution",
    context_tile_weight=0.0,
)
used_pixel = used_high_res.array[0, 8, 8]
assert_equal(float(used_pixel[2]) > 0.75 and float(used_pixel[0]) < 0.1, True, "higher-resolution large tiles should be allowed to replace the base canvas")

reference = np.zeros((16, 16, 3), dtype=np.float32)
for y in range(16):
    for x in range(16):
        reference[y, x, :] = [(x + y) / 30.0, x / 15.0, y / 15.0]
shifted_region = reference[5:13, 6:14].copy()
alignment_mask = np.ones((8, 8, 1), dtype=np.float32)
alignment_weights = np.ones((16, 16, 1), dtype=np.float32)
dx, dy = ttp._ttp_find_pixel_alignment_offset(
    shifted_region,
    reference,
    alignment_weights,
    4,
    4,
    alignment_mask,
    radius=4,
    mode="mask_edge_match",
)
assert_equal((dx, dy), (2, 1), "pixel alignment should find the best local offset from surrounding pixels")
auto_dx, auto_dy, auto_info = ttp._ttp_find_pixel_alignment_offset_auto(
    shifted_region,
    reference,
    alignment_weights,
    4,
    4,
    alignment_mask,
    radius=4,
    mode="mask_edge_match",
    device_mode="cpu",
)
assert_equal((auto_dx, auto_dy), (2, 1), "pixel alignment auto wrapper should preserve the CPU result when forced to CPU")
assert_equal(auto_info["device"], "cpu", "pixel alignment info should report forced CPU execution")

fallback_layout = ttp._ttp_boxes_to_auto_layout([], 900, 600, max_tiles=4, include_background=False)
fallback_meta = ttp._ttp_parse_smart_tile_layout(fallback_layout, 900, 600)
assert_equal(len(fallback_meta), 4, "auto fallback should produce a 2x2 layout")

print("smoke_smart_tile_grid_crop: ok")
