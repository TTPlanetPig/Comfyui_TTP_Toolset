import json
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image

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


install_comfy_stubs()

import TTP_toolsets as ttp  # noqa: E402


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
assert_equal("vision_model" in optional_inputs, True, "interactive crop should expose a vision model input")
assert_equal("vision_conditioning" in optional_inputs, True, "interactive crop should expose official SAM3 conditioning input")
assert_equal("clip" in optional_inputs, True, "interactive crop should optionally encode SAM3 prompts with CLIP")
assert_equal(hidden_inputs["unique_id"], "UNIQUE_ID", "interactive crop should receive its ComfyUI node id")

assemble_inputs = ttp.TTP_Smart_Tile_Assemble_Experimental.INPUT_TYPES()
assert_equal("sampled_tiles" in assemble_inputs["optional"], True, "assemble should keep batch tiles as an optional compatibility input")
assert_equal("tile_meta" in assemble_inputs["optional"], True, "assemble should keep batch metadata as an optional compatibility input")
assert_equal("tile_set" in assemble_inputs["optional"], True, "assemble should accept true variable-size tile sets")
assert_equal("local_mean_std" in assemble_inputs["required"]["color_correction"][0], True, "assemble should expose local mean/std color correction")
assert_equal("tile_alignment" in assemble_inputs["required"], True, "assemble should expose tile alignment")
assert_equal("edge_crop_px" in assemble_inputs["required"], True, "assemble should expose edge crop control")
assert_equal("mask_blend_mode" in assemble_inputs["required"], True, "assemble should expose object mask blending")
assert_equal("pixel_alignment" in assemble_inputs["required"], True, "assemble should expose pixel alignment")
assert_equal("pixel_alignment_radius" in assemble_inputs["required"], True, "assemble should expose pixel alignment radius")
preview_inputs = ttp.TTP_Smart_Tile_Set_Preview_Experimental.INPUT_TYPES()
assert_equal(preview_inputs["required"]["tile_set"][0], "TTP_SMART_TILE_SET", "tile set preview should accept Smart Tile Set")
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Set_Preview_Experimental"],
    ttp.TTP_Smart_Tile_Set_Preview_Experimental,
    "tile set preview should be registered",
)
prompt_builder_inputs = ttp.TTP_Smart_Tile_QwenVL_Prompt_Set_Builder_Experimental.INPUT_TYPES()
assert_equal(prompt_builder_inputs["required"]["reference_image_mode"][0], ["none", "first_message", "every_tile"], "prompt builder should support both reference image strategies")
assert_equal(prompt_builder_inputs["required"]["mode"][0], ["template", "qwen_vl_api", "qwen_vl_local"], "prompt builder should expose template, API, and local QwenVL modes")
assert_equal("seed" in prompt_builder_inputs["required"], False, "prompt builder seed should not disturb existing required widget order")
assert_equal(prompt_builder_inputs["optional"]["seed"][0], "INT", "prompt builder should expose an optional seed for local QwenVL sampling")
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
assert_equal(
    ttp.NODE_CLASS_MAPPINGS["TTP_Smart_Tile_Image_Upscale_Prep_Experimental"],
    ttp.TTP_Smart_Tile_Image_Upscale_Prep_Experimental,
    "upscale prep should be registered",
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

class FakeQwenClip:
    last_seed = None
    last_text = ""

    def tokenize(self, text, **kwargs):
        self.last_text = text
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
    seed=1234,
    qwen_vl_model={
        "type": "ttp_qwenvl3_model",
        "model_file": "fake_qwen_vl.safetensors",
        "clip": fake_qwen_clip,
    },
)
qwen_prompt_data = json.loads(qwen_prompt_json)
assert_equal(qwen_prompt_data["mode"], "qwen_vl_local", "connected local QwenVL model should override template fallback")
assert_equal(qwen_prompt_data["seed"], 1234, "QwenVL prompt set JSON should record the local sampling seed")
assert_equal("mode=qwen_vl_local" in qwen_summary, True, "QwenVL summary should report the effective mode")
assert_equal(qwen_tile_set["tile_meta"]["tiles"][0]["prompt"], "qwen detailed tile 0 prompt", "QwenVL prompt should come from model output")
assert_equal(qwen_tile_set["tile_meta"]["tiles"][1]["prompt"], "qwen detailed tile 1 prompt", "QwenVL should write a distinct prompt per tile")
assert_equal(fake_qwen_clip.last_seed, 1234, "local QwenVL generate should receive an integer seed")

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
    assert_equal("did not return JSON" in str(exc), True, "QwenVL instruction echo should fail with a clear JSON error")

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

fallback_layout = ttp._ttp_boxes_to_auto_layout([], 900, 600, max_tiles=4, include_background=False)
fallback_meta = ttp._ttp_parse_smart_tile_layout(fallback_layout, 900, 600)
assert_equal(len(fallback_meta), 4, "auto fallback should produce a 2x2 layout")

print("smoke_smart_tile_grid_crop: ok")
