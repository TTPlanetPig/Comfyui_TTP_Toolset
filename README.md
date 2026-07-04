# **Amazing Upscale Node Workflow for DIT Model**

This workflow is designed for **simple logic amazing upscale nodes** in the **DIT model**. It supports common applications for **Flux**, **Hunyuan**, and **SD3**. The workflow tiles the initial image into smaller pieces, uses an image-interrogator to extract prompts for each tile, and performs an accurate upscale process. This approach minimizes hallucinations and ensures proper condition handling.

We hope you enjoy using it!

## **What's New**

### **TeaCache Sampler Integration for Hunyuan Video**

Thanks to the contributions from the TeaCache code repository ([ali-vilab/TeaCache](https://github.com/ali-vilab/TeaCache)) and code references from [facok/ComfyUI-TeaCacheHunyuanVideo](https://github.com/facok/ComfyUI-TeaCacheHunyuanVideo), we’ve added support for the **TeaCache sampler**.

- **How to Use:**
  Replace the `samplercustomadvanced` node in the official workflow with the TeaCache sampler node. Adjust the acceleration rate as needed to start using it.
  
- **Performance:**
  In testing with an NVIDIA 4090, rendering a 720×480 resolution video with 65 frames took only 55 seconds using a speedup factor of `x2.1`. This is approximately twice as fast as the original method.

- **Caution:**
  While the TeaCache sampler significantly accelerates processing, it may reduce image quality and dynamic effects. Use with discretion.

- **Precision Support:**
  Supports `bf16` and `fp8`.
  
![image](https://github.com/user-attachments/assets/9e890a64-7502-4e1f-8739-15748efc1768)


https://github.com/user-attachments/assets/af06b9d3-9c84-4a83-ba90-eb4ec4bb2e99


---

## **Instructions**

### **Smart Tile Workflow**

**Smart Tile** is an object-aware tiled img2img workflow for ComfyUI. Instead of cutting every image into equal squares, it lets you build variable-size tiles around visual structure: face, eyes, hands, text, subject, clothing, foreground details, and background/context regions.

![Smart Tile pipeline](docs/images/smart_tile_pipeline.svg)

Smart Tile is designed for workflows where small areas need more detail than the full image. You can draw or auto-detect tiles, process one tile at a time through your sampler, then assemble the final image with feathering, masks, semantic priority, color correction, and optional pixel alignment.

#### **What It Solves**

- Avoids cutting important objects through the middle when a simple grid would split a face, hand, or text.
- Allows different tile sizes, so a face tile can be small and high detail while the background stays large.
- Sends one tile at a time through img2img, so the number of tiles can be 4, 8, 16, or anything produced by the editor.
- Keeps high-detail focus tiles visible during assembly with priority, layer, and semantic ranking.
- Supports mask-aware pasteback, soft detail overlay, final-only assembly, and black-base canvas mode.

#### **Visual Tile Editor**

![Smart Tile editor controls](docs/images/smart_tile_editor_controls.svg)

`TTP Smart Tile Interactive Crop` is the recommended starting point. It can load an image like ComfyUI's official `Load Image`, or receive a connected `source_image`. The editor stores the current layout in a hidden `layout_json` widget, so your workflow keeps the tile plan.

Useful editor actions:

- `Replace grid`: replace the whole layout with an even grid.
- `Grid in T#`: subdivide the selected tile without rebuilding the whole layout.
- `Mask to Tile`: turn painted regions into object tiles.
- `Refresh masks`: after manually moving sub-tiles from a masked tile, re-crop the inherited masks to the current tile boxes.
- `Fill gaps`: add background tiles for uncovered areas.
- `Auto Tile`: run SAM3.1 or QwenVL3 detection and write the detected layout back into the editor.

When subdividing a masked tile with `Grid in`, the Mask mode can crop the original object mask into child tiles or skip empty mask children. The child tiles keep a parent mask source, so `Refresh masks` can update them after manual edits.

#### **Core Nodes**

| Node | Purpose |
|---|---|
| `TTP Smart Tile Interactive Crop` | Load/connect an image, create manual/painted/auto tiles, and output a variable-size `tile_set`. |
| `TTP Smart Tile Set Preview` | Preview a tile set as a contact sheet or one selected tile. |
| `TTP QwenVL3 Local Loader` | Load a local QwenVL tagging model from `ComfyUI/models/text_encoders`. |
| `TTP Smart Tile QwenVL Prompt Set Builder` | Generate per-tile prompts before loop processing. |
| `TTP Smart Tile Semantic Rank` | Classify tiles and write recommended layer, priority, scale weight, and composite metadata. |
| `TTP Smart Tile Loop Source` | Output one tile at a time for VAE Encode / sampler / VAE Decode. |
| `TTP Smart Tile Loop Collect` | Collect each processed tile back into the same tile set. |
| `TTP Smart Tile Image Upscale Prep` | Optionally upscale or resize each tile before sampling, with a megapixel cap. |
| `TTP Smart Tile Output Size Estimate` | Estimate final output scale/resolution from processed tile sizes. |
| `TTP Smart Tile Assemble` | Paste processed tiles back with feathering, masks, color correction, priority, optional GPU paste, and optional alignment. |
| `TTP Smart Tile Save Final Image` | Save only the final completed loop result and embed workflow metadata. |

#### **Recommended Loop Workflow**

```text
TTP Smart Tile Interactive Crop
  -> TTP Smart Tile QwenVL Prompt Set Builder (optional)
  -> TTP Smart Tile Semantic Rank (optional)
  -> TTP Smart Tile Loop Source
  -> VAE Encode / Sampler / VAE Decode
  -> TTP Smart Tile Loop Collect
  -> TTP Smart Tile Semantic Rank (optional final refresh)
  -> TTP Smart Tile Output Size Estimate (optional)
  -> TTP Smart Tile Assemble
  -> TTP Smart Tile Save Final Image
```

Use `TTP Smart Tile Loop Source` and `TTP Smart Tile Loop Collect` instead of manually duplicating sampler chains. The loop source outputs the current tile image and the current tile prompt. After the sampler finishes, loop collect stores the result and advances to the next tile.

By default, `TTP Smart Tile Assemble` uses `assemble_mode=final_only`. Connect `TTP Smart Tile Loop Collect.done` to `TTP Smart Tile Assemble.done` so unfinished loop runs return a lightweight preview while `done=false`, then perform the full assemble once after the last tile. Switch `assemble_mode` to `always` only when you really want a full recomposite after every tile. If pixel alignment is enabled, unfinished loop runs are automatically treated as final-only to avoid repeated expensive alignment passes.

#### **Auto Tile: SAM3.1 or QwenVL3**

![Smart Tile automatic detection modes](docs/images/smart_tile_auto_modes.svg)

`auto_detect_mode` controls how `Auto Tile` analyzes the image:

| Mode | Required input | Notes |
|---|---|---|
| `none` | none | Auto Tile is disabled; use manual grid, painted masks, or saved layout. |
| `sam3.1` | official SAM3/SAM3.1 model to `vision_model`, plus `CLIP` or `vision_conditioning` | Uses ComfyUI's official SAM3 Detect path. The built-in `auto_prompt` is encoded internally when `clip` is connected. |
| `qwenvl3` | `TTP QwenVL3 Local Loader -> qwen_vl_model` | Uses QwenVL for bbox JSON. Model files are selected from `ComfyUI/models/text_encoders`. If Qwen returns one large full-frame box, Smart Tile splits it into useful detail tiles. |

For QwenVL3, `TTP Smart Tile Interactive Crop` does not read `.safetensors` directly. Add `TTP QwenVL3 Local Loader`, choose the QwenVL model file, then connect its `qwen_vl_model` output into Interactive Crop.

`auto_max_tiles` limits the total Auto Tile layout after automatic gap filling, so detected objects and generated background gap tiles stay within the same cap.

#### **Per-Tile Prompts**

For manual prompt workflows, connect `TTP Smart Tile Loop Source.prompt` to your text encoder path. For automatic prompt workflows, place `TTP Smart Tile QwenVL Prompt Set Builder` before the loop:

```text
Interactive Crop
  -> QwenVL Prompt Set Builder
  -> Semantic Rank
  -> Loop Source
```

The prompt builder can use a full image, every tile, or a contact sheet as QwenVL visual context. It caches results by model file, tile hash, prompts, and seed, so reruns do not need to interrogate unchanged tiles again.

#### **Upscale Prep and Size Estimate**

`TTP Smart Tile Image Upscale Prep` prepares each loop tile before img2img sampling. It can use a connected ComfyUI `UPSCALE_MODEL` through the same tiled upscale-model path as the built-in upscale node, or fall back to `lanczos`, `bicubic`, `bilinear`, `area`, or nearest resize when no model is connected or `use_upscale_model` is off. `scale` sets the requested enlargement, `max_megapixels` caps the final tile pixel count, and `round_to` snaps the final width/height after the cap. When the cap is active, the node rounds down so the rounded tile stays under the megapixel budget. Tile coordinates are not changed; they remain in original-image space so assemble can map the processed tile back by `sample_box` and `output_scale`.

`TTP Smart Tile Output Size Estimate` reads the processed `tile_set` after `Loop Collect` and reports `output_scale`, final `width`/`height`, separate `scale_x`/`scale_y`, and a per-tile info log. The default `median` strategy matches Assemble's automatic tile-scale inference, and the `output_scale` output can be connected directly to `TTP Smart Tile Assemble.output_scale`. `focus_weighted` uses semantic scale weights so low-detail full/background tiles do not drag the final-only canvas scale below high-detail face/eye/text tiles. For final-only loops, connect `TTP Smart Tile Loop Collect.done` to both this node's `done` input and `TTP Smart Tile Assemble.done`; while `done=false`, this node returns a deferred zero-scale placeholder and skips tile scanning, then estimates once when `done=true`. Mixed tile scales are reported in the info string so capped or unevenly enlarged tiles are visible before final assembly.

#### **Assembly, Masks, and Layer Priority**

![Smart Tile assembly layer policy](docs/images/smart_tile_layer_policy.svg)

`TTP Smart Tile Semantic Rank` is optional, but useful after QwenVL prompting or auto/manual tile creation. It classifies each tile as background, subject, face, eyes, hands, text, detail, or normal from existing labels/captions/prompts, then writes semantic score, scale weight, recommended layer, priority, occlusion priority, and composite mode metadata back into the tile set. With `apply_composite_rank` on, face/eyes/text/detail tiles are promoted above background/context tiles and are marked for soft overlay blending.

Assembly options worth starting with:

| Setting | Suggested value | Why |
|---|---|---|
| `assemble_mode` | `final_only` | Assemble once when the loop is complete; much faster than recompositing every step. |
| `assemble_device` | `auto` or `gpu` | Uses GPU paste/weight accumulation when available. |
| `base_canvas_mode` | `black` when you do not want source pixels underneath | Prevents the original image from showing through uncovered or low-weight areas. |
| `auto_composite_policy` | `safe_auto` | Keeps background/context low and promotes focus/detail tiles. |
| `small_tile_on_top` | `true` for detail workflows | Helps small face/eye/text tiles win overlaps against larger body/context tiles. |
| `mask_blend_mode` | `auto` or `mask_feather` | Uses object masks when present and feathered rectangular masks otherwise. |
| `color_correction` | `off` first, then `reinhard_lab`, `mkl_lab`, or `histogram` as needed | Uses ComfyUI's official Transfer Color logic; reference defaults to the source image. |

Detail tiles should blend over lower layers instead of cutting holes into them. The safe auto policy treats face, eyes, mouth, glasses, text, and similar focus regions as soft overlays when possible.

#### **Recommended Starting Parameters**

| Scenario | Recommended settings |
|---|---|
| General manual grid | `grid=3x3`, `default_pad=32`, `default_blend=32` or `64`, `round_to=8`, `include_full_image=false` |
| Portrait/detail workflow | Auto Tile + Qwen/SAM prompt for `person, face, eyes, hands, text, foreground object` |
| Variable-size detail tiles | Use `TTP Smart Tile Image Upscale Prep` before sampler, then `Output Size Estimate` before Assemble |
| Avoid original image leaking through | `base_canvas_mode=black` |
| Small details should stay visible | `small_tile_on_top=true`, `auto_composite_policy=safe_auto` |
| Expensive alignment workflow | `assemble_mode=final_only`, connect `done` from Loop Collect |
| Strict manual layout | Keep Auto Tile mode as `none` after you finish editing, so formal runs do not replace your layout |

#### **Troubleshooting**

| Symptom | What to check |
|---|---|
| Auto Tile did not change the layout | Make sure `auto_detect_mode` is `sam3.1` or `qwenvl3`, the required model input is connected, then read the editor status message. |
| QwenVL3 does not split the image | Confirm `TTP QwenVL3 Local Loader` is connected to `qwen_vl_model`. Qwen must return bbox JSON; Smart Tile accepts common `bbox`, `bbox_2d`, `box_2d`, `objects`, and `xywh` formats. |
| Running the full workflow replaces manual edits | Set `auto_detect_mode=none` after Auto Tile if you only wanted detection once, then manually edit the saved layout. |
| A manually moved mask tile still uses the old mask | Click `Refresh masks` after moving Grid-in child tiles from a masked parent. |
| Native Save Image saves multiple loop frames | Use `TTP Smart Tile Save Final Image`; loop previews are not treated as final output. |
| Face or eyes are hidden under a larger tile | Use `TTP Smart Tile Semantic Rank`, enable `small_tile_on_top`, and keep `auto_composite_policy=safe_auto`. |
| The original image shows through | Use `base_canvas_mode=black` and make sure `Fill gaps` has covered empty regions. |
| Assemble is slow | Use `assemble_mode=final_only`; avoid repeated pixel alignment during unfinished loop steps. |

#### **Advanced Layout JSON**

Example layout:

```json
{
  "defaults": {
    "pad": 128,
    "blend": 48,
    "priority": 50,
    "importance": 1.0
  },
  "tiles": [
    {
      "name": "full_image",
      "x": 0,
      "y": 0,
      "w": 1.0,
      "h": 1.0,
      "pad": 0,
      "blend": 96,
      "priority": 10,
      "importance": 0.5
    },
    {
      "name": "face",
      "x": 0.35,
      "y": 0.08,
      "w": 0.30,
      "h": 0.28,
      "pad": 192,
      "blend": 64,
      "priority": 100,
      "importance": 1.0
    }
  ]
}
```

Coordinates can be pixel values or normalized values from `0.0` to `1.0`. A rectangle whose coordinates are all in `0..1` is treated as normalized, including browser-serialized `0` and `1` edges. `pad` is seam overlap: it expands only the tile edges that touch another tile. Outer canvas edges and non-adjacent gap edges are not expanded. `blend`, `priority`, and `importance` control how the sampled tile is pasted back.

For standard grid layouts, edge tiles are expanded inward with real source pixels when needed so the ComfyUI `IMAGE` batch has a consistent size without fake outer padding. Irregular manual layouts with uncovered gaps may still need transport padding because a single `IMAGE` batch cannot contain mixed image sizes.

### **1. Image Tile Batch Node**
This node cuts an image into pieces automatically based on your specified width and height. It also records the necessary information for further processing.

| Parameter | Description                         |
|-----------|-------------------------------------|
| **Width** | The width of each tile.            |
| **Height** | The height of each tile.           |
| **Image** | The image to be tiled.             |

**Node View**:

![Image Tile Batch Node](https://github.com/user-attachments/assets/9e808b33-37ff-4800-abdf-a22cce9825c1)

---

### **2. Image Assembly Node**
This node reassembles image tiles back into a complete image while preventing visible lines between the tiles. It operates in pixel mode.

| Parameter   | Description                                                   |
|-------------|---------------------------------------------------------------|
| **Tiles**   | Input the tiled image batch. Replace individual tiles if needed. |
| **Position** | Paired with the Image Tile Batch Node.                        |
| **Original Size** | Paired with the Image Tile Batch Node.                  |
| **Grid Size** | Paired with the Image Tile Batch Node.                      |
| **Padding** | The padding value used to merge the image pieces.             |

**Node View**:

![Image Assembly Node](https://github.com/user-attachments/assets/3f9e8ba9-0c79-4984-ae8e-90b3a8ce23f1)

---

### **3. Tile Image Size Node**
This node calculates the resolution of each tile based on the original image dimensions and your specified width/height factors.

| Parameter         | Description                                                        |
|-------------------|--------------------------------------------------------------------|
| **Width Factor**  | Divides the image width into equal parts.                          |
| **Height Factor** | Divides the image height into equal parts.                         |

For example: A width factor of `2` and a height factor of `3` will divide the image into `6` equal tiles.

**Node View**:

![Tile Image Size Node](https://github.com/user-attachments/assets/b3ef38df-a620-4930-9288-d0881cfe7148)

---

### **4. Coordinate Splitter Node**
This node converts position information into coordinates and connects them to the corresponding positions.

**Node View**:

![Coordinate Splitter Node](https://github.com/user-attachments/assets/25b73335-db42-4110-8138-6af07e45a8d8)

---

### **5. Cond to Batch Node**
This node converts condition lists into batches. It is reserved for future functionality expansion and connects to the conditions.

**Node View**:

![Cond to Batch Node](https://github.com/user-attachments/assets/f92a9ddc-1a98-4687-8875-03802e916dd4)

---

### **6. Condition Merge Node**
This node merges all tiled conditions into one and prepares them for building the final image. It connects to the **Coordinate Splitter Node** and **Cond to Batch Node**.

**Node View**:

![Condition Merge Node](https://github.com/user-attachments/assets/3039c8a3-8284-4b71-a9de-4120723258c7)

---

## **Examples**

### **Pixel Example (Recommended)**

![Pixel Example Workflow](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset/blob/main/examples/Flux_8Mega_Pixel_image_upscale_process_pixel.png)

### **Latent Example**

![Latent Example Workflow](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset/blob/main/examples/Flux_8Mega_Pixel_image_upscale_process.png)

---

### **ControlNet Tile Integration**
This workflow supports **ControlNet Tile** for enhanced upscaling. Here's an example of using tiles with the **Hunyuan DIT** model:

| Resource | Link                                                                                          |
|----------|-----------------------------------------------------------------------------------------------|
| **Tile Example** | [Hugging Face Tile](https://huggingface.co/TTPlanet)                                  |
| **Hunyuan 1.2**  | [Download Hunyuan 1.2](https://huggingface.co/comfyanonymous/hunyuan_dit_comfyui/blob/main/hunyuan_dit_1.2.safetensors) |

**Workflow Example**:

![Hunyuan Example Workflow](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset/blob/main/examples/Hunyuan_8Mega_Pixel_image_upscale_process_with_tile_cn.png)

---

## **Star History**
<a href="https://star-history.com/#TTPlanetPig/Comfyui_TTP_Toolset&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_TTP_Toolset&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_TTP_Toolset&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_TTP_Toolset&type=Date" />
 </picture>
</a>
