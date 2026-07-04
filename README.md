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

The **Smart Tile** nodes provide an interactive tile workflow for layouts where you want to split an image by visual structure, such as face, hands, subject, clothes, foreground, and background.

Current workflow nodes:

- `TTP Smart Tile Interactive Crop`: loads or receives a source image, provides the visual tile editor, supports manual tiles, painted masks, and SAM/QwenVL auto tile requests, and outputs a variable-size `tile_set`.
- `TTP Smart Tile Set Preview`: previews a tile set as a contact sheet or a single selected tile.
- `TTP QwenVL3 Local Loader`: loads a local QwenVL tagging model from ComfyUI models.
- `TTP Smart Tile QwenVL Prompt Set Builder`: prepares per-tile prompts before loop processing.
- `TTP Smart Tile Loop Source`: outputs one tile at a time for sampler/img2img processing.
- `TTP Smart Tile Loop Collect`: collects processed tiles back into the tile set.
- `TTP Smart Tile Image Upscale Prep`: optionally upscales one tile before sampling.
- `TTP Smart Tile Output Size Estimate`: calculates final output scale and resolution from a processed tile set.
- `TTP Smart Tile Assemble`: assembles processed tiles back into the final image with feathered blending, mask support, priority/layer handling, color correction, optional CPU/GPU pixel alignment, and optional GPU paste/weight accumulation.
- `TTP Smart Tile Save Final Image`: saves only the final loop result and embeds workflow metadata.

Interactive loop workflow:

```text
TTP Smart Tile Interactive Crop
  -> TTP Smart Tile QwenVL Prompt Set Builder (optional)
  -> TTP Smart Tile Loop Source
  -> VAE Encode / Sampler / VAE Decode
  -> TTP Smart Tile Loop Collect
  -> TTP Smart Tile Output Size Estimate (optional)
  -> TTP Smart Tile Assemble
  -> TTP Smart Tile Save Final Image
```

By default, `TTP Smart Tile Assemble` uses `assemble_mode=final_only`. Connect `TTP Smart Tile Loop Collect.done` to `TTP Smart Tile Assemble.done` so loop runs return a lightweight preview while `done` is false, then perform the full assemble once after the last tile. Switch `assemble_mode` to `always` only when you want a full recomposite after every tile; if pixel alignment is enabled, unfinished loop runs are automatically treated as `final_only` to avoid repeated expensive alignment passes. `assemble_device` controls the paste/weight accumulation device (`auto`, `cpu`, or `gpu`), and pixel alignment can use the GPU canvas when GPU assemble is active. Use `base_canvas_mode=black` when you want connected source/base images to remain available as references without being pasted underneath the tiles. Enable `small_tile_on_top` when small detail tiles should automatically stack above larger body/background/context tiles in overlap areas. `auto_composite_policy=safe_auto` keeps background/context tiles low, promotes detected details, and blends face/eye/glasses/mouth-style detail masks as soft overlays instead of cutting holes in lower face tiles; use `strict_layer` to restore the raw metadata layer behavior. Official `Transfer Color` methods and PIL resize/crop handling remain on their existing paths for compatibility.

`TTP Smart Tile Image Upscale Prep` prepares each loop tile before img2img sampling. It can use a connected ComfyUI `UPSCALE_MODEL` through the same tiled upscale-model path as the built-in upscale node, or fall back to `lanczos`, `bicubic`, `bilinear`, `area`, or nearest resize when no model is connected or `use_upscale_model` is off. `scale` sets the requested enlargement, `max_megapixels` caps the final tile pixel count, and `round_to` snaps the final width/height after the cap. When the cap is active, the node rounds down so the rounded tile stays under the megapixel budget. Tile coordinates are not changed; they remain in original-image space so assemble can map the processed tile back by `sample_box` and `output_scale`.

`TTP Smart Tile Output Size Estimate` reads the processed `tile_set` after `Loop Collect` and reports `output_scale`, final `width`/`height`, separate `scale_x`/`scale_y`, and a per-tile info log. The default `median` strategy matches Assemble's automatic tile-scale inference, and the `output_scale` output can be connected directly to `TTP Smart Tile Assemble.output_scale`. For final-only loops, connect `TTP Smart Tile Loop Collect.done` to both this node's `done` input and `TTP Smart Tile Assemble.done`; while `done=false`, this node returns a deferred zero-scale placeholder and skips tile scanning, then estimates once when `done=true`. Mixed tile scales are reported in the info string so capped or unevenly enlarged tiles are visible before final assembly.

`TTP Smart Tile Interactive Crop` is the recommended starting point when you want to manually or automatically split a still image by visual regions. Its `image` input follows the official `Load Image` pattern, so uploads go to ComfyUI's input folder and the workflow stores the selected filename instead of embedding the whole image. The editor can generate a standard grid from column/row numbers, replace the full layout with that grid, subdivide the currently selected tile, add painted-mask tiles, and fill uncovered gaps. It stores the tile layout in a hidden widget so the workflow keeps the current plan.

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
