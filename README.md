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

### **Smart Tile Experimental Workflow**

The **Smart Tile Experimental** nodes provide a manual tile workflow that keeps the existing automatic tile nodes unchanged. They are designed for layouts where you want to split the image by visual structure, such as face, hands, subject, clothes, foreground, and background.

The first experimental version uses a **larger overlap sampling area** around real tile seams and a **smaller paste area** for the final composition:

- `TTP Smart Tile Layout (Experimental)`: stores a JSON tile layout.
- `TTP Smart Tile Crop (Experimental)`: reads the input image directly, crops tile images with seam-only overlap, and outputs tile metadata plus a debug preview.
- `TTP Smart Tile Param Crop (Experimental)`: creates a parameter-based crop plan without JSON, using grid controls plus optional focus regions.
- `TTP Smart Tile Interactive Crop (Experimental)`: adds a front-end tile editor for still images. It uses the same official ComfyUI image upload/dropdown widget style as `Load Image`, can also use a connected `source_image`, and lets you drag/resize tile rectangles on top of the image.
- `TTP Smart Tile Assemble (Experimental)`: assembles sampled tiles back into the final image with feathered weighted blending, priority, and importance weights.

Interactive image workflow:

```text
TTP Smart Tile Interactive Crop (Experimental)
  → upload/select image with the official image widget or connect source_image
  → drag/resize/add/delete/fill tile rectangles in the node
  → VAE Encode / Sampler / VAE Decode
  → TTP Smart Tile Assemble (Experimental)
  → Final Image
```

Parameter workflow without JSON:

```text
Load Image
  → TTP Smart Tile Param Crop (Experimental)
  → Preview the plan output and tune grid/focus parameters
  → VAE Encode / Sampler / VAE Decode
  → TTP Smart Tile Assemble (Experimental)
  → Final Image
```

`TTP Smart Tile Interactive Crop (Experimental)` is the recommended starting point when you want to manually split a still image by visual regions. Its `image` input follows the official `Load Image` pattern, so uploads go to ComfyUI's input folder and the workflow stores the selected filename instead of embedding the whole image. The editor can generate a standard grid from column/row numbers, replace the full layout with that grid, or subdivide the currently selected tile. It stores the tile layout in a hidden widget so the workflow keeps the current plan.

`TTP Smart Tile Param Crop (Experimental)` is useful when you want a repeatable grid/focus layout from numeric controls. It provides controls for grid columns/rows, grid overlap/blending, optional full-image coverage, and two optional focus boxes. Connect its `preview` output to `PreviewImage` while adjusting parameters.

JSON workflow:

```text
Load Image
  → TTP Smart Tile Layout (Experimental)
  → TTP Smart Tile Crop (Experimental)
  → VAE Encode / Sampler / VAE Decode
  → TTP Smart Tile Assemble (Experimental)
  → Final Image
```

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
