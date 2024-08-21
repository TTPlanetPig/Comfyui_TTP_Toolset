This is a workflow for my simple logic amazing upscale node for DIT model. it can be common use for Flux,Hunyuan,SD3 It can simple tile the initial image into pieces and then use image-interrogator to get each tile prompts for more accurate upscale process. The condition will be properly handled and the hallucination will be significantly eliminated.

#Wish you will enjoy it.

# Instruction:

# Image Tile batch node:

This one will cut image into piece by automaticly by your set width and height.
Record the necessory informaion for further use

![image](https://github.com/user-attachments/assets/9e808b33-37ff-4800-abdf-a22cce9825c1)

width: the tile image width

height: the tile image height

image: the image to be tiled

# Image Assy node:
This node will assemble the image piece into one again and provent the possible lines between the pieces of images. it works in pixel mode

![image](https://github.com/user-attachments/assets/3f9e8ba9-0c79-4984-ae8e-90b3a8ce23f1)


tiles: input the tiled image batch, you can replace one of them in the middle if you want by other batch nodes

postion: paired with image tile batch node

original size: paired with image tile batch node

grid size: paired with image tile batch node

Padding: the padding for the image to merge together.

# Tile image size node:
This node is build to decide how many pieces you want to divide by image tile batch node, it will obtain the information from the original image and caluclate the resolution for tile image:

![image](https://github.com/user-attachments/assets/b3ef38df-a620-4930-9288-d0881cfe7148)

just input the width and height factors, it will cut. the 2,3 means from width cut into 2 pieces and from height cut into 3 pieces. total 6 pieces.

# CoordinateSplitter node:
Convert the Position information into Coordinate, connect it to the positions

![image](https://github.com/user-attachments/assets/25b73335-db42-4110-8138-6af07e45a8d8)


# Cond to Batch node:
Convert the cond list into batch, I keep this node for future fuction expansion. connect it to the conditions.
![image](https://github.com/user-attachments/assets/f92a9ddc-1a98-4687-8875-03802e916dd4)

# Condition merge node:
This one will merge all the tiled condition into one piece and ready for build the image!
just connect it with CoordinateSplitter node and Cond to Batch node.

![image](https://github.com/user-attachments/assets/3039c8a3-8284-4b71-a9de-4120723258c7)


For the instant Flux example: please refer to this image with workflow in it.

it can support the controlnet Tile to enhance the upscale if you have it ready, here is an example to use tile for Hunyuan DiT
you can find the tile from my huggingface https://huggingface.co/TTPlanet

![image](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset/blob/main/examples/Flux_8Mega_Pixel_image_upscale_process.png)
and hunyuan 1.2 from here https://huggingface.co/comfyanonymous/hunyuan_dit_comfyui/blob/main/hunyuan_dit_1.2.safetensors
and here is the workflow for example:

![image](https://github.com/TTPlanetPig/Comfyui_TTP_Toolset/blob/main/examples/Hunyuan_8Mega_Pixel_image_upscale_process_with_tile_cn.png)


## Star History

<a href="https://star-history.com/#TTPlanetPig/Comfyui_TTP_Toolset&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_TTP_Toolset&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_TTP_Toolset&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=TTPlanetPig/Comfyui_TTP_Toolset&type=Date" />
 </picture>
</a>

