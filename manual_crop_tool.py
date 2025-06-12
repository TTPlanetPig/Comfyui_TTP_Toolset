import argparse
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def split_box(box, tile_w, tile_h):
    x1, y1, x2, y2 = box
    boxes = []
    for top in range(y1, y2, tile_h):
        for left in range(x1, x2, tile_w):
            boxes.append([
                left,
                top,
                min(left + tile_w, x2),
                min(top + tile_h, y2),
            ])
    return boxes


def crop_tiles(img, boxes, out_dir):
    tiles = []
    for idx, b in enumerate(boxes):
        tile = img.crop(b)
        tiles.append(tile)
        tile.save(os.path.join(out_dir, f"tile_{idx}.png"))
    return tiles


def main():
    parser = argparse.ArgumentParser(description="Draw rectangles on an image and export tiles")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("output", help="Directory to save tiles")
    parser.add_argument("--tile-size", type=int, default=512, help="Max tile size when splitting selections")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    img = Image.open(args.image).convert("RGB")

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Drag rectangles, press 'q' when done")

    boxes = []

    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red'))
        boxes.append([x1, y1, x2, y2])
        fig.canvas.draw()

    toggle_selector = RectangleSelector(ax, onselect, drawtype="box", useblit=True)

    def on_key(event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    final_boxes = []
    for b in boxes:
        width = b[2] - b[0]
        height = b[3] - b[1]
        if width > args.tile_size or height > args.tile_size:
            final_boxes.extend(split_box(b, args.tile_size, args.tile_size))
        else:
            final_boxes.append(b)

    with open(os.path.join(args.output, "boxes.json"), "w", encoding="utf-8") as f:
        json.dump(final_boxes, f)

    crop_tiles(img, final_boxes, args.output)
    print(f"Saved {len(final_boxes)} tiles to {args.output}")


if __name__ == "__main__":
    main()
