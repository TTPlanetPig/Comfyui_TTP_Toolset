import fs from "node:fs";

const source = fs.readFileSync("web/js/ttp_smart_tile_interactive.js", "utf8");

function assert(condition, message) {
    if (!condition) {
        throw new Error(message);
    }
}

function sliceBetween(startText, endText) {
    const start = source.indexOf(startText);
    assert(start >= 0, `missing start marker: ${startText}`);
    const end = source.indexOf(endText, start);
    assert(end > start, `missing end marker: ${endText}`);
    return source.slice(start, end);
}

const beginDrag = sliceBetween("const beginDrag = (event, tileIndex, mode, region) =>", "for (const index of renderOrder)");
const createTile = sliceBetween("stage.addEventListener(\"pointerdown\", (event) =>", "const controls = document.createElement(\"div\")");
const createTileBeforeMove = createTile.slice(0, createTile.indexOf("const apply ="));

assert(/const dragThresholdPx = 3/.test(source), "tile editor should use a small drag threshold");
assert(!/apply\(event\);/.test(beginDrag), "clicking an existing tile must not immediately apply a drag");
assert(/if \(!hasDragged && movedPx < dragThresholdPx\)/.test(beginDrag), "existing tile drag should wait until movement crosses the threshold");
assert(/renderEditor\(node\);/.test(beginDrag), "existing tile click should still re-render selection on pointerup");
assert(/pointercancel/.test(beginDrag), "existing tile drag should clean up pointercancel");
assert(/blur/.test(beginDrag), "existing tile drag should clean up window blur");
assert(!/setPointerCapture/.test(beginDrag), "existing tile drag should not rely on pointer capture");
assert(!/releasePointerCapture/.test(beginDrag), "existing tile drag should not rely on pointer capture release");
assert(!/writeLayout\(node, nextTiles, newIndex\);/.test(createTileBeforeMove), "empty-stage pointerdown should not create a tile before dragging");
assert(/if \(!hasDragged && movedPx < dragThresholdPx\)/.test(createTile), "new tile creation should wait until movement crosses the threshold");
assert(/async function inferSmartTileLayout/.test(source), "tile editor should expose an Infer action");
assert(/app\.queuePrompt\(0\)/.test(source), "Infer action should queue the current ComfyUI graph");
assert(/ttp-smart-tile-layout/.test(source), "Infer action should listen for backend layout updates");
assert(/auto_detect_request/.test(source), "Infer action should increment the node inference request widget");
assert(!/\/ttp\/smart_tile\/analyze/.test(source), "Infer action should not use the old analysis route");
assert(/occlusion_priority/.test(source), "interactive layout should preserve auto tile priority metadata");
assert(/TTP_Smart_Tile_Loop_Source_Experimental/.test(source), "frontend should know the Smart Tile Loop Source node");
assert(/Process All Tiles/.test(source), "loop source should expose a Process All Tiles button");
assert(/Start Loop \/ Process All Tiles/.test(source), "loop source start button should be clearly labeled");
assert(/function highlightLoopStartButton/.test(source), "loop source should highlight the start loop button");
assert(/widget\.computeSize = \(width\) => \[Math\.max\(300/.test(source), "loop source start button should be larger than a default widget");
assert(/Stop Tile Loop/.test(source), "loop source should expose a Stop Tile Loop button");
assert(/ttp-smart-tile-loop/.test(source), "frontend should listen for Smart Tile loop progress events");
assert(/queueSmartTileLoop\(node, false\)/.test(source), "loop event handler should queue the next tile automatically");
assert(/restart_request/.test(source), "loop source should increment restart requests");
assert(/loop_request/.test(source), "loop source should increment loop requests to avoid stale cached execution");
assert(/snapGuideThresholdPx = 10/.test(source), "tile editor should define a pixel threshold for guide snapping");
assert(/const snapTargets = \(axis, skipIndex\) =>/.test(source), "tile editor should build snap targets");
assert(/snapMovedTile/.test(source), "tile editor should snap moved tiles");
assert(/snapResizedTile/.test(source), "tile editor should snap resized tiles");
assert(/snapCreatedTile/.test(source), "tile editor should snap newly drawn tiles");
assert(/function fillTileGaps/.test(source), "tile editor should share automatic gap filling logic");
assert(/const filled = fillTileGaps\(node, layout\.tiles\);/.test(source), "Auto Tile should fill uncovered gaps before writing layout");
assert(/source: "auto_gap"/.test(source), "auto-filled gap tiles should be marked as background gap tiles");
assert(/function editorStageSize/.test(source), "tile editor should compute a fixed source-ratio stage size");
assert(!/aspect-ratio:/.test(source), "tile editor should not rely on CSS aspect-ratio when the node is resized");
assert(!/max-height:720px/.test(source), "tile editor should not clamp only stage height and distort the image");
assert(/"object_mask"/.test(source), "tile editor should preserve SAM object masks in layout metadata");
assert(/GRID_MASK_MODES/.test(source), "tile editor should expose grid mask inheritance modes");
assert(/async function gridTilesWithInheritedMask/.test(source), "Grid in should be able to inherit and crop object masks");
assert(/function cropObjectMaskForTile/.test(source), "Grid in should crop object masks per child tile");
assert(/crop_mask_skip_empty/.test(source), "Grid in should allow skipping empty mask child tiles");
assert(/createButton\(`Grid in T\$\{selectedIndex \+ 1\}`, async \(\) =>/.test(source), "Grid in should run asynchronously for mask cropping");
assert(/ensurePaintMaskCanvas/.test(source), "tile editor should create a paint mask canvas");
assert(/auto_paint_mask/.test(source), "tile editor should sync painted masks into the hidden backend input");
assert(/Brush/.test(source), "tile editor should expose a paint brush action");
assert(/Erase/.test(source), "tile editor should expose an erase action");
assert(/Mask to Tile/.test(source), "tile editor should expose a button that commits painted masks into layout tiles");
assert(/function addPaintMaskTiles/.test(source), "tile editor should convert painted masks into persisted tiles");
assert(/source: "paint_mask"/.test(source), "paint-created tiles should be marked as paint mask tiles");
assert(/object_mask:\s*\{/.test(source), "paint-created tiles should persist object masks in layout metadata");
assert(/Clear mask/.test(source), "tile editor should expose a clear mask action");
assert(/syncPaintMaskWidget\(node\);\s*inferSmartTileLayout\(node\);/s.test(source), "Auto Tile should submit the latest painted mask before inference");
assert(!/PROMPT_BUILDER_NAME/.test(source), "frontend must not hook the QwenVL prompt builder node");
assert(!/repairPromptBuilderWidgets/.test(source), "prompt builder values must be left to native ComfyUI widget serialization");
assert(!/ttp_prompt_builder_values/.test(source), "prompt builder must not restore stale named widget values from properties");
assert(!/ttp_prompt_builder_named_values/.test(source), "prompt builder should not use custom frontend persistence");
assert(!/node\.widgets\?\.(?:sort|splice|reverse)/.test(source), "frontend must not reorder the LiteGraph widget array");

console.log("smoke_smart_tile_interactive: ok");
