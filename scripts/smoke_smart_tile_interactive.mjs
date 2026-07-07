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
assert(/app\.queuePrompt\(0,\s*1,\s*\[String\(node\.id\)\]\)/.test(source), "Infer action should queue only the current Smart Tile node");
assert(/ttp-smart-tile-layout/.test(source), "Infer action should listen for backend layout updates");
assert(/auto_detect_request/.test(source), "Infer action should increment the node inference request widget");
assert(/options\.fillGaps !== false/.test(source), "Infer action should allow disabling automatic gap filling");
assert(/modeOverride: "sam3\.1"/.test(source), "Auto SAM should force SAM3.1 detection");
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
assert(/function autoMaxTiles/.test(source), "tile editor should read the Auto Tile max tile limit");
assert(/const filled = fillTileGaps\(node, layout\.tiles, autoMaxTiles\(node\)\);/.test(source), "Auto Tile gap fill should respect auto_max_tiles");
assert(/if \(fillGaps\) \{\s*const filled = fillTileGaps\(node, layout\.tiles, autoMaxTiles\(node\)\);/s.test(source), "Auto Tile should only fill gaps when requested");
assert(/writeLayout\(node, layout\.tiles, 0\);\s*const coverage = analyzeCoverage\(layout\.tiles\);/s.test(source), "Auto SAM should write detected tiles without gap filling");
assert(/source: "auto_gap"/.test(source), "auto-filled gap tiles should be marked as background gap tiles");
assert(/function editorStageSize/.test(source), "tile editor should compute a fixed source-ratio stage size");
assert(!/aspect-ratio:/.test(source), "tile editor should not rely on CSS aspect-ratio when the node is resized");
assert(!/max-height:720px/.test(source), "tile editor should not clamp only stage height and distort the image");
assert(/"object_mask"/.test(source), "tile editor should preserve SAM object masks in layout metadata");
assert(/"object_mask_source"/.test(source), "tile editor should preserve parent object masks for refresh");
assert(/GRID_MASK_MODES/.test(source), "tile editor should expose grid mask inheritance modes");
assert(/async function gridTilesWithInheritedMask/.test(source), "Grid in should be able to inherit and crop object masks");
assert(/async function refreshInheritedMasks/.test(source), "tile editor should refresh inherited masks after manual edits");
assert(/function cropObjectMaskForTile/.test(source), "Grid in should crop object masks per child tile");
assert(/function isLargeContextTile/.test(source), "Grid in should detect huge context tiles");
assert(/function contextGridTileMetadata/.test(source), "Grid in should keep huge context children on the lowest layer");
assert(/tileAreaRatio\(tile\) >= largeContextAreaRatio/.test(source), "huge non-detail tiles should be treated as context tiles");
assert(/const maskData = sourceObjectMaskData\(sourceTile\);/.test(source), "huge context Grid in should still preserve and crop the body mask");
assert(!/sourceIsLargeContext \? null : sourceObjectMaskData/.test(source), "huge context Grid in must not drop the source body mask");
assert(/crop_mask_skip_empty/.test(source), "Grid in should allow skipping empty mask child tiles");
assert(/createButton\(`Grid in T\$\{selectedIndex \+ 1\}`, async \(\) =>/.test(source), "Grid in should run asynchronously for mask cropping");
assert(/ensurePaintMaskCanvas/.test(source), "tile editor should create a paint mask canvas");
assert(/auto_paint_mask/.test(source), "tile editor should sync painted masks into the hidden backend input");
assert(/MASK_OVERLAY_COLORS/.test(source), "tile editor should define distinct colors for object-mask overlays");
assert(/function maskOverlayEnabled/.test(source), "tile editor should expose a mask overlay toggle state");
assert(/async function renderMaskOverlay/.test(source), "tile editor should render object masks as a colored overlay");
assert(/tintedMaskCanvas/.test(source), "tile editor should tint mask pixels before showing overlays");
assert(!/canvas\.isConnected/.test(source), "mask overlay rendering should not require the staged canvas to be connected before renderEditor appends it");
assert(/Show masks/.test(source), "tile editor should expose a show masks toggle");
assert(/Hide masks/.test(source), "tile editor should expose a hide masks toggle");
assert(/ttpSmartTileShowMasks/.test(source), "mask overlay visibility should be stored on the editor node instance");
assert(/renderMaskOverlay\(node, maskOverlay, tiles\)/.test(source), "renderEditor should draw the colored mask overlay when enabled");
assert(/Brush/.test(source), "tile editor should expose a paint brush action");
assert(/Erase/.test(source), "tile editor should expose an erase action");
assert(/Mask to Tile/.test(source), "tile editor should expose a button that commits painted masks into layout tiles");
assert(/Mask Replace/.test(source), "tile editor should expose a button that replaces the layout with painted mask tiles");
assert(/Refresh masks/.test(source), "tile editor should expose a refresh masks action");
assert(/refreshInheritedMasks\(node, tiles, gridMaskMode\(node\)\)/.test(source), "Refresh masks should use the current grid mask mode");
assert(/function selectedTileIndexes/.test(source), "tile editor should track multi-selected tiles");
assert(/function setTileSelection/.test(source), "tile editor should support modifier-click multi-selection");
assert(/boolEventModifier\(event\)/.test(beginDrag), "modifier-clicking an existing tile should toggle multi-selection");
assert(/async function mergedObjectMaskForTiles/.test(source), "tile editor should union selected tile masks");
assert(/async function mergeSelectedMasks/.test(source), "tile editor should expose selected mask merging logic");
assert(/async function mergeSelectedTiles/.test(source), "tile editor should expose selected tile merging logic");
assert(/Merge masks/.test(source), "tile editor should expose a merge masks action");
assert(/Merge tiles/.test(source), "tile editor should expose a merge tiles action");
assert(/ownObjectMaskData/.test(source), "mask merging should prefer each tile's cropped mask over parent source masks");
assert(/function addPaintMaskTiles/.test(source), "tile editor should convert painted masks into persisted tiles");
assert(/auto_mask_expand/.test(source), "paint-created tiles should read the auto mask expansion widget");
assert(/expandMaskImageData/.test(source), "paint-created masks should support foreground expansion");
assert(/maskCropData\(canvas, box, maskExpand\)/.test(source), "paint-created masks should apply the requested mask expansion");
assert(/source: "paint_mask"/.test(source), "paint-created tiles should be marked as paint mask tiles");
assert(/object_mask:\s*\{/.test(source), "paint-created tiles should persist object masks in layout metadata");
assert(/Clear mask/.test(source), "tile editor should expose a clear mask action");
assert(/syncPaintMaskWidget\(node\);\s*inferSmartTileLayout\(node\);/s.test(source), "Auto Tile should submit the latest painted mask before inference");
assert(/Auto SAM/.test(source), "tile editor should expose an Auto SAM action");
assert(/syncPaintMaskWidget\(node\);\s*inferSmartTileLayout\(node, \{\s*fillGaps: false,/s.test(source), "Auto SAM should submit masks and disable automatic gap filling");
assert(!/PROMPT_BUILDER_NAME/.test(source), "frontend must not hook the QwenVL prompt builder node");
assert(!/repairPromptBuilderWidgets/.test(source), "prompt builder values must be left to native ComfyUI widget serialization");
assert(!/ttp_prompt_builder_values/.test(source), "prompt builder must not restore stale named widget values from properties");
assert(!/ttp_prompt_builder_named_values/.test(source), "prompt builder should not use custom frontend persistence");
assert(!/node\.widgets\?\.(?:sort|splice|reverse)/.test(source), "frontend must not reorder the LiteGraph widget array");

console.log("smoke_smart_tile_interactive: ok");
