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
assert(/"object_mask"/.test(source), "tile editor should preserve SAM object masks in layout metadata");

console.log("smoke_smart_tile_interactive: ok");
