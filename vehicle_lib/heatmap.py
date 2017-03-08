import numpy as np

class StableHeatMaps:

    # Class for stablising detected heatmaps
    # Maintains history of heat maps over multiple frames
    # and takes aggregate of all frames
    
    def __init__(self, threshold, num_frames=20):
        self.frames = []
        self.threshold = threshold
        self.num_frames = num_frames

    def _add(self, frame):
        self.frames.insert(0, frame)

    def _sum(self):
        if len(self.frames) > self.num_frames:
            self.frames.pop()
        return np.sum(np.array(self.frames), axis=0)

    def _add_heat(self, heat_map, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat_map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        # Return updated heat_map
        return heat_map

    def _apply_threshold(self, heat_map, threshold):
        # Zero out pixels below the threshold
        heat_map[heat_map <= threshold] = 0
        # Return thresholded map
        return heat_map

    # Get aggregate heatmap over history of num_frames
    def generate(self, img, on_windows):
        heat_map = np.zeros_like(img)
        heat_map = self._add_heat(heat_map, on_windows)
        # Add heat map to frame buffer
        self._add(heat_map)
        aggregate_heatmap = self._sum()
        heat_map = self._apply_threshold(aggregate_heatmap, self.threshold)
        return heat_map