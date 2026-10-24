import math
import time


class PushDetector:
    """
    Detects the 'push' gesture: with two fingers standing on the board, one
    finger reaches out past the other one (like a foot kicking off the ground).

    It doesn't care which finger does the pushing. It measures how far apart
    the index and middle fingertips are *along the direction the hand points*,
    divided by palm size, so it still works if you move the hand closer to or
    further from the camera, or rotate it a bit.

    Feed it the landmarks (wrist, index tip, middle knuckle, middle tip) once
    per frame with update(). It returns True on the frame a push starts.

    Tuning (watch the 'reach' number on screen while you do it):
      on_thresh  - reach above this = a push has happened
      off_thresh - reach must drop below this before the next push can count
                   (stops one push being counted several times)
      cooldown   - minimum seconds between two pushes
      alpha      - smoothing, 0-1. Higher = quicker response but more jitter
    """

    LANDMARKS = (0, 8, 9, 12)  # wrist, index tip, middle knuckle, middle tip

    def __init__(self, on_thresh=0.35, off_thresh=0.20, cooldown=0.25, alpha=0.6):
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.cooldown = cooldown
        self.alpha = alpha

        self.reach = None      # smoothed reach, None while no hand is visible
        self.armed = False     # True once reach has dropped back to rest
        self.last_push = 0.0

    def _raw_reach(self, pts):
        wrist, index_tip, mid_knuckle, mid_tip = pts
        ax, ay = mid_knuckle[0] - wrist[0], mid_knuckle[1] - wrist[1]
        palm = math.hypot(ax, ay)
        if palm < 1:
            return None
        ax, ay = ax / palm, ay / palm  # unit vector pointing along the hand

        def along(p):
            return ((p[0] - wrist[0]) * ax + (p[1] - wrist[1]) * ay) / palm

        return abs(along(index_tip) - along(mid_tip))

    def update(self, pts, now=None):
        now = time.time() if now is None else now
        raw = self._raw_reach(pts) if len(pts) == len(self.LANDMARKS) else None

        if raw is None:
            # hand lost: forget the state, and don't allow a push until we've
            # seen the fingers at rest again (avoids a false push on re-detect)
            self.reach = None
            self.armed = False
            return False

        if self.reach is None:
            self.reach = raw
        else:
            self.reach = self.alpha * raw + (1 - self.alpha) * self.reach

        if not self.armed:
            if self.reach < self.off_thresh:
                self.armed = True
            return False

        if self.reach > self.on_thresh and now - self.last_push >= self.cooldown:
            self.armed = False
            self.last_push = now
            return True
        return False
