import math

# generate evenly spaced points on bezier curve for cursor to follow as "objects"

def interpolate(a, b, t):
    return a + (b - a) * t

def bezier_point(points, t):
    # de casteljau algorithm for bezier path
    n = len(points)
    if n == 1:
        return points[0]

    new_points = []
    for i in range(n - 1):
        x = interpolate(points[i][0], points[i + 1][0], t)
        y = interpolate(points[i][1], points[i + 1][1], t)
        new_points.append((x, y))

    return bezier_point(new_points, t)

def calculate_bezier_path(control_points, num_points=50):
    # complex sliders, e.g. Sotarks wave sliders, etc.
    path = []
    for i in range(num_points + 1):
        t = i / num_points
        point = bezier_point(control_points, t)
        path.append(point)
    return path

def calculate_linear_path(control_points, num_points=20):
    # straight sliders
    if len(control_points) < 2:
        return control_points

    path = []
    total_length = 0
    segments = []

    for i in range(len(control_points) - 1):
        dx = control_points[i + 1][0] - control_points[i][0]
        dy = control_points[i + 1][1] - control_points[i][1]
        length = math.sqrt(dx * dx + dy * dy)
        segments.append((control_points[i], control_points[i + 1], length))
        total_length += length

    if total_length == 0:
        return [control_points[0]]

    for i in range(num_points + 1):
        target_dist = (i / num_points) * total_length
        current_dist = 0

        for start, end, seg_len in segments:
            if current_dist + seg_len >= target_dist:
                t = (target_dist - current_dist) / seg_len if seg_len > 0 else 0
                x = interpolate(start[0], end[0], t)
                y = interpolate(start[1], end[1], t)
                path.append((x, y))
                break
            current_dist += seg_len
        else:
            path.append(control_points[-1])

    return path


def calculate_perfect_circle_path(control_points, num_points=50):
    # simple curved sliders, C or O shaped sliders
    if len(control_points) < 3:
        return calculate_linear_path(control_points, num_points)

    p0, p1, p2 = control_points[0], control_points[1], control_points[2]

    ax, ay = p0
    bx, by = p1
    cx, cy = p2

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return calculate_linear_path(control_points, num_points)

    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d

    center = (ux, uy)
    radius = math.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)

    start_angle = math.atan2(ay - uy, ax - ux)
    mid_angle = math.atan2(by - uy, bx - ux)
    end_angle = math.atan2(cy - uy, cx - ux)

    def angle_diff(a, b):
        diff = b - a
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    dir1 = angle_diff(start_angle, mid_angle)
    dir2 = angle_diff(mid_angle, end_angle)

    if dir1 * dir2 < 0:
        # reverse direction
        total_angle = angle_diff(start_angle, end_angle)
        if total_angle > 0:
            total_angle -= 2 * math.pi
        else:
            total_angle += 2 * math.pi
    else:
        total_angle = angle_diff(start_angle, end_angle)

    path = []
    for i in range(num_points + 1):
        t = i / num_points
        angle = start_angle + total_angle * t
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        path.append((x, y))

    return path


def calculate_slider_path(hit_obj, num_points=50):
    if not hit_obj.curve_points:
        return [(hit_obj.x, hit_obj.y)]

    curve_type = hit_obj.curve_type or 'B'
    points = hit_obj.curve_points

    if curve_type == 'L':
        path = calculate_linear_path(points, num_points)
    elif curve_type == 'P' and len(points) == 3:
        path = calculate_perfect_circle_path(points, num_points)
    elif curve_type == 'B':
        # Bezier curves can have multiple segments separated by duplicate points (sharp angle sliders)
        segments = []
        current_segment = [points[0]]

        for i in range(1, len(points)):
            if points[i] == points[i - 1]:
                if len(current_segment) > 1:
                    segments.append(current_segment)
                current_segment = [points[i]]
            else:
                current_segment.append(points[i])

        if len(current_segment) > 1:
            segments.append(current_segment)

        path = []
        points_per_segment = max(10, num_points // max(1, len(segments)))
        for seg in segments:
            seg_path = calculate_bezier_path(seg, points_per_segment)
            if path and seg_path:
                seg_path = seg_path[1:]  
            path.extend(seg_path)
    else:
        path = calculate_linear_path(points, num_points)

    return path


class HitObject: 
    def __init__(self, x, y, time, type):
        self.x = x
        self.y = y
        self.time = time
        self.type = type

        self.curve_type = None  # B=Bezier, L=Linear, P=Perfect, C=Catmull
        self.curve_points = []  # list of (x, y) control points
        self.slides = 1  # number of slides (1 = no repeat)
        self.length = 0  # pixel length
        self.end_time = None  # calculated end time for sliders
        self.end_x = None  # end position
        self.end_y = None

    def is_circle(self):
        return (self.type & 1) != 0

    def is_slider(self):
        return (self.type & 2) != 0

    def is_spinner(self):
        return (self.type & 8) != 0
    
class Parser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}
        self.difficulty = {}
        self.timing_points = []
        self.hit_objects = []
        
    def parse(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            current_section = None
            
            for line in f:
                line = line.strip()
                
                if not line or line.startswith("//"):
                    continue
                
                if line.startswith("[") and line.endswith("]"): # start of new section
                    current_section = line[1:-1]
                    continue
                
                if current_section == "Metadata":
                    self._parse_metadata(line)
                elif current_section == "Difficulty":
                    self._parse_difficulty(line)
                elif current_section == "TimingPoints":
                    self._parse_timing_point(line)
                elif current_section == "HitObjects":
                    self._parse_hit_object(line)
        
        return self
    
    def _parse_metadata(self, line): 
        if ':' in line:
            key, value = line.split(":", 1)
            self.metadata[key] = value

    def _parse_difficulty(self, line):
        if ':' in line:
            key, value = line.split(":", 1)
            self.difficulty[key] = float(value)
            
    def _parse_timing_point(self, line): 
        # format: time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        # uninherited (red line): beatLength is ms per beat (positive)
        # inherited (green line): beatLength is -100/SV multiplier (negative)
        if ',' in line:
            arr = line.split(",")
            if len(arr) >= 2:
                time = int(float(arr[0]))
                beat_length_raw = float(arr[1])
                # check if uninherited or inherited
                uninherited = int(arr[6]) == 1 if len(arr) > 6 else (beat_length_raw > 0)

                if uninherited:
                    # red timing point: defines BPM
                    timing_point = {
                        "time": time,
                        "beat_length": beat_length_raw,  # ms per beat
                        "slider_velocity": 1.0,  # default SV
                        "uninherited": True
                    }
                else:
                    # green timing point: defines SV multiplier
                    sv_multiplier = -100.0 / beat_length_raw if beat_length_raw != 0 else 1.0
                    timing_point = {
                        "time": time,
                        "beat_length": None,  # inherited from previous
                        "slider_velocity": sv_multiplier,
                        "uninherited": False
                    }
                self.timing_points.append(timing_point)

    def _parse_hit_object(self, line):
        if ',' in line:
            arr = line.split(",")
            if len(arr) >= 5:
                hit_obj = HitObject(
                    x=int(arr[0]),
                    y=int(arr[1]),
                    time=int(arr[2]),
                    type=int(arr[3])
                )
                # circles
                if hit_obj.is_circle():
                    self.hit_objects.append(hit_obj)
                # sliders
                elif hit_obj.is_slider() and len(arr) >= 8:
                    self._parse_slider(hit_obj, arr)
                    self.hit_objects.append(hit_obj)

    def _parse_slider(self, hit_obj, arr):
        # format: x,y,time,type,hitSound,curveType|curvePoints,slides,length
        curve_data = arr[5]
        if '|' in curve_data:
            parts = curve_data.split('|')
            hit_obj.curve_type = parts[0]  # B, L, P, or C

            # parse control points
            hit_obj.curve_points = [(hit_obj.x, hit_obj.y)]  # start with slider head
            for point_str in parts[1:]:
                if ':' in point_str:
                    px, py = point_str.split(':')
                    hit_obj.curve_points.append((int(px), int(py)))

        hit_obj.slides = int(arr[6])
        hit_obj.length = float(arr[7])

        # calculate slider duration and end time
        base_sv = self.difficulty.get('SliderMultiplier', 1.4)
        sv_multiplier = self._get_sv_at_time(hit_obj.time)

        # velocity in osu! pixels per beat
        velocity = base_sv * sv_multiplier * 100

        # get beat length from timing points
        beat_length = self._get_beat_length_at_time(hit_obj.time)

        # duration = (length * slides) / (velocity / beat_length * 1000)
        # simplified: duration_ms = length * slides * beat_length / (velocity)
        if velocity > 0:
            duration = (hit_obj.length * hit_obj.slides * beat_length) / velocity
        else:
            duration = 0

        hit_obj.end_time = hit_obj.time + duration

        # calculate end position (last control point for odd slides, head for even)
        if hit_obj.slides % 2 == 1:
            # ends at slider tail
            if len(hit_obj.curve_points) > 0:
                hit_obj.end_x = hit_obj.curve_points[-1][0]
                hit_obj.end_y = hit_obj.curve_points[-1][1]
            else:
                hit_obj.end_x = hit_obj.x
                hit_obj.end_y = hit_obj.y
        else:
            # ends back at slider head
            hit_obj.end_x = hit_obj.x
            hit_obj.end_y = hit_obj.y

    def _get_sv_at_time(self, time):
        sv = 1.0
        for tp in self.timing_points:
            if tp['time'] <= time:
                sv = tp['slider_velocity']
            else:
                break
        return sv

    def _get_beat_length_at_time(self, time):
        beat_length = 500 
        for tp in self.timing_points:
            if tp['time'] <= time:
                if tp['uninherited'] and tp['beat_length'] is not None:
                    beat_length = tp['beat_length']
            else:
                break
        return beat_length
        