class HitObject: # single hitobject in beatmap
    def __init__(self, x, y, time, type):
        self.x = x
        self.y = y
        self.time = time
        self.type = type

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
    
    def _parse_metadata(self, line): # parse metadata
        if ':' in line:
            key, value = line.split(":", 1)
            self.metadata[key] = value

    def _parse_difficulty(self, line): # parse difficulty
        if ':' in line:
            key, value = line.split(":", 1)
            self.difficulty[key] = float(value)
            
    def _parse_timing_point(self, line): # parse timing points
        # 117245,-100,4,2,1,5,0,0
        # time, sv = -100/value, 4, 2, 1, volume, ?, bool kiai
        if ',' in line:
            arr = line.split(",")
            if len(arr) > 1:
                timing_point = {"time": int(arr[0]), "slider_velocity": -100.0 / float(arr[1])}
                self.timing_points.append(timing_point)

    def _parse_hit_object(self, line): # parse objects in map
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
        