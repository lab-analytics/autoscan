import re
import pandas as pd
from datetime import datetime, timezone
import hashlib
import numpy as np
import logging

# logging.basicConfig(
#     level=logging.INFO,  # Log INFO level and above
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )

# Create a logger object
logger = logging.getLogger("ASDParser")
logger.setLevel(logging.WARNING)

# Create a FileHandler that writes log messages to a file
file_handler = logging.FileHandler("asd_parser.log")
file_handler.setLevel(logging.DEBUG)

# Define a logging format
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)

# Optionally, also add a StreamHandler to see logs on screen if desired
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Configuration for each tip type.
# function for perm
def parse_perm_line(tokens) -> dict:
    """
    Extracts desired fields from a perm measurement token list.
    Expected tokens (after splitting) might look like:
    ["{", "Permeability", "31.471", "Pressure", "296.94", "Flow", "368.983",
     "Measurement_Time", "37.4235", "Measurement_Code", "6", "Klinkenberg_B", "13.7614",
     "Forchheimer_Factor", "2.082e-05", "Geometry_Factor", "0.0059", "Viscosity", "1.78e-05",
     "Uncorrected_Permeability", "27.2581", "ptip", "204.61", "Command", "...", "}"]
     
    This function extracts:
      - Permeability
      - Pressure
      - Measurement_Code
      - Klinkenberg_B
      - Forchheimer_Factor
      - Geometry_Factor
    """
    # Remove leading and trailing braces if present.
    tokens = [t for t in tokens if t not in ['{', '}']]
    
    # Define a mapping from token keys to our desired dictionary keys.
    desired_keys = {
        'Permeability': 'permeability',
        'Pressure': 'pressure',
        'Measurement_Code': 'measurement_code',
        'Klinkenberg_B': 'klinkenberg_b',
        'Forchheimer_Factor': 'forchheimer_factor',
        'Geometry_Factor': 'geometry_factor'
    }
    result = {}
    # Iterate over tokens and check if token matches a desired key.
    for i, token in enumerate(tokens):
        if token in desired_keys:
            try:
                result[desired_keys[token]] = convert_to_float(tokens[i + 1])
            except (IndexError, ValueError):
                result[desired_keys[token]] = None
    return result

def convert_to_float(token) -> float:
    """convert_to_float 
    Converts a token to a float. If the token cannot be converted to a float, it returns NaN.

    Args:
        token (str): str or token to conver to float

    Returns:
        float: float of token or np.nan if token cannot be converted to a float
    """
    try:
        return float(token)
    except ValueError:
        return np.nan

def strlist_to_float(l:list):
    """strlist_to_float 
    Converts a list of strings to a list of floats.

    Args:
        l (list): list of strings

    Returns:
        list: list of floats
    """
    
    return [convert_to_float(x) for x in l]
    
# For velocity tips, both 'velax' and 'velay' use the same configuration.
TIP_CONFIG = {
    "ftir": {
        "lines_per_measurement": 15,
        "add_missing_fields": False,
        "fields": {
            "temp": ("temp", convert_to_float),
            "rel_humidity": ("rel_humidity", convert_to_float),
            "abs_humidity": ("abs_humidity", convert_to_float),
            "NSS": ("NSS", convert_to_float),
            "quality": ("quality", convert_to_float),
            "AB": ("ftir_reflectances", lambda vals: dict(zip(strlist_to_float(vals[::2]), strlist_to_float(vals[1::2])))),
                #dict(zip(map(convert_to_float, vals[::2]), map(convert_to_float, vals[1::2]))))
                   #lambda vals: list(map(convert_to_float, vals[1::2]))) 
        }
    },
    "impulse": {
        "lines_per_measurement": 12,
        "add_missing_fields": False,
        "fields": {
            "force/sampleinterval": ("force_sampleinterval", convert_to_float),
            "force/scoperange": ("force_scoperange", convert_to_float),
            "force/starttime": ("force_starttime", str),
            "force/value": ("force_value", strlist_to_float),
            "accel/sampleinterval": ("accel_sampleinterval", convert_to_float),
            "accel/scoperange": ("accel_scoperange", convert_to_float),
            "accel/starttime": ("accel_starttime", str),
            "accel/value": ("accel_value", strlist_to_float),
            "tipattributes/value": ("tipattributes", 
                                    lambda vals: {vals[i].lower(): convert_to_float(vals[i+1]) for i in range(0, len(vals), 2)}),
            "height/value": ("height_value", convert_to_float),
            "IHproc/values": ('ihproc_values', strlist_to_float),
            "IHproc/Estar_findex/value": ('ihproc_estar_findex', convert_to_float),
            "IHproc/Estar/units": ('ihproc_estar_units', str),
            "IHproc/Estar/value": ('ihproc_estar_value', convert_to_float)
        }
    },
    "perm": {
        "lines_per_measurement": 2,
        "add_missing_fields": False,
        "fields": {
            "tagvalrec_psimp": ("perm_values", parse_perm_line),
            # "asctab_log": ("asctab_log", lambda x: None)
        }
    },
    "vel": {  # used for both velax and velay
        "lines_per_measurement": 18,
        "add_missing_fields": False,
        'data_pattern': r"angle/(\d+)/(\w+)/",
        'field_pattern': r"/(\w+)$",
        "fields": {
            "velocity": ("velocity", convert_to_float),
            "h2h": ("h2h", convert_to_float),
            "measuredangle": ("measuredangle", convert_to_float),
            "misfit": ("misfit", convert_to_float),
            "offsetrcv": ("offsetrcv", convert_to_float),
            "offsetsrc": ("offsetsrc", convert_to_float),
            "pick": ("pick", convert_to_float),
            "rcvrgain": ("rcvrgain", convert_to_float),
            "sampleinterval": ("sampleinterval", convert_to_float),
            "scoperange": ("scoperange", convert_to_float),
            "starttime": ("starttime", str),
            "tipforceaux": ("tipforceaux", convert_to_float),
            "tipforcemain": ("tipforcemain", convert_to_float),
            "value": ("value", strlist_to_float), 
            #[convert_to_float(vals[i]) for i in range(len(vals))]),
            "average": ("average", convert_to_float)
        }
    },
    "LProfile": {
        "lines_per_measurement": 1,
        "add_missing_fields": True,
        "fields": {
            "Lprofile_record": ("laser_profile", lambda x: convert_to_float(x[2]) if len(x) >= 2 else np.nan) #(re.search(r'\{ las (-\d+\.\d+) \}', x).group(1))),
        }
    },
    "default": {
        "lines_per_measurement": 1,
        "add_missing_fields": True,
        "fields": {
            "default_field": ("default_value", convert_to_float)
        }
    }
}

MEAS_CATEGORY = {
    'spectrum': 'ftir',
    'impulse': 'impulse',
    'perm': 'perm',
    'vel': 'vel',
    'height': 'LProfile'
}

BLOCK_CONFIG = {
    "tile" : {
        'pattern': r'tile/\d+/type',
        'lines': 5,
        'fields': {
            'tile_number': ("tile_number", int),
            'tile_grid_type': ("tile_grid_type", str),
            'info_pattern' : r"tile/(\d+)/type\s+(.+)",
            'coords_origin_pattern' : r"tile/\d+/origin/value (\d+\.\d+) (\d+\.\d+)",
            'coords_corner_pattern' : r"tile/\d+/corner/value (\d+\.\d+) (\d+\.\d+)",
            'x_origin': ("x_origin", float),
            'y_origin': ("y_origin", float),
            'x_corner': ("x_corner", float),
            'y_corner': ("y_corner", float)
        }
    },
    "category": {
        'pattern': r'tile/\d+/tip/.*?/category',
        'lines': 1,
        'fields': {
            'category': ("category", str)
        }
    },
    "tip" : {
        'pattern': r"tip/(.+?)/category",
        'lines': 1,
        'fields': {
            'tip_name': ("tip_name", str)
        }
    },
    'measurement' : {
        'coords_pattern' : r"tile/(\d+)/tip/([^/]+)/pt/([\d\.]+)/([\d\.]+)/data/"
    }
}

class ASDParser:
    def __init__(self, file_path:str, tip_name:str = "default", base:str = "origin", **kwargs):
        """__init__ 
            Initialize ASDParser tip_name: the measurement tip type provided by the user,
                  e.g. "ftir", "impulse", "perm", "velax", or "velay".
                  For velocity, both "velax" and "velay" use the "vel" configuration.

        Args:
            file_path (str): path to the ASD file
            tip_name (str, optional): tip used for characterization (e.g. "ftir", "impulse", "perm", "velax", or "velay".). 
                                      It can be inferred from ASD file. Defaults to "default".
            base (str, optional): sample state, before (origin) or after some experiment. Defaults to "origin".
        """

        self.file_path = file_path
        self.tip_name = tip_name  # initial tip type
        self.sample_state = base
        self.sample_info = {}
        self.tile_geometries = {}  # keyed by tile number
        self.measurements = {}     # aggregated by (tile, x, y)
        self.tip_config = {}       # configuration for the tip type
        self._tips_config = TIP_CONFIG # add the dictionaries of tip_configs
        self._meas_category = MEAS_CATEGORY # add the dictionary of measurement categories
        self._block_config = BLOCK_CONFIG # add the dictionary of block configurations
        self._update_tip_config(tip_name) # update tip_name and tip_config
        self.sample_side = kwargs.get('side', None)
        self.sample_shape = kwargs.get('shape', None)
        self._compile_patterns()
        # re.match(self._block_config['measurement']['coords_pattern'], first_line)
        # m_origin = re.match(self._block_config['tile']['fields']['coords_origin_pattern'], origin_line)
        # m_corner = re.match(self._block_config['tile']['fields']['coords_corner_pattern'], corner_lines)

    def _compile_patterns(self):
        """_compile_patterns 
        Compile the regular expressions for the block patterns.
        """

        # tile_pattern = re.compile(self._block_config['tile']['pattern'])
        # category_pattern = re.compile(self._block_config['category']['pattern'])
        # tip_pattern = re.compile(self._block_config['tip']['pattern'])
        self._tile_pattern = re.compile(self._block_config['tile']['pattern'])
        self._category_pattern = re.compile(self._block_config['category']['pattern'])
        self._tip_pattern = re.compile(self._block_config['tip']['pattern'])
        self._measurement_coords_pattern = re.compile(self._block_config['measurement']['coords_pattern'])
        self._tile_origin_pattern = re.compile(self._block_config['tile']['fields']['coords_origin_pattern'])
        self._tile_corner_pattern = re.compile(self._block_config['tile']['fields']['coords_corner_pattern'])
        self._tile_info_pattern = re.compile(self._block_config['tile']['fields']['info_pattern'])
        
    def _update_tip_config(self, tip_name:str = "default", tip_config:dict = None):
        """_update_tip_config
        Update the tip_name and tip_config attributes. If a new tip_config is passed, then updates the _tips_config dictionary. 

        Args:
            tip_name (str, optional): _description_. Defaults to "default".
            tip_config (dict, optional): _description_.
        """
        # if tip_config is not None, then append the tip_config dict to self._tips_config dictionary.
        if tip_config is not None:
            self._tips_config[tip_name] = tip_config
        
        # update tip_name and tip_config. If tip_name is None, then get "default" config from TIP_CONFIG
        self.tip_name = tip_name if tip_name not in ['velax', 'velay'] else "vel"
        self.tip_config = self._tips_config.get(self.tip_name, self._tips_config["default"])

    # generate a numeric UID for the sample as a unique identifier using a hash of the file. 
    # This will help us track if the file has been modified.
    def _data_uid(self) -> str:
        """_sample_uid 
        Returns a unique identifier for the sample using a hash of the file.
        
        Returns:
            hash (str): hexadecimal hash of the file_path.
        """
        return hashlib.sha256(open(self.file_path, 'rb').read()).hexdigest()

    # generate a UID for the sample using sample name and epoch creation
    def _sample_uid(self) -> str:
        """_sample_uid 
        Returns a unique identifier for the sample using the sample name and epoch creation.
        
        Returns:
            hash (str): hexadecimal hash of the sample name and epoch creation.
        """
        return hashlib.sha256(f"{self.sample_info['sample_name']}_{self.sample_info['epoch_creation']}".encode()).hexdigest()

    def _sample_base_uid(self) -> str:
        """_sample_uid 
        Returns a unique identifier for the sample using the sample name and epoch creation.
        
        Returns:
            hash (str): hexadecimal hash of the sample's name, side, and state.
        """
        # concatenate the values of sample_name, side, and side into a single text string
        base = f"{self.sample_info['sample_name']}_{self.sample_info.get('side', '')}_{self.sample_info.get('state', '')}"
        return hashlib.sha256(base.encode()).hexdigest()

    def parse(self, force_add_missing_fields:bool = False, header_lines:int = 12) -> list:
        """parse 

        General method to parse an ASD file.
        The method reads the file, processes the header and body, and returns the sample information and measurements.

        Returns:
            sample, measurements (list): list of dataframes with sample information and measurement. 
                                         Measurements is a dictionary with keys for each tip type and values as dataframes.   
        """
        
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        # --- Process header (first 12 lines) ---
        self._parse_header(lines[:12])
        self.sample_info['file_hash'] = self._data_uid()
        self.sample_info['side'] = self.sample_side
        self.sample_info['shape'] = self.sample_shape
        self.sample_info['state'] = self.sample_state
        self.sample_info['sample_uid'] = self._sample_uid()
        self.sample_info['base_uid'] = self._sample_base_uid()
        
        current_sample_uid = self.sample_info.get('sample_uid', None)
        current_sample_based_uid = self.sample_info.get('base_uid', None)
        current_sample = self.sample_info.get('sample_name', "unknown")

        logger.info(f"Parsing ASD file: {self.file_path}")
        logger.info(f"Sample name: {current_sample}, sample UID: {current_sample_uid[:4]}, base UID: {current_sample_based_uid[:4]}")
        
        # --- Process body (line 13 onward) ---
        # Set initial tip configuration and block size.
        tip_config = self.tip_config
        meas_block_size = tip_config.get("lines_per_measurement", 1)
        add_missing_fields = tip_config.get("add_missing_fields", False)
        if force_add_missing_fields:
            add_missing_fields = True
            logger.warning("Focing add of missing fields to measurements.")
        
        # State variables for current tile and category.
        current_tile = None
        current_category = None
        current_tip = self.tip_name
        
        # measure_line_data_pattern = r"/data/(.+)"
        # precompile common regular expressions


        # get tile and category lines from _block_configs sub-dictionaries using get method and default to 1 if not specified
        tile_lines = self._block_config['tile'].get('lines', 5)
        category_lines = self._block_config['category'].get('lines', 1)
        
        # Iterate through the lines to parse the measurements.
        i = header_lines  # start reading from line 13 (0-indexed)
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check for tile definition block.
            if self._tile_pattern.match(line): #re.match(self._block_config['tile']['pattern'], line):
                # Process 5-line tile definition block.
                current_tile = self._parse_tile_definition(lines[i:i+5])
                current_category = None # reset category with new tile definition
                i += tile_lines #self._block_config['tile']['lines']
            
            # Check for category definition block.
            elif self._category_pattern.match(line): #re.match(self._block_config['category']['pattern'], line):
                parts = line.split('category ', 1)
                current_category = parts[1].strip() if len(parts) > 1 else None

                # Update the tip configuration based on the category or tip
                try:
                    current_tip = self._tip_pattern.search(line).group(1)
                    logger.info(f'found category \"{current_category}\" with tip \"{current_tip}\" at line {i}')
                    #re.search(self._block_config['tip']['pattern'], line).group(1)
                except:
                    current_tip = "default"
                    logger.warning(f'category \"{current_category}\" without tip at line {i}. Tip set to default')
                
                # if the current_tip is "velax" or "velay" extract the direction, x or y and save it for later
                if current_tip in ['velax', 'velay']:
                    direction = current_tip[-1]
                # update the tip configuration based on the current_tip value
                self._update_tip_config(current_tip)
                tip_config = self.tip_config
                meas_block_size = tip_config.get("lines_per_measurement", 1)
                current_tip = self.tip_name
                add_missing_fields = tip_config.get("add_missing_fields", False)
                if force_add_missing_fields:
                    add_missing_fields = True

                # check if self.measurements already has the key for the tip
                if current_tip not in self.measurements:
                    logger.info(f"Adding new tip \"{current_tip}\" to measurements")
                    self.measurements[current_tip] = {}  # Initialize the measurement block for the current tip.
                
                i += category_lines #self._block_config['category']['lines']
            
            # Otherwise, assume measurement block.
            elif "/tip/" in line and "/pt/" in line:
                block = [lines[i+j].strip() for j in range(meas_block_size) if i+j < len(lines)]
                record = self._parse_measurement_block(block=block, tip_config=tip_config, add_missing_fields=add_missing_fields)
                
                # Ensure current tile is set.
                if current_tile is not None:
                    record["tile_number"] = current_tile
                if current_category:
                    record["measurement_category"] = current_category
                if current_tip:
                    record["measurement_tip_name"] = current_tip
                if current_sample:
                    record['sample_name'] = current_sample    
                if current_sample_uid:
                    record['sample_uid'] = current_sample_uid
                if current_sample_based_uid:
                    record['base_uid'] = current_sample_based_uid
                    
                key = (
                    # record.get('sample_uid'),
                    # record.get('sample_name'),
                    # record.get('base_uid'),
                    record.get("tile_number", 1),
                    record.get("measurement_tip_name"),
                    record.get("measurement_category", ""),
                    record.get("measurement_x_coordinate"),
                    record.get("measurement_y_coordinate"),
                    record.get("data_type", "raw")
                )

                # if category is "vel" then we need to extract the angle and wave type from the line
                # For example, tile/{tile_number}/tip/{tip_name}/pt/{x}/{y}/data/angle/{angle_value}/{wave_type}/{feature} 4
                if current_category == 'vel':
                    record['direction'] = direction
                    angle_match = re.search(tip_config['data_pattern'], line) #r'angle/(\d+)/(\w+)/', line)
                    if angle_match:
                        record['measurement_angle'] = int(angle_match.group(1))
                        record['measurement_wave_type'] = angle_match.group(2)
                    else:
                        record['measurement_angle'] = 0
                        record['measurement_wave_type'] = 'p'

                    # update key
                    key += (record.get('measurement_angle', 0), 
                            record.get('measurement_wave_type', 'p'),
                            record.get('direction', 'x'))
                
                # For aggregation, later blocks for the same point could update the record.
                self.measurements[current_tip][key] = record
                i += meas_block_size
            else:
                i += 1

        # Build sample DataFrame.
        sample_record = {
            **self.sample_info,
            'tiles_geometry': list(self.tile_geometries.values())
        }
        sample_df = pd.DataFrame([sample_record])
        # iterate over the keys of self.measurements to creat a dict of dataframea
        measurement_dfs = {}
        measurement_dfs = {tip: pd.DataFrame(list(records.values())) 
                           for tip, records in self.measurements.items()}

        return sample_df, measurement_dfs

    def _parse_header(self, header_lines:list):
        """_parse_header
        Process the file header block.
        Extracts information from the header lines and adss them to the sample_info dictionary.

        Args:
            header_lines (list): List of header lines to extract
        """
        # Assume header_lines is a list of 12 lines.
        epoch_creation = int(header_lines[0].split()[1]) if len(header_lines[0].split()) > 1 else None
        epoch_ctime_str = header_lines[1].strip().split('epoch/ctime ', 1)[1] if 'epoch/ctime' in header_lines[1] else None

        # Convert epoc_ctime to timestamp
        try:
            epoch_ctime = datetime.strptime(epoch_ctime_str, '%Y-%m-%d %H:%M:%S %Z')
            epoch_ctime = epoch_ctime.astimezone(timezone.utc) if epoch_ctime else None
        except:
            logger.error('Error parsing epoch')
            epoch_ctime = epoch_ctime_str

        # Get project name
        project_name = header_lines[4].split('project/name ', 1)[1] if 'project/name ' in header_lines[4] else None

        # Get sample name
        sample_name = header_lines[5].split('sample/name ', 1)[1] if 'sample/name ' in header_lines[5] else None

        # Get sample depth (in meters)
        try:
            sample_depth = convert_to_float(header_lines[6].split()[1]) if len(header_lines[6].split()) > 1 else None
        except ValueError:
            sample_depth = None
            logger.error('Error parsing sample depth')

        # Get the origin of the rectangle inscribing the sample
        origin_parts = header_lines[8].split()
        try:
            x_origin = convert_to_float(origin_parts[1]) if len(origin_parts) >= 3 else None
            y_origin = convert_to_float(origin_parts[2]) if len(origin_parts) >= 3 else None
        except ValueError:
            x_origin, y_origin = None, None
            logger.error('Error parsing sample origin')

        # Get the corner of the rectangle inscribing the sample
        corner_parts = header_lines[10].split()
        try:
            x_corner = convert_to_float(corner_parts[1]) if len(corner_parts) >= 3 else None
            y_corner = convert_to_float(corner_parts[2]) if len(corner_parts) >= 3 else None
        except ValueError:
            x_corner, y_corner = None, None
            logger.error('Error parsing sample corner')

        self.sample_info = {
            'epoch_creation': epoch_creation,
            'sample_ctime': epoch_ctime,
            'project_name': project_name,
            'sample_name': sample_name,
            'sample_depth': sample_depth,
            'x_origin': x_origin,
            'y_origin': y_origin,
            'x_corner': x_corner,
            'y_corner': y_corner
        }

    def _parse_tile_definition(self, tile_lines:list) -> int:
        """_parse_tile_definition

        Process a block of lines that define a tile.
        The first line has the pattern: tile/{tile_number}/type {tile_grid_type}

        Args:
            tile_lines (list): Block with lines that define a tile.

        Returns:
            tile_number (int): tile number
        """
        
        first_line = tile_lines[0].strip()
        origin_line = tile_lines[1].strip()
        corner_lines = tile_lines[3].strip()

        # Extract the origin and corner values from the respective lines.
        # Example tile/2/origin/value 0.0 300.0        
        m_origin = self._tile_origin_pattern.match(origin_line) #re.match(self._block_config['tile']['fields']['coords_origin_pattern'], origin_line)
        m_corner = self._tile_corner_pattern.match(corner_lines) #re.match(self._block_config['tile']['fields']['coords_corner_pattern'], corner_lines)
        x_origin, y_origin = None, None
        x_corner, y_corner = None, None
        if m_corner:
            x_corner = convert_to_float(m_corner.group(1))
            y_corner = convert_to_float(m_corner.group(2))
        if m_origin:
            x_origin = convert_to_float(m_origin.group(1))
            y_origin = convert_to_float(m_origin.group(2))
        
        # Extract tile number and grid type from the first line.
        tile_number = 1
        tile_grid_type = None
        
        m = self._tile_info_pattern.match(first_line) #re.match(self._block_config['tile']['fields']['info_pattern'], first_line)
        if m:
            try: 
                tile_number = int(m.group(1))
            except ValueError:
                tile_number = 1
                logger.error('Error parsing tile number')
            tile_grid_type = m.group(2)

        self.tile_geometries[tile_number] = {
                "tile_type": tile_grid_type,
                "origin": [x_origin, y_origin],
                "corner": [x_corner, y_corner]
            }
        return tile_number

    def _parse_measurement_block(self, block:list, tip_config:dict, add_missing_fields:bool = False) -> dict:
        """_parse_measurement_block

        Process a block of measurement lines (the number of lines is specified in tip_config).
        A record is built by scanning each line and using the configuration mapping.
        

        Args:
            block (list): Block of lines that define a measurement.
            tip_config (dict): Dictionary mapping configuration

        Returns:
            record (dict): Measurement information as defined in the mapping configuration
        """

        record = {}
        # Extract common fields (tile, tip, coordinates) from the first line.
        first_line = block[0]
        m = self._measurement_coords_pattern.match(first_line) #re.match(self._block_config['measurement']['coords_pattern'], first_line)
        #m = re.match(r"tile/(\d+)/tip/([^/]+)/pt/([\d\.]+)/([\d\.]+)/data/", first_line)
        if m:
            record["measurement_x_coordinate"] = convert_to_float(m.group(3))
            record["measurement_y_coordinate"] = convert_to_float(m.group(4))

        # check if the line includes a processing label
        record["data_type"] = 'raw'
        if 'proc' in first_line:
            logger.info("Found a processing label \"proc\" as measurement tag")
            record['data_type'] = 'processed'

        # Get the field pattern from tip_config or use the default
        field_pattern = tip_config.get('field_pattern', r"/data/(.+)")

        for line in block:    
            parts = line.split()

            # Extract the field part from the identifier.
            m_field = re.search(field_pattern, parts[0])
            if not m_field:
                continue
            field = m_field.group(1)

            # verify that the field is in the keys of self.tip_config['fields'] dictionary
            if field in tip_config["fields"]:
                col_name, conv_func = tip_config["fields"][field]
            elif add_missing_fields:
                logger.warning(f"Adding missing field \"{field}\" for tip to measurements.")
                col_name = str(field)
                conv_func = str
            else:
                continue
            
            # Convert the value(s) to the specified type and store in the record.
            try:
                vals = parts[1:]
                value = conv_func(vals[0]) if len(vals) == 1 else conv_func(vals)
                record[col_name] = value
            except Exception as e:
                logger.error(f"Error parsing field \"{field}\" for tip in line\n\t{line}:\n\t{e}")
                record[col_name] = None
        return record

# Example main function.
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Efficient ASD to Parquet Converter")
    parser.add_argument("file", required=True, 
                        help="Path to the ASD file")
    parser.add_argument("--tip", required=False, default="Default",
                        help="Measurement tip type (e.g., ftir, impulse, perm, velax, velay)")
    parser.add_argument("--sample_output", 
                        default="sample_info.parquet", 
                        help="Output file for sample info")
    parser.add_argument("--measurement_output", 
                        default="data.parquet", 
                        help="Output file for measurements")
    parser.add_argument('--force_add_missing_fields', default=False, action='store_true',
                        help="Add missing fields")
    
    args = parser.parse_args()
    
    asd_parser = ASDParser(args.file, args.tip, args.add_missing_fields)
    sample_df, measurement_dfs = asd_parser.parse(force_add_missing_fields=args.force_add_missing_fields)
    
    sample_df.to_parquet(args.sample_output, index=False)
    for tip, measurement_df in measurement_dfs.items():
        measurement_df.to_parquet(tip + args.measurement_output, index=False)
    print("Conversion complete.")

if __name__ == "__main__":
    main()
