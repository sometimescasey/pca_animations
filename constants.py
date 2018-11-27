import os
import numpy as np

env_MEDIA_DIR = None
MEDIA_DIR = "#ERROR#"

try:
    env_MEDIA_DIR = os.getenv("MEDIA_DIR")
    print(env_MEDIA_DIR)
except NameError:
    try:
        env_MEDIA_DIR = os.environ['MEDIA_DIR']
    except KeyError:
        pass

if not (env_MEDIA_DIR is None):
    MEDIA_DIR = env_MEDIA_DIR
elif os.path.exists("media_dir.txt"):
    with open("media_dir.txt", 'rU') as media_file:
        MEDIA_DIR = media_file.readline().strip()
        print(MEDIA_DIR)
else:
    MEDIA_DIR = os.path.join(
        os.path.expanduser('~'),
        "~/gitrepo/pca_vid/casey_sandbox/media_output"
    )

if not os.path.exists(MEDIA_DIR):
    raise Exception("""
        Redefine MEDIA_DIR by changing the MEDIA_DIR
        environment constant or by changing
        media_dir.txt to point to a valid directory
        where movies and images will be written
    """)

with open("media_dir.txt", 'w') as media_file:
    media_file.write(MEDIA_DIR)
#

LOW_QUALITY_FRAME_DURATION = 1. / 15
MEDIUM_QUALITY_FRAME_DURATION = 1. / 30
PRODUCTION_QUALITY_FRAME_DURATION = 1. / 60

# There might be other configuration than pixel shape later...
PRODUCTION_QUALITY_CAMERA_CONFIG = {
    "pixel_height": 1440,
    "pixel_width": 2560,
}

HIGH_QUALITY_CAMERA_CONFIG = {
    "pixel_height": 1080,
    "pixel_width": 1920,
}

MEDIUM_QUALITY_CAMERA_CONFIG = {
    "pixel_height": 720,
    "pixel_width": 1280,
}

LOW_QUALITY_CAMERA_CONFIG = {
    "pixel_height": 480,
    "pixel_width": 854,
}

DEFAULT_PIXEL_HEIGHT = PRODUCTION_QUALITY_CAMERA_CONFIG["pixel_height"]
DEFAULT_PIXEL_WIDTH = PRODUCTION_QUALITY_CAMERA_CONFIG["pixel_width"]

DEFAULT_POINT_DENSITY_2D = 25
DEFAULT_POINT_DENSITY_1D = 250

DEFAULT_STROKE_WIDTH = 4

FRAME_HEIGHT = 8.0
FRAME_WIDTH = FRAME_HEIGHT * DEFAULT_PIXEL_WIDTH / DEFAULT_PIXEL_HEIGHT
FRAME_Y_RADIUS = FRAME_HEIGHT / 2
FRAME_X_RADIUS = FRAME_WIDTH / 2

SMALL_BUFF = 0.1
MED_SMALL_BUFF = 0.25
MED_LARGE_BUFF = 0.5
LARGE_BUFF = 1

DEFAULT_MOBJECT_TO_EDGE_BUFFER = MED_LARGE_BUFF
DEFAULT_MOBJECT_TO_MOBJECT_BUFFER = MED_SMALL_BUFF


# All in seconds
DEFAULT_ANIMATION_RUN_TIME = 1.0
DEFAULT_POINTWISE_FUNCTION_RUN_TIME = 3.0
DEFAULT_WAIT_TIME = 1.0


ORIGIN = np.array((0., 0., 0.))
UP = np.array((0., 1., 0.))
DOWN = np.array((0., -1., 0.))
RIGHT = np.array((1., 0., 0.))
LEFT = np.array((-1., 0., 0.))
IN = np.array((0., 0., -1.))
OUT = np.array((0., 0., 1.))
X_AXIS = np.array((1., 0., 0.))
Y_AXIS = np.array((0., 1., 0.))
Z_AXIS = np.array((0., 0., 1.))

# Useful abbreviations for diagonals
UL = UP + LEFT
UR = UP + RIGHT
DL = DOWN + LEFT
DR = DOWN + RIGHT

TOP = FRAME_Y_RADIUS * UP
BOTTOM = FRAME_Y_RADIUS * DOWN
LEFT_SIDE = FRAME_X_RADIUS * LEFT
RIGHT_SIDE = FRAME_X_RADIUS * RIGHT

PI = np.pi
TAU = 2 * PI
DEGREES = TAU / 360

ANIMATIONS_DIR = os.path.join(MEDIA_DIR, "animations")
RASTER_IMAGE_DIR = os.path.join(MEDIA_DIR, "designs", "raster_images")
SVG_IMAGE_DIR = os.path.join(MEDIA_DIR, "designs", "svg_images")
# TODO, staged scenes should really go into a subdirectory of a given scenes directory
STAGED_SCENES_DIR = os.path.join(ANIMATIONS_DIR, "staged_scenes")
###
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
FILE_DIR = os.path.join(THIS_DIR, "files")
TEX_DIR = os.path.join(FILE_DIR, "Tex")
TEX_IMAGE_DIR = TEX_DIR  # TODO, What is this doing?
# These two may be depricated now.
MOBJECT_DIR = os.path.join(FILE_DIR, "mobjects")
IMAGE_MOBJECT_DIR = os.path.join(MOBJECT_DIR, "image")

for folder in [FILE_DIR, RASTER_IMAGE_DIR, SVG_IMAGE_DIR, ANIMATIONS_DIR, TEX_DIR,
               TEX_IMAGE_DIR, MOBJECT_DIR, IMAGE_MOBJECT_DIR,
               STAGED_SCENES_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

TEX_USE_CTEX = False
TEX_TEXT_TO_REPLACE = "YourTextHere"
TEMPLATE_TEX_FILE = os.path.join(THIS_DIR, "tex_template.tex" if not TEX_USE_CTEX
    else "ctex_template.tex")
with open(TEMPLATE_TEX_FILE, "r") as infile:
    TEMPLATE_TEXT_FILE_BODY = infile.read()
    TEMPLATE_TEX_FILE_BODY = TEMPLATE_TEXT_FILE_BODY.replace(
        TEX_TEXT_TO_REPLACE,
        "\\begin{align*}" + TEX_TEXT_TO_REPLACE + "\\end{align*}",
    )

FFMPEG_BIN = "ffmpeg"


# Colors

COLOR_MAP = {
    "DARK_BLUE": "#236B8E",
    "DARK_BROWN": "#8B4513",
    "LIGHT_BROWN": "#CD853F",
    "BLUE_E": "#1C758A",
    "BLUE_D": "#29ABCA",
    "BLUE_C": "#58C4DD",
    "BLUE_B": "#9CDCEB",
    "BLUE_A": "#C7E9F1",
    "TEAL_E": "#49A88F",
    "TEAL_D": "#55C1A7",
    "TEAL_C": "#5CD0B3",
    "TEAL_B": "#76DDC0",
    "TEAL_A": "#ACEAD7",
    "GREEN_E": "#699C52",
    "GREEN_D": "#77B05D",
    "GREEN_C": "#83C167",
    "GREEN_B": "#A6CF8C",
    "GREEN_A": "#C9E2AE",
    "YELLOW_E": "#E8C11C",
    "YELLOW_D": "#F4D345",
    "YELLOW_C": "#FFFF00",
    "YELLOW_B": "#FFEA94",
    "YELLOW_A": "#FFF1B6",
    "GOLD_E": "#C78D46",
    "GOLD_D": "#E1A158",
    "GOLD_C": "#F0AC5F",
    "GOLD_B": "#F9B775",
    "GOLD_A": "#F7C797",
    "RED_E": "#CF5044",
    "RED_D": "#E65A4C",
    "RED_C": "#FC6255",
    "RED_B": "#FF8080",
    "RED_A": "#F7A1A3",
    "MAROON_E": "#94424F",
    "MAROON_D": "#A24D61",
    "MAROON_C": "#C55F73",
    "MAROON_B": "#EC92AB",
    "MAROON_A": "#ECABC1",
    "PURPLE_E": "#644172",
    "PURPLE_D": "#715582",
    "PURPLE_C": "#9A72AC",
    "PURPLE_B": "#B189C6",
    "PURPLE_A": "#CAA3E8",
    "WHITE": "#FFFFFF",
    "BLACK": "#000000",
    "LIGHT_GRAY": "#BBBBBB",
    "LIGHT_GREY": "#BBBBBB",
    "GRAY": "#888888",
    "GREY": "#888888",
    "DARK_GREY": "#444444",
    "DARK_GRAY": "#444444",
    "GREY_BROWN": "#736357",
    "PINK": "#D147BD",
    "GREEN_SCREEN": "#00FF00",
    "ORANGE": "#FF862F",
    "BRIGHT_RED": "#FF4C4C",
}
PALETTE = list(COLOR_MAP.values())
locals().update(COLOR_MAP)
for name in [s for s in list(COLOR_MAP.keys()) if s.endswith("_C")]:
    locals()[name.replace("_C", "")] = locals()[name]
