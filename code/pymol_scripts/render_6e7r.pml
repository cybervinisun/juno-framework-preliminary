# PyMOL script: 6E7R (hybrid ligand 93-4) redocking pose overlay
#
# NOTE: the raw GOLD .mol2 outputs (protein + poses) are NOT part of
# this repository (large, idiosyncratic per-system GOLD output, not
# redistributed here). Point GOLD_RAW_DIR at your own copy of the raw
# redocking folders to reproduce this figure; the pre-rendered PNG is
# already provided under results/ for readers who just want the image.
#
# Run with: pymol -cq render_6e7r.pml
python
import os
GOLD_RAW_DIR = os.environ.get(
    "GOLD_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "raw", "gold_redocking_raw"),
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "regenerated", "figuras_pymol"),
)
os.makedirs(OUT_DIR, exist_ok=True)
SYS_DIR = os.path.join(GOLD_RAW_DIR, "6e7r", "redocagem")
POSE_DIR = os.path.join(SYS_DIR, "goldscore", "gold_(N1000)")
cmd.load(os.path.join(POSE_DIR, "gold_protein.mol2"), "prot")
cmd.load(os.path.join(SYS_DIR, "93_4_exp_6e7r.mol2"), "crystal")
cmd.load(os.path.join(POSE_DIR, "gold_soln_93_4_prot_exp_m1_503.mol2"), "pose_a")
cmd.load(os.path.join(POSE_DIR, "gold_soln_93_4_prot_exp_m1_919.mol2"), "pose_b")
cmd.load(os.path.join(POSE_DIR, "gold_soln_93_4_prot_exp_m1_771.mol2"), "pose_c")
cmd.load(os.path.join(POSE_DIR, "gold_soln_93_4_prot_exp_m1_497.mol2"), "pose_d")
python end

bg_color white
set ray_opaque_background, 1
set antialias, 2
set ray_trace_mode, 1
set ray_trace_color, grey30
set ray_shadows, 0
set specular, 0.1
set depth_cue, 0
set stick_quality, 15
set line_smooth, 1
set ray_trace_gain, 0

hide everything
remove hydro

select pocket, byres (prot within 4.0 of (crystal or pose_a or pose_b or pose_c or pose_d))
show lines, pocket
color grey80, pocket
set line_width, 1.2

show sticks, crystal
color black, crystal
set stick_radius, 0.30, crystal

show sticks, pose_a
color marine, pose_a
set stick_radius, 0.16, pose_a

show sticks, pose_b
color purple, pose_b
set stick_radius, 0.16, pose_b

show sticks, pose_c
color deeppink, pose_c
set stick_radius, 0.16, pose_c

show sticks, pose_d
color yellow, pose_d
set stick_radius, 0.16, pose_d

orient crystal or pose_a or pose_b or pose_c or pose_d
zoom crystal or pose_a or pose_b or pose_c or pose_d, buffer=2.5

ray 2000, 1800
python
cmd.png(os.path.join(OUT_DIR, "6e7r_pymol_v2.png"), dpi=300)
python end

quit
