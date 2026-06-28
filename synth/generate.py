"""
Coffee Grind Synthetic Dataset Generator (Blender + Cycles)
============================================================

Photorealistic batch generator for ML training data, ported from the
Three.js prototype. Same bimodal log-normal distribution model, same
Fibonacci-spiral placement, but rendered with path-traced Cycles for
true photorealism (proper subsurface, caustics through glass, real
shadows, environment reflection on the silver table).

Usage:
    blender --background --python synth/generate.py

Or interactively: open the script in Blender's Text Editor and Run.

Output: For each sample, writes
    synth/images/grind_00000.png   (1024x1024 PNG)
    synth/images/grind_00000.json  (labels + parameters)

Tested on Blender 4.0+. Requires NumPy (bundled with Blender).
"""

import bpy
import bmesh
import math
import json
import os
import sys
from pathlib import Path
import numpy as np
from mathutils import Vector, Matrix, Euler

# ============================================================
# CONFIGURATION
# ============================================================

N_SAMPLES        = 5000           # Total images per run
UNIFORM_FINENESS = True           # Spread samples evenly across 0-100% fineness
N_BUCKETS        = 50             # Number of fineness buckets (each = 100/N_BUCKETS %)
MAX_RETRIES      = 1000           # Max attempts per bucket before skipping
RENDER_W         = 1024
RENDER_H         = 1024
CYCLES_SAMPLES   = 256            # 128 fast, 256 good, 512+ for hero shots
USE_DENOISING    = True
ENGINE           = 'CYCLES'       # 'CYCLES' or 'BLENDER_EEVEE' for ~10x speed, lower quality
USE_GPU          = True
GPU_BACKEND      = 'OPTIX'        # 'OPTIX' (NVIDIA RTX), 'CUDA' (older NVIDIA), 'HIP' (AMD), 'METAL' (Mac), 'NONE'
SEED             = int(os.environ.get('GRIND_SEED', 42))
SAVE_BLEND       = True           # save synth/images/scene_preview.blend after first sample

OUTPUT_DIR       = Path(__file__).parent / 'images'
OUTPUT_DIR.mkdir(exist_ok=True)

# Physical dimensions (cm — Blender's default unit is meters but we work in cm scaled down)
SCENE_SCALE      = 0.01           # 1 cm in our model = 0.01 m in Blender
CUP_INNER_R      = 3.95
CUP_HEIGHT       = 7.2
LAYER_FLOOR_Y    = 0.45
LAYER_TOP_Y      = 1.8
HARD_CAP_UM      = 1200

MAX_PARTICLES    = 800000         # cap; higher = slower mesh build but better detail

# ============================================================
# DISTRIBUTION MODEL  (identical math to the JS prototype)
# ============================================================

def pick_random_params(rng):
    return {
        'mainMedian':  float(np.exp(np.log(75) + rng.random() * (np.log(1100) - np.log(75)))),
        'mainSigma':   float(rng.uniform(0.32, 0.55)),
        'finesMedian': float(rng.uniform(15, 95)),
        'finesSigma':  float(rng.uniform(0.40, 0.65)),
        'finesFrac':   float(rng.random() * 0.50),
    }

def pick_params_for_bucket(lo, hi, rng):
    """Rejection-sample params until fineness_pct lands in [lo, hi]."""
    for _ in range(MAX_RETRIES):
        params = pick_random_params(rng)
        sizes  = sample_bimodal_fast(5000, params, rng)
        stats  = calc_stats(sizes)
        if lo <= stats['fineness_pct'] < hi:
            return params
    print(f"[warn] bucket {lo:.0f}-{hi:.0f}% not filled after {MAX_RETRIES} tries, using last sample")
    return params


def sample_bimodal_fast(n, params, rng):
    """Lightweight version for rejection sampling (no HARD_CAP clamping loop)."""
    mu_main  = math.log(params['mainMedian'])
    mu_fines = math.log(params['finesMedian'])
    sig_m    = params['mainSigma']
    sig_f    = params['finesSigma']
    frac     = params['finesFrac']
    choose   = rng.random(n)
    z        = rng.standard_normal(n)
    sizes    = np.where(choose < frac,
                        np.exp(mu_fines + sig_f * z),
                        np.exp(mu_main  + sig_m * z))
    return np.clip(sizes, 3, HARD_CAP_UM)


def sample_bimodal(n, params, rng):
    """Bimodal log-normal mixture with rejection sampling above HARD_CAP_UM."""
    mu_main  = math.log(params['mainMedian'])
    mu_fines = math.log(params['finesMedian'])
    sig_m    = params['mainSigma']
    sig_f    = params['finesSigma']
    frac     = params['finesFrac']

    choose = rng.random(n)
    z      = rng.standard_normal(n)
    main   = np.exp(mu_main  + sig_m * z)
    fines  = np.exp(mu_fines + sig_f * z)
    sizes  = np.where(choose < frac, fines, main)

    over = sizes > HARD_CAP_UM
    n_over = int(over.sum())
    if n_over > 0:
        sizes[over] = params['mainMedian'] + rng.random(n_over) * (HARD_CAP_UM - params['mainMedian'])
    np.maximum(sizes, 3, out=sizes)
    return sizes

def calc_stats(sizes):
    sV = sizes ** 3
    totV = sV.sum()
    f = sizes < 355
    m = (sizes >= 355) & (sizes <= 710)
    c = sizes > 710
    return {
        'x10': float(np.percentile(sizes, 10)),
        'x50': float(np.percentile(sizes, 50)),
        'x90': float(np.percentile(sizes, 90)),
        'fineness_pct': float(sV[f].sum() / totV * 100),
        'mid_pct':      float(sV[m].sum() / totV * 100),
        'coarse_pct':   float(sV[c].sum() / totV * 100),
    }

def choose_rendering_params(sizes):
    """Adaptive visual scale and N based on volume-mean particle size."""
    vol_mean = ((sizes ** 3).mean()) ** (1/3)
    t = max(0.0, min(1.0, (vol_mean - 100) / 900))
    visual_scale = 5.5 - t * 2.8

    eff_cm = vol_mean * 1e-4 * visual_scale
    part_vol = 2.3 * eff_cm ** 3
    proj_area = math.pi * 1.32 * eff_cm ** 2

    floor_area = math.pi * CUP_INNER_R ** 2
    layer_vol  = floor_area * (LAYER_TOP_Y - LAYER_FLOOR_Y)

    n_vol = layer_vol * 0.65 / part_vol
    n_cov = 160 * floor_area / proj_area
    n = int(round(max(n_vol, n_cov)))
    return visual_scale, max(2000, min(MAX_PARTICLES, n))

# ============================================================
# FIBONACCI SPIRAL PARTICLE PLACEMENT
# ============================================================

def place_particles(n, sizes_um, visual_scale, rng):
    golden_angle = math.pi * (3 - math.sqrt(5))
    max_rho = CUP_INNER_R - 0.08
    layer_h = LAYER_TOP_Y - LAYER_FLOOR_Y
    spacing = math.sqrt(math.pi * max_rho ** 2 / n)

    sizes_cm = sizes_um * 1e-4 * visual_scale

    ax = 0.50 + rng.random(n) * 0.50
    ay = 0.20 + rng.random(n) * 0.20
    az = 0.50 + rng.random(n) * 0.50

    rx = sizes_cm * ax
    ry_vert = sizes_cm * ay
    rz = sizes_cm * az

    idx = np.arange(n)
    t = (idx + 0.5) / n
    base_rho = np.sqrt(t) * (max_rho - rx * 0.5)
    base_ang = idx * golden_angle

    jit_r = (rng.random(n) - 0.5) * spacing * 5.5
    jit_a = (rng.random(n) - 0.5) * 0.18
    rho = np.clip(base_rho + jit_r, 0, max_rho - rx)
    ang = base_ang + jit_a

    x = rho * np.cos(ang)
    z_horiz = rho * np.sin(ang)
    yT = rng.random(n) ** 0.55
    y_vert = LAYER_FLOOR_Y + ry_vert + yT * layer_h

    positions = np.column_stack([x, z_horiz, y_vert]) * SCENE_SCALE
    scales    = np.column_stack([rx, rz, ry_vert]) * SCENE_SCALE
    rotations = rng.random((n, 3)) * math.pi * 2
    rotations[:, 2] *= 0.5
    return positions, scales, rotations

# ============================================================
# BLENDER SCENE BUILD
# ============================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for mat in list(bpy.data.materials):
        if mat.users == 0:
            bpy.data.materials.remove(mat)

    for light in list(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)

    for cam in list(bpy.data.cameras):
        if cam.users == 0:
            bpy.data.cameras.remove(cam)

def setup_world_environment():
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out  = nt.nodes.new('ShaderNodeOutputWorld');   out.location = (400, 0)
    bg   = nt.nodes.new('ShaderNodeBackground');    bg.location  = (200, 0)
    mix  = nt.nodes.new('ShaderNodeMix');           mix.location = (-100, 0); mix.data_type = 'RGBA'; mix.blend_type = 'MIX'
    grad = nt.nodes.new('ShaderNodeTexGradient');   grad.location = (-400, 0); grad.gradient_type = 'LINEAR'
    map  = nt.nodes.new('ShaderNodeMapping');       map.location = (-650, 0); map.inputs['Rotation'].default_value = (math.radians(90), 0, 0)
    tex  = nt.nodes.new('ShaderNodeTexCoord');      tex.location = (-900, 0)

    mix.inputs[6].default_value = (0.04, 0.03, 0.02, 1.0)
    mix.inputs[7].default_value = (0.95, 0.88, 0.72, 1.0)

    nt.links.new(tex.outputs['Generated'], map.inputs['Vector'])
    nt.links.new(map.outputs['Vector'],    grad.inputs['Vector'])
    nt.links.new(grad.outputs['Fac'],      mix.inputs[0])
    nt.links.new(mix.outputs[2],           bg.inputs['Color'])
    nt.links.new(bg.outputs['Background'], out.inputs['Surface'])

    bg.inputs['Strength'].default_value = 1.2

def setup_camera():
    cam_data = bpy.data.cameras.new('Camera')
    cam_data.lens = 65
    cam_data.sensor_width = 36

    cam = bpy.data.objects.new('Camera', cam_data)
    bpy.context.scene.collection.objects.link(cam)

    cam.location = (0.0, 0, 0.18)
    cam_data.type="ORTHO"
    cam_data.ortho_scale = 0.14

    target = Vector((
        0.0,
        0.0,
        ((LAYER_FLOOR_Y + LAYER_TOP_Y) * 0.5) * SCENE_SCALE
    ))

    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.scene.camera = cam

def setup_lights():
    key = bpy.data.lights.new('Key', 'AREA')
    key.energy = 1.0
    key.size = 0.4
    key.color = (1.0, 0.95, 0.85)
    keo = bpy.data.objects.new('Key', key)
    keo.location = (0.18, -0.12, 0.32)
    keo.rotation_euler = (math.radians(35), 0, math.radians(35))
    bpy.context.scene.collection.objects.link(keo)

    fil = bpy.data.lights.new('Fill', 'AREA')
    fil.energy = 0.6
    fil.size = 0.6
    fil.color = (0.78, 0.85, 1.0)
    flo = bpy.data.objects.new('Fill', fil)
    flo.location = (-0.15, 0.10, 0.18)
    flo.rotation_euler = (math.radians(45), 0, math.radians(-120))
    bpy.context.scene.collection.objects.link(flo)

def create_brushed_silver_material():
    mat = bpy.data.materials.new('silver_brushed')
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()

    out  = nt.nodes.new('ShaderNodeOutputMaterial'); out.location  = (640, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (340, 0)
    bsdf.inputs['Base Color'].default_value           = (0.385, 0.385, 0.383, 1.0)
    bsdf.inputs['Metallic'].default_value             = 1.0
    bsdf.inputs['Roughness'].default_value            = 0.25
    bsdf.inputs['Anisotropic'].default_value          = 0.6
    bsdf.inputs['Anisotropic Rotation'].default_value = 0.0

    noise   = nt.nodes.new('ShaderNodeTexNoise');  noise.location   = (-220, 220)
    noise.inputs['Scale'].default_value      = 120
    noise.inputs['Detail'].default_value     = 4
    noise.inputs['Distortion'].default_value = 0.15

    mapping = nt.nodes.new('ShaderNodeMapping');   mapping.location = (-440, 220)
    mapping.inputs['Scale'].default_value = (1.0, 40.0, 1.0)
    coord   = nt.nodes.new('ShaderNodeTexCoord'); coord.location   = (-640, 220)

    remap   = nt.nodes.new('ShaderNodeMapRange');  remap.location   = (80, 220)
    remap.inputs['From Min'].default_value = 0.0
    remap.inputs['From Max'].default_value = 1.0
    remap.inputs['To Min'].default_value   = 0.15
    remap.inputs['To Max'].default_value   = 0.35

    nt.links.new(coord.outputs['Generated'],  mapping.inputs['Vector'])
    nt.links.new(mapping.outputs['Vector'],   noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'],        remap.inputs['Value'])
    nt.links.new(remap.outputs['Result'],     bsdf.inputs['Roughness'])
    nt.links.new(bsdf.outputs['BSDF'],        out.inputs['Surface'])
    return mat

def create_table():
    bpy.ops.mesh.primitive_circle_add(radius=0.50, vertices=128, fill_type='NGON', location=(0, 0, 0))
    table = bpy.context.object
    table.name = 'silver_table'
    table.data.materials.append(create_brushed_silver_material())

def create_glass_material():
    mat = bpy.data.materials.new('glass')
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (300, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs['Metallic'].default_value = 0.0
    bsdf.inputs['Roughness'].default_value = 0.03
    bsdf.inputs['Transmission Weight'].default_value = 1.0
    bsdf.inputs['IOR'].default_value = 1.51
    bsdf.inputs['Coat Weight'].default_value = 0.3
    bsdf.inputs['Coat Roughness'].default_value = 0.05
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat

def create_glass_beaker():
    profile = [
        (0.000, 0.000), (0.0415, 0.000), (0.0430, 0.0012),
        (0.0425, 0.0030), (0.0435, (CUP_HEIGHT - 0.35) * SCENE_SCALE),
        (0.0455, (CUP_HEIGHT - 0.08) * SCENE_SCALE),
        (0.0455, CUP_HEIGHT * SCENE_SCALE),
        (0.0435, CUP_HEIGHT * SCENE_SCALE),
        (0.0415, (CUP_HEIGHT - 0.20) * SCENE_SCALE),
        (0.0405, 0.0035),
        (0.000, LAYER_FLOOR_Y * SCENE_SCALE)
    ]
    segments = 128
    bm = bmesh.new()
    rings = []
    for r, h in profile:
        ring = []
        for s in range(segments):
            a = 2 * math.pi * s / segments
            v = bm.verts.new((r * math.cos(a), r * math.sin(a), h))
            ring.append(v)
        rings.append(ring)
    bm.verts.ensure_lookup_table()
    for i in range(len(rings) - 1):
        for s in range(segments):
            ns = (s + 1) % segments
            try:
                bm.faces.new([rings[i][s], rings[i][ns], rings[i+1][ns], rings[i+1][s]])
            except ValueError:
                pass
    bm.normal_update()

    mesh = bpy.data.meshes.new('glass_beaker')
    bm.to_mesh(mesh); bm.free()
    obj = bpy.data.objects.new('glass_beaker', mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(create_glass_material())
    return obj

def create_dust_layer():
    bpy.ops.mesh.primitive_circle_add(
        radius=CUP_INNER_R * SCENE_SCALE,
        vertices=64, fill_type='NGON',
        location=(0, 0, LAYER_FLOOR_Y * SCENE_SCALE)
    )
    dust = bpy.context.object
    dust.name = 'dust_layer'
    mat = bpy.data.materials.new('dust')
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    out  = nt.nodes.new('ShaderNodeOutputMaterial'); out.location  = (300, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (0.035, 0.016, 0.006, 1.0)
    bsdf.inputs['Roughness'].default_value  = 0.97
    bsdf.inputs['Metallic'].default_value   = 0.0
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    dust.data.materials.append(mat)

def create_dust_veil():
    layers = [
        (0.50, 0.46),
        (0.75, 0.42),
        (1.00, 0.38),
        (1.30, 0.33),
        (1.60, 0.28),
        (2.20, 0.21),
        (2.80, 0.15),
        (3.45, 0.09),
    ]
    for i, (height_cm, max_alpha) in enumerate(layers):
        bpy.ops.mesh.primitive_circle_add(
            radius=CUP_INNER_R * SCENE_SCALE,
            vertices=64, fill_type='NGON',
            location=(0, 0, height_cm * SCENE_SCALE)
        )
        veil = bpy.context.object
        veil.name = f'dust_veil_{i}'
        mat = bpy.data.materials.new(f'dust_veil_{i}')
        mat.use_nodes = True
        nt = mat.node_tree; nt.nodes.clear()

        out  = nt.nodes.new('ShaderNodeOutputMaterial'); out.location  = (500, 0)
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (200, 0)
        bsdf.inputs['Base Color'].default_value = (0.045, 0.020, 0.008, 1.0)
        bsdf.inputs['Roughness'].default_value  = 0.97
        bsdf.inputs['Metallic'].default_value   = 0.0

        noise = nt.nodes.new('ShaderNodeTexNoise'); noise.location = (-400, 0)
        noise.inputs['Scale'].default_value      = 180
        noise.inputs['Detail'].default_value     = 7
        noise.inputs['Roughness'].default_value  = 0.75
        noise.inputs['Distortion'].default_value = 0.4

        remap = nt.nodes.new('ShaderNodeMapRange'); remap.location = (-100, 0)
        remap.inputs['From Min'].default_value = 0.25
        remap.inputs['From Max'].default_value = 0.75
        remap.inputs['To Min'].default_value   = 0.0
        remap.inputs['To Max'].default_value   = max_alpha

        nt.links.new(noise.outputs['Fac'],    remap.inputs['Value'])
        nt.links.new(remap.outputs['Result'], bsdf.inputs['Alpha'])
        nt.links.new(bsdf.outputs['BSDF'],    out.inputs['Surface'])
        veil.data.materials.append(mat)
        try:
            mat.blend_method = 'BLEND'
        except AttributeError:
            pass

# ============================================================
# COFFEE PARTICLES
# ============================================================

def create_coffee_material():
    mat = bpy.data.materials.new('coffee')
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()

    out  = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (860, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (520, 0)

    bsdf.inputs['Roughness'].default_value         = 0.93
    bsdf.inputs['Metallic'].default_value          = 0.0
    bsdf.inputs['Subsurface Weight'].default_value = 0.06
    bsdf.inputs['Subsurface Radius'].default_value = (0.0012, 0.0007, 0.0003)

    noise_col = nt.nodes.new('ShaderNodeTexNoise'); noise_col.location = (-560, 0)
    noise_col.inputs['Scale'].default_value      = 70
    noise_col.inputs['Detail'].default_value     = 3
    noise_col.inputs['Roughness'].default_value  = 0.6
    noise_col.inputs['Distortion'].default_value = 0.3

    cramp = nt.nodes.new('ShaderNodeValToRGB'); cramp.location = (-300, 0)
    cramp.color_ramp.elements[0].position = 0.2
    cramp.color_ramp.elements[1].position = 0.8
    cramp.color_ramp.elements[0].color = (0.020, 0.008, 0.003, 1.0)
    cramp.color_ramp.elements[1].color = (0.080, 0.038, 0.014, 1.0)

    nt.links.new(noise_col.outputs['Fac'],  cramp.inputs['Fac'])

    ao_node = nt.nodes.new('ShaderNodeAmbientOcclusion'); ao_node.location = (-560, -180)
    ao_node.samples  = 4
    ao_node.inputs['Distance'].default_value = 0.004

    inv = nt.nodes.new('ShaderNodeMath'); inv.location = (-320, -180)
    inv.operation = 'SUBTRACT'; inv.use_clamp = True
    inv.inputs[0].default_value = 1.0
    nt.links.new(ao_node.outputs['AO'], inv.inputs[1])

    ao_mix = nt.nodes.new('ShaderNodeMix'); ao_mix.location = (-80, 0)
    ao_mix.data_type = 'RGBA'; ao_mix.blend_type = 'MIX'
    ao_mix.inputs[7].default_value = (0.16, 0.082, 0.032, 1.0)
    nt.links.new(cramp.outputs['Color'],   ao_mix.inputs[6])
    nt.links.new(inv.outputs['Value'],     ao_mix.inputs[0])
    nt.links.new(ao_mix.outputs[2],        bsdf.inputs['Base Color'])

    voronoi = nt.nodes.new('ShaderNodeTexVoronoi'); voronoi.location = (-180, 220)
    voronoi.feature = 'DISTANCE_TO_EDGE'
    voronoi.inputs['Scale'].default_value       = 300
    voronoi.inputs['Randomness'].default_value  = 1.0

    bump = nt.nodes.new('ShaderNodeBump');          bump.location = (220, 220)
    bump.inputs['Strength'].default_value = 1.2
    bump.inputs['Distance'].default_value = 0.0006

    nt.links.new(voronoi.outputs['Distance'], bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'],      bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'],      out.inputs['Surface'])
    return mat

def create_light_coffee_material():
    mat = bpy.data.materials.new('coffee_light')
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()

    out  = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (640, 0)
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (300, 0)
    bsdf.inputs['Roughness'].default_value         = 0.90
    bsdf.inputs['Metallic'].default_value          = 0.0
    bsdf.inputs['Subsurface Weight'].default_value = 0.04
    bsdf.inputs['Subsurface Radius'].default_value = (0.002, 0.0015, 0.001)

    noise_col = nt.nodes.new('ShaderNodeTexNoise'); noise_col.location = (-400, 0)
    noise_col.inputs['Scale'].default_value      = 70
    noise_col.inputs['Detail'].default_value     = 3
    noise_col.inputs['Roughness'].default_value  = 0.6
    noise_col.inputs['Distortion'].default_value = 0.3

    cramp = nt.nodes.new('ShaderNodeValToRGB'); cramp.location = (-140, 0)
    cramp.color_ramp.elements[0].position = 0.2
    cramp.color_ramp.elements[1].position = 0.8
    cramp.color_ramp.elements[0].color = (0.055, 0.030, 0.012, 1.0)
    cramp.color_ramp.elements[1].color = (0.115, 0.062, 0.026, 1.0)

    nt.links.new(noise_col.outputs['Fac'], cramp.inputs['Fac'])
    nt.links.new(cramp.outputs['Color'],   bsdf.inputs['Base Color'])
    nt.links.new(bsdf.outputs['BSDF'],     out.inputs['Surface'])
    return mat

def create_particle_mesh(positions, scales, rotations):
    n = len(positions)
    N_TEMPLATES = 8

    rng_t = np.random.default_rng(SEED + 17)
    templates = []
    for t in range(N_TEMPLATES):
        tbm = bmesh.new()
        bmesh.ops.create_icosphere(tbm, subdivisions=1, radius=1.0)
        for v in tbm.verts:
            jitter = rng_t.uniform(-0.26, 0.26, 3)
            v.co.x *= 1.0 + jitter[0]; v.co.y *= 1.0 + jitter[1]; v.co.z *= 1.0 + jitter[2]
        tbm.normal_update()
        tverts = [v.co.copy() for v in tbm.verts]
        tfaces = [[v.index for v in f.verts] for f in tbm.faces]
        tbm.free()
        templates.append((np.array(tverts), tfaces))

    n_v = len(templates[0][0])
    n_f = len(templates[0][1])
    verts_per_face = len(templates[0][1][0])

    template_ids = rng_t.integers(0, N_TEMPLATES, size=n)

    total_verts = n * n_v
    total_faces = n * n_f
    all_verts = np.empty((total_verts, 3), dtype=np.float32)
    all_faces = np.empty((total_faces, verts_per_face), dtype=np.int32)

    for i in range(n):
        template_arr, template_faces = templates[template_ids[i]]
        v = template_arr * scales[i]
        m = Euler(rotations[i].tolist(), 'XYZ').to_matrix()
        v = v @ np.array(m).T
        v += positions[i]
        all_verts[i * n_v:(i + 1) * n_v] = v
        for j, face in enumerate(template_faces):
            all_faces[i * n_f + j] = [k + i * n_v for k in face]

    mesh = bpy.data.meshes.new('coffee_particles')
    mesh.from_pydata(all_verts.tolist(), [], all_faces.tolist())
    mesh.update()
    mesh.materials.append(create_coffee_material())
    mesh.materials.append(create_light_coffee_material())

    tan_frac = rng_t.uniform(0.004, 0.05)
    tan_mask = rng_t.random(n) < tan_frac
    mat_indices = np.zeros(total_faces, dtype=np.int32)
    for i in np.where(tan_mask)[0]:
        mat_indices[i * n_f:(i + 1) * n_f] = 1
    mesh.polygons.foreach_set('material_index', mat_indices.tolist())

    mesh.polygons.foreach_set('use_smooth', [True] * len(mesh.polygons))

    obj = bpy.data.objects.new('coffee_particles', mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def remove_coffee_particles():
    obj = bpy.data.objects.get('coffee_particles')
    if obj is not None:
        m = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(m)

# ============================================================
# RENDER CONFIG
# ============================================================

def setup_render():
    scene = bpy.context.scene
    scene.render.engine = ENGINE
    scene.render.resolution_x = RENDER_W
    scene.render.resolution_y = RENDER_H
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    for _vt in ('AgX', 'Filmic', 'Raw'):
        try:
            scene.view_settings.view_transform = _vt
            break
        except Exception:
            continue

    for _look in ('AgX - Medium High Contrast', 'Medium High Contrast', 'None'):
        try:
            scene.view_settings.look = _look
            break
        except Exception:
            continue
    scene.view_settings.exposure = 0.9

    if ENGINE == 'CYCLES':
        scene.cycles.samples = CYCLES_SAMPLES
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01

        scene.cycles.use_denoising = USE_DENOISING
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'

        scene.cycles.max_bounces          = 12
        scene.cycles.diffuse_bounces      = 4
        scene.cycles.glossy_bounces       = 8
        scene.cycles.transmission_bounces = 16
        scene.cycles.volume_bounces       = 0
        scene.cycles.transparent_max_bounces = 8

        scene.cycles.caustics_reflective = True
        scene.cycles.caustics_refractive = True

        scene.cycles.sample_clamp_direct   = 0
        scene.cycles.sample_clamp_indirect = 10

        try:
            scene.cycles.use_light_tree = True
        except Exception:
            pass

        if USE_GPU and GPU_BACKEND != 'NONE':
            scene.cycles.device = 'GPU'
            prefs = bpy.context.preferences.addons['cycles'].preferences
            try:
                prefs.compute_device_type = GPU_BACKEND
                try:
                    prefs.get_devices(compute_device_type=GPU_BACKEND)
                except TypeError:
                    prefs.get_devices()
                gpu_found = False
                for d in prefs.devices:
                    if d.type == GPU_BACKEND:
                        d.use = True
                        gpu_found = True
                        print(f"[GPU] enabled: {d.name}")
                    else:
                        d.use = False
                if not gpu_found:
                    print(f"[GPU] no {GPU_BACKEND} device found — falling back to CPU.")
                    scene.cycles.device = 'CPU'
            except Exception as e:
                print(f"[GPU setup error] {e}; falling back to CPU.")
                scene.cycles.device = 'CPU'

def render_and_save(idx, params, sizes, stats, visual_scale):
    fname = f"grind_{idx:05d}"
    scene = bpy.context.scene
    scene.render.filepath = str(OUTPUT_DIR / f"{fname}.png")
    bpy.ops.render.render(write_still=True)

    meta = {
        'filename': f"{fname}.png",
        'params': params,
        'stats': stats,
        'n_particles': int(len(sizes)),
        'visual_scale': float(visual_scale),
        'hard_cap_um': HARD_CAP_UM,
        'placement': 'fibonacci_spiral',
        'model': 'bimodal_lognormal_mixture',
        'render_engine': ENGINE,
        'cycles_samples': CYCLES_SAMPLES if ENGINE == 'CYCLES' else None,
    }
    with open(OUTPUT_DIR / f"{fname}.json", 'w') as f:
        json.dump(meta, f, indent=2)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    rng = np.random.default_rng(SEED)
    print(f"[init] seed={SEED}, samples={N_SAMPLES}, engine={ENGINE}")

    clear_scene()
    setup_world_environment()
    setup_camera()
    setup_lights()
    create_table()
    create_dust_layer()
    create_dust_veil()
    create_glass_beaker()
    setup_render()

    if UNIFORM_FINENESS:
        bucket_edges = np.linspace(0, 100, N_BUCKETS + 1)
        assignments  = rng.permutation(N_SAMPLES) % N_BUCKETS
        print(f"[init] UNIFORM_FINENESS: {N_BUCKETS} buckets ({100/N_BUCKETS:.1f}% each), {N_SAMPLES} samples")

    for i in range(N_SAMPLES):
        remove_coffee_particles()
        if UNIFORM_FINENESS:
            b  = int(assignments[i])
            lo, hi = float(bucket_edges[b]), float(bucket_edges[b + 1])
            params = pick_params_for_bucket(lo, hi, rng)
        else:
            params = pick_random_params(rng)
        sizes_t = sample_bimodal(8000, params, rng)
        visual_scale, n = choose_rendering_params(sizes_t)
        sizes = sample_bimodal(n, params, rng)
        stats = calc_stats(sizes)
        positions, scales, rotations = place_particles(n, sizes, visual_scale, rng)
        create_particle_mesh(positions, scales, rotations)

        if SAVE_BLEND and i == 0:
            blend_path = str(OUTPUT_DIR / 'scene_preview.blend')
            bpy.ops.wm.save_as_mainfile(filepath=blend_path)
            print(f"[blend] scene saved -> {blend_path}")

        bucket_str = f" bucket={lo:.0f}-{hi:.0f}%" if UNIFORM_FINENESS else ""
        print(f"[{i+1}/{N_SAMPLES}]{bucket_str} x50={stats['x50']:.0f} fineness={stats['fineness_pct']:.1f}% N={n} vs={visual_scale:.2f}")
        render_and_save(i, params, sizes, stats, visual_scale)

    print(f"[done] {N_SAMPLES} samples written to {OUTPUT_DIR.resolve()}")

if __name__ == '__main__':
    main()
