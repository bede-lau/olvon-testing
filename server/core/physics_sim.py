"""
Blender physics simulation script.
NOT importable as a module - runs via: blender --background --python physics_sim.py -- <args>

Usage:
    blender --background --python server/core/physics_sim.py -- \
        --body body.obj --garment garment.obj --output result.glb --frames 40
"""

import sys


def main():
    import bpy
    import argparse

    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        print("ERROR: No arguments found. Use -- to separate Blender args from script args.", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Cloth physics simulation in Blender")
    parser.add_argument("--body", required=True, help="Path to body OBJ mesh")
    parser.add_argument("--garment", required=True, help="Path to garment OBJ mesh")
    parser.add_argument("--output", required=True, help="Output GLB file path")
    parser.add_argument("--frames", type=int, default=40, help="Number of simulation frames")
    args = parser.parse_args(argv)

    try:
        # Step 1: Clear the default scene
        print("[physics_sim] Clearing scene...")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete()

        # Step 2: Import body mesh
        print(f"[physics_sim] Importing body: {args.body}")
        bpy.ops.wm.obj_import(filepath=args.body)
        body_obj = bpy.context.selected_objects[0]
        body_obj.name = "Body"

        # Step 3: Add Collision modifier to body
        print("[physics_sim] Adding Collision modifier to body...")
        bpy.context.view_layer.objects.active = body_obj
        bpy.ops.object.modifier_add(type="COLLISION")
        body_obj.collision.thickness_outer = 0.02
        body_obj.collision.thickness_inner = 0.01

        # Step 4: Import garment mesh
        print(f"[physics_sim] Importing garment: {args.garment}")
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.wm.obj_import(filepath=args.garment)
        garment_obj = bpy.context.selected_objects[0]
        garment_obj.name = "Garment"

        # Step 5: Add Cloth modifier to garment (cotton preset)
        print("[physics_sim] Adding Cloth modifier to garment...")
        bpy.context.view_layer.objects.active = garment_obj
        bpy.ops.object.modifier_add(type="CLOTH")
        cloth = garment_obj.modifiers["Cloth"]
        cloth.settings.mass = 0.3
        cloth.settings.quality = 5
        cloth.collision_settings.use_self_collision = True
        cloth.collision_settings.self_distance_min = 0.005

        # Step 6: Set frame range and bake
        print(f"[physics_sim] Baking {args.frames} frames...")
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_end = args.frames
        cloth.point_cache.frame_start = 1
        cloth.point_cache.frame_end = args.frames

        # Bake the simulation
        override = bpy.context.copy()
        override["point_cache"] = cloth.point_cache
        with bpy.context.temp_override(**override):
            bpy.ops.ptcache.bake(bake=True)

        # Step 7: Go to last frame and apply modifier
        print("[physics_sim] Applying cloth modifier at final frame...")
        scene.frame_set(args.frames)
        bpy.context.view_layer.objects.active = garment_obj
        bpy.ops.object.modifier_apply(modifier="Cloth")

        # Step 8: Export as GLB
        print(f"[physics_sim] Exporting to {args.output}")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.export_scene.gltf(
            filepath=args.output,
            export_format="GLB",
            use_selection=True,
        )

        print("[physics_sim] Done! Output:", args.output)

    except Exception as e:
        print(f"[physics_sim] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
