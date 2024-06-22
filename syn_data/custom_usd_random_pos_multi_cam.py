import omni.replicator.core as rep
with rep.new_layer():
    # Load in asset
    local_path = "/media/yobi/hugeDrive/nvidia-omniverse/synthetic_data_with_nvidia_replicator_and_edge_impulse/"
    furniture_USD = f"{local_path}/asset/Collected_Furniture/Furniture.usd"
    apple_USD = f"{local_path}/asset/Collected_SM_Apple/SM_Apple.usd"
  # Camera paramters
    cam_position = (20, 200, 78)
    cam_position2 = (46, 120, 25)
    cam_position_random = rep.distribution.uniform((0, 181, 0), (0, 300, 0))
    cam_rotation = (-60, 0, 0)
    focus_distance = 114
    focus_distance2 = 39.1
    focal_length = 27
    focal_length2 = 18.5
    f_stop = 1.8
    f_stop2 = 1.8
    focus_distance_random = rep.distribution.normal(500.0, 100)
    # Cultery path
    current_obj = apple_USD  
    output_path = current_obj.split(".")[0].split("/")[-1]
    def spherical_lights(num=1):
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(100, 8000),
            position=(45, 110, 0),
            rotation=(-90, 0, 0),
            scale=100,
            count=num
        )
        return lights.node
    def dome_shaped_lights(num=3):
        lights = rep.create.light(
            light_type="dome",
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(100, 8000),
            position=(45, 120, 18),
            rotation=(225, 0, 0),
            scale = 100,
            count=num
        )
        return lights.node
    def furniture():
        furniture = rep.create.from_usd(furniture_USD, semantics=[('class', 'furniture')])
        with furniture:
            rep.modify.pose(
                position=(20, 31, 12.3),
                rotation=(0, 0, 0),
            )
        return furniture
    # Define randomizer function for Apple assets. This randomization includes placement and rotation of the assets on the surface.
    def obj_props(size=15):
        instances = rep.randomizer.instantiate(rep.utils.get_usd_files(
            current_obj), size=size, mode='point_instance')
        with instances:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (0, 76.3651, 0), (40, 76.3651, 24)),
                rotation=rep.distribution.uniform(
                    (-90, -180, 0), (-90, 180, 0)),
            )
        return instances.node
    # Register randomization
    rep.randomizer.register(furniture)
    rep.randomizer.register(obj_props)
    rep.randomizer.register(spherical_lights)
    rep.randomizer.register(dome_shaped_lights)
    # Multiple setup cameras and attach it to render products
    camera = rep.create.camera(focus_distance=focus_distance, focal_length=focal_length,
                               position=cam_position, rotation=cam_rotation, f_stop=f_stop)
    camera2 = rep.create.camera(focus_distance=focus_distance2, focal_length=focal_length2,
                                position=cam_position2, rotation=cam_rotation, f_stop=f_stop)
    # Will render 1024x1024 images and 512x512 images
    render_product = rep.create.render_product(camera, (1024, 1024))
    render_product2 = rep.create.render_product(camera2, (512, 512))
    # Initialize and attach writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=f"{local_path}/data/angled_60_sphere/{output_path}",
                      rgb=True, bounding_box_2d_tight=True, semantic_segmentation=True)
    writer.attach([render_product, render_product2])
    with rep.trigger.on_frame(num_frames=250, interval=10):
        rep.randomizer.furniture()
        rep.randomizer.spherical_lights(1)
        rep.randomizer.dome_shaped_lights(1)
        rep.randomizer.obj_props(5)
    # Run the simulation graph
    rep.orchestrator.run()
