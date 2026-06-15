def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    real_env = env.unwrapped
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in real_env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in real_env.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]
