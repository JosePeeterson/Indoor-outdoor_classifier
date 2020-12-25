import os

num_box = 4
scene_type_list = ['scene_l', 'scene_straight', 'scene_open', 'scene_plus']
scene_list = ['Perpendicular', 'Towards', 'Parallel', 'Circular']
world_list = ['l-corridor', 'corridor', 'open', 'plus-corridor']
goal_dir_list = ['adjacent', 'straight']


def write_to_file(writer, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart, goal_direction):
    writer.write("scene:\n")
    writer.write("  world: %s\n" % world)
    writer.write("  goal_direction: %s\n" % goal_direction)
    writer.write("  type: %s\n" % scn_type)
    writer.write("  use_static_obstacles: %s\n" % static_ob)
    writer.write("  dual_direction: %s\n" % dual)
    writer.write("  num_dynamic: %d\n" % dynamic_box)
    writer.write("  num_box: %d\n" % total_box)
    writer.write("  reverse_collision: %s\n" % reverse)
    writer.write("  restart_episode: %s\n" % restart)
    writer.write("  scene_list: %s\n" % scene_list)
    writer.write("  world_list: %s\n" % world_list)
    writer.write("  goal_dir_list: %s\n" % goal_dir_list)
    config_wtf.close()


for k in range(len(scene_type_list)):
    a_scene = scene_type_list[k]
    world = world_list[k]
    config_path = '../configs/eval/'
    if not os.path.exists(config_path):
        os.makedirs(config_path)
    count = 1

    # For scene 1 - 4 #
    for i in range(num_box):
        file_name = a_scene + "_0" + str(count) + ".yaml"
        config_file = config_path + file_name
        config_wtf = open(config_file, 'w')
        scn_type = "Static"
        static_ob = True
        dual = False
        dynamic_box = 2
        total_box = i + 1
        reverse = True
        restart = True
        write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                      goal_dir_list[0])
        if world == 'plus-corridor':
            file_name = a_scene + "_" + str(20 + count) + ".yaml"
            config_file = config_path + file_name
            config_wtf = open(config_file, 'w')
            write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                          goal_dir_list[1])
        count += 1

    # For scene 5 - 10 #
    for n_box in range(1, 3):
        for a_dir in scene_list:
            if a_dir == 'Circular':
                continue
            if count < 10:
                file_name = a_scene + "_0" + str(count) + ".yaml"
            else:
                file_name = a_scene + "_" + str(count) + ".yaml"
            config_file = config_path + file_name
            config_wtf = open(config_file, 'w')
            scn_type = a_dir
            static_ob = False
            dual = False
            dynamic_box = n_box
            total_box = 4
            reverse = True
            restart = True
            write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                          goal_dir_list[0])
            if world == 'plus-corridor':
                file_name = a_scene + "_" + str(20 + count) + ".yaml"
                config_file = config_path + file_name
                config_wtf = open(config_file, 'w')
                write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                              goal_dir_list[1])
            count += 1

    # For scene 11 - 12, 19 - 20 #
    flag = False
    for static in [False, True]:
        for a_dir in ['Parallel', 'Perpendicular']:
            if static:
                    cur_count = count + 6
                    file_name = a_scene + "_" + str(cur_count) + ".yaml"
            else:
                cur_count = count
                file_name = a_scene + "_" + str(count) + ".yaml"
            print file_name
            config_file = config_path + file_name
            config_wtf = open(config_file, 'w')
            scn_type = a_dir
            static_ob = static
            dual = True
            dynamic_box = 2
            total_box = 4
            reverse = True
            restart = True
            write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                          goal_dir_list[0])
            if world == 'plus-corridor':
                file_name = a_scene + "_" + str(20 + cur_count) + ".yaml"
                print file_name
                config_file = config_path + file_name
                config_wtf = open(config_file, 'w')
                write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                              goal_dir_list[1])
            count += 1

    count = count - 2

    # For scene 13 - 18 #
    for n_box in range(1, 3):
        for a_dir in scene_list:
            if a_dir == 'Circular':
                continue
            file_name = a_scene + "_" + str(count) + ".yaml"
            config_file = config_path + file_name
            config_wtf = open(config_file, 'w')
            scn_type = a_dir
            static_ob = True
            dual = False
            dynamic_box = n_box
            total_box = 4
            reverse = True
            restart = True
            write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                          goal_dir_list[0])
            if world == 'plus-corridor':
                file_name = a_scene + "_" + str(20 + count) + ".yaml"
                config_file = config_path + file_name
                config_wtf = open(config_file, 'w')
                write_to_file(config_wtf, world, scn_type, static_ob, dual, dynamic_box, total_box, reverse, restart,
                              goal_dir_list[1])
            count += 1
