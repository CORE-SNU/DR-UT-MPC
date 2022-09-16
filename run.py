#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    O            : open/close all doors of vehicle
    T            : toggle vehicle's telemetry

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

import time

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


red                 = carla.Color(255, 0, 0)
green               = carla.Color(0, 255, 0)
blue                = carla.Color(47, 210, 231)
cyan                = carla.Color(0, 255, 255)
yellow              = carla.Color(255, 255, 0)
orange              = carla.Color(255, 162, 0)
white               = carla.Color(255, 255, 255)
plot_COLOR          = (248, 64, 24)
ellipse_color       = (0,255,0)
ellipse_life_time   = 1/100
path_life_time      = 30
trail_life_time     = 5
main_display_width  = 1280
main_display_height = 720
mini_display_width  = int(1280/3)
mini_display_height = int(720/3)
main_display_fov    = 90
mini_display_fov    = 90
line_thickness      = 3

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def AngleToRad(theta):
     if theta < 0:
          theta = theta + 360
     theta = theta*np.pi/180
     return theta

def get_bounding_box(v):
     """Gets the bounding box corners of an actor in world space"""
     trans = v.get_transform()
     bb = v.bounding_box.extent
     corners = [carla.Location(x=-bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=bb.y),
                carla.Location(x=-bb.x, y=-bb.y)
               ]
     corners_nom = corners
     trans.transform(corners)
     bbox = np.array([[c.x, c.y] for c in corners])
     corners_nom_ = [bb.x, bb.y]
     return bbox, corners_nom_

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, Ts, carla_world, hud, args):
        current_time = time.strftime("%m%d-%H%M%S")
        self.path = './data/' + 'data_' + current_time + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.Ts = Ts
        self.world = carla_world
        self.with_obs = args.with_obs
        self.recording = args.record
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

        self.waypoint_sep = 0.5
        self.N = 30
        self.hud = hud
        self.player = None
        self.obstacle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        minicam_index     = self.camera_manager.mini_index if self.camera_manager is not None else 0
        minicam_pos_index = self.camera_manager.mini_transform_index if self.camera_manager is not None else 0

        #------ PLAYER----------
        # Get a random blueprint.
        blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        blueprint.set_attribute('role_name', self.actor_role_name)
        color = blueprint.get_attribute('color').recommended_values[1]
        blueprint.set_attribute('color', color)

        #--------OSBTACLE-------
        #Get a random blueprint.
        blueprint_obs = get_actor_blueprints(self.world, "vehicle.mercedes*", self._actor_generation)[0]
        blueprint_obs.set_attribute('role_name', 'obstacle')
        color = blueprint_obs.get_attribute('color').recommended_values[0]
        blueprint_obs.set_attribute('color', color)

        # Spawn the player and obstacle.
        if (self.player is not None):

            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0

            if self.with_obs and (self.obstacle is not None):
                spawn_point_obs = self.obstacle.get_transform()
                spawn_point_obs.location.z += 2.0
                spawn_point_obs.rotation.roll = 0.0
                spawn_point_obs.rotation.pitch = 0.0

            self.destroy()

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
            if self.with_obs:
                self.obstacle = self.world.try_spawn_actor(blueprint_obs, spawn_point_obs)
                self.modify_vehicle_physics(self.obstacle)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            
            spawn_point = spawn_points[111] if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)

        if self.with_obs:
            
            while self.obstacle is None:
                if not self.map.get_spawn_points():
                    print('There are no spawn points available in your map/town.')
                    print('Please add some Vehicle Spawn Point to your UE4 scene.')
                    sys.exit(1)
                spawn_points_obs = self.map.get_spawn_points()
                spawn_point_obs = spawn_points_obs[81] if spawn_points_obs else carla.Transform()
                self.obstacle = self.world.try_spawn_actor(blueprint_obs, spawn_point_obs)
                self.modify_vehicle_physics(self.obstacle)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.path, self.player, self.hud, self._gamma, self.recording)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.mini_transform_index = minicam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.camera_manager.set_mini_sensor(minicam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        waypoint = self.map.get_waypoint(spawn_point.location, project_to_road = True)
        self.waypoint_list =  [waypoint]
        for i in range(2000):
            x = self.waypoint_list[-1].next(self.waypoint_sep)[0]
            self.waypoint_list.append(x)

        waypoints = [[w.transform.location.x, w.transform.location.y]  for w in self.waypoint_list]

        self.waypoints_array = np.array(waypoints)
        dist = np.sqrt(((self.waypoints_array[0] - self.waypoints_array)**2).sum(axis=1))
        ind = np.argmin(dist[1:])

        if ind > 100:
            self.waypoint_list = self.waypoint_list[:ind+1]
            self.waypoints_array = self.waypoints_array[:ind+1]
        np.save('waypoints.npy', self.waypoints_array)

        goal_idx = np.argmin(((np.array([spawn_point.location.x, spawn_point.location.y])
                                       - self.waypoints_array)**2).sum(axis=1))
        self.waypoints = self.waypoint_list[goal_idx:goal_idx+self.N]
       
            
        self.world.tick()
        

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)
        x = self.player.get_location()
        goal_idx = np.argmin(((np.array([x.x, x.y]) -
                             self.waypoints_array)**2).sum(axis=1))
        self.waypoints = self.waypoint_list[goal_idx:goal_idx+self.N]

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)
        w = [self.player.get_location()] + [self.waypoints[-1].transform.location]
        for i in range(len(w)-1):
            self.world.debug.draw_line(w[i], w[i+1], thickness = .01, color = carla.Color(0, 0, 255), life_time = 2*self.Ts)


    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.mini_sensor.destroy()
        self.camera_manager.sensor      = None
        self.camera_manager.mini_sensor = None
        self.camera_manager.index       = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.camera_manager.mini_sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if self.obstacle is not None:
            self.obstacle.destroy()





# ==============================================================================
# -- MPC Control ---------------------------------------------------------------
# ==============================================================================

class MPCControl(object):
    def __init__(self, tm, world, Ts, gp, method, scenario):
        self.Ts = Ts
        if gp:
            if method=='mean':
                from Mean_MPC import MPC
            else:
                from DR_UT_MPC import MPC #For GP-based
        else:
            from Vanilla_MPC import MPC #For model-based
        print('Done importing')
        self.mpc = MPC(world.path, world.with_obs, 2*world.player.bounding_box.extent.x, self.Ts, world.N,
                       [200, 200, 2*np.pi, 5*world.player_max_speed],
                       [-200, -200, 0, -5*world.player_max_speed],
                       [0.05, 3, 1.22], [-0.05, -3, -1.22],
                       [1, 1, 0, 0.2], [1, 1, 0, 0.2], [1.5, 3], method=method)
        print('Finished init')
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        self.autopilot = False
        self.tm = tm
#        self.obstacle_controls = np.load('obs_inputs_sc_{}.npy'.format(scenario))


    def parse_events(self, world, t):

        if world.with_obs:
            tm_port = self.tm.get_port()
            world.obstacle.set_autopilot(True, tm_port)
            if t >=50 and t<=100:
                self.tm.vehicle_percentage_speed_difference(world.obstacle, 30)
            else:
                self.tm.vehicle_percentage_speed_difference(world.obstacle, 60)
            self.tm.ignore_lights_percentage(world.obstacle,  100)
            self.tm.ignore_vehicles_percentage(world.obstacle, 100)
            self.tm.distance_to_leading_vehicle(world.obstacle, 0)


        # Infer the current location and orientation of hero vehicle
        velocity_vec      = world.player.get_velocity()
        current_transform = world.player.get_transform()
        x         = current_transform.location.x
        y         = current_transform.location.y
        yaw       = AngleToRad(current_transform.rotation.yaw)

        speed     = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        robot_state       = np.array([x, y, yaw, speed])

        goal = [[world.waypoints[i].transform.location.x, world.waypoints[i].transform.location.y, 0, AngleToRad(world.waypoints[i].transform.rotation.yaw)] for i in range(world.N)]
        if world.with_obs:
            obs_box, obs_box_nom = get_bounding_box(world.obstacle)
            # Infer the current location and orientation of hero vehicle
            current_transform_obs = world.obstacle.get_transform()
            velocity_vec_obs      = world.obstacle.get_velocity()
            x_obs         = current_transform_obs.location.x
            y_obs         = current_transform_obs.location.y
            speed_obs     = math.sqrt(velocity_vec_obs.x**2 + velocity_vec_obs.y**2 + velocity_vec_obs.z**2)
            obs_state       = [x_obs, y_obs, speed_obs]

        else:
            obs_box, obs_box_nom = [], [0, 0]
            obs_state = [0.0, 50.0, -10000.0, 0.0, 0.0]

        car_box, car_box_nom = get_bounding_box(world.player)
        controls, pred, pred_obs, flag, gp_on = self.mpc.solve_MPC(robot_state, obs_state, goal, obs_box, car_box, obs_box_nom, car_box_nom, t)
        if False:
            print("Infeasible!")
            return True
        else:
            if pred is not None:

                for k in range(pred.shape[1]):
                    world.world.debug.draw_point(carla.Location(x=pred[0,k], y=pred[1,k], z=current_transform.location.z), size=0.05, color = carla.Color(255, 0, 0), life_time = 2*self.Ts)
                    world.world.debug.draw_point(carla.Location(x=pred_obs[0,k], y=pred_obs[1,k], z=current_transform.location.z), size=0.05, color = carla.Color(0, 255, 0), life_time = 2*self.Ts)

            world.player.apply_control(carla.VehicleControl(throttle=float(controls[0]), brake=float(controls[1]), steer=float(controls[2])))
            #world.player.apply_control(carla.VehicleControl(throttle=float(0), brake=float(0), steer=float(0)))
#
#            if world.with_obs:
#                 world.obstacle.set_autopilot(True)
#                 obs_cont = self.obstacle_controls[t]
#                 world.obstacle.apply_control(carla.VehicleControl(throttle=float(obs_cont[0]), brake=float(obs_cont[1]), steer=float(obs_cont[2])))
# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, mini_width, mini_height):
        self.dim = (width, height, mini_width, mini_height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, path, parent_actor, hud, gamma_correction, record):
        self.path = path
        self.sensor = None
        self.mini_sensor = None
        self.surface = None
        self.mini_surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = record
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-7.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=-8.0, y=-6, z=6.0), carla.Rotation(pitch=10.0, yaw=-5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=0.0, z=10.0), carla.Rotation(pitch=-90.0)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
                {'lens_circle_multiplier': '3.0',
                'lens_circle_falloff': '3.0',
                'chromatic_aberration_intensity': '0.5',
                'chromatic_aberration_offset': '0'}]]
        world = self._parent.get_world()
        self.bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = self.bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)
            item.append(bp)
        self.index = None
        self.mini_index = None


    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            # If the self.sensor is None, spawn the actor
            cam_transform_index = self.transform_index+3

            bp = self.bp_library.find(self.sensors[index][0])
            bp.set_attribute('image_size_x', str(self.hud.dim[0]))
            bp.set_attribute('image_size_y', str(self.hud.dim[1]))
            bp.set_attribute('fov', str(main_display_fov))

            self.sensor = self._parent.get_world().spawn_actor(bp,
                                                               self._camera_transforms[cam_transform_index][0],
                                                               attach_to=self._parent,
                                                               attachment_type=self._camera_transforms[cam_transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            mini_flag = 0
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image, mini_flag))
            sensor_calibration = np.identity(3)
            sensor_calibration[0, 2] = main_display_width / 2.0
            sensor_calibration[1, 2] = main_display_height / 2.0
            sensor_calibration[0, 0] = sensor_calibration[1, 1] = main_display_width / (2.0 * np.tan(main_display_fov * np.pi / 360.0))
            self.sensor.calibration  = sensor_calibration
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def set_mini_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.mini_index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.mini_index][2]))
        if needs_respawn:
            if self.mini_sensor is not None:
                self.mini_sensor.destroy()
                self.mini_surface = None
            # If the self.mini_sensor is None, spawn the actor
            cam_transform_index = self.transform_index+5
            mini_cam_transform  = self._camera_transforms[cam_transform_index][0]
            mini_cam_attacher   = self._camera_transforms[cam_transform_index][1]

            # Modify the blueprints of the mini_sensor
            bp = self.bp_library.find(self.sensors[index][0])
            bp.set_attribute('image_size_x', str(self.hud.dim[2]))
            bp.set_attribute('image_size_y', str(self.hud.dim[3]))
            bp.set_attribute('fov', str(mini_display_fov))

            # Spawn the actor
            self.mini_sensor = self._parent.get_world().spawn_actor(bp, mini_cam_transform, attach_to=self._parent,
                                                                    attachment_type=mini_cam_attacher)
            # We need to pass the lambda a weak reference to self to avoid circular reference.
            weak_self = weakref.ref(self)
            mini_flag = 1
            self.mini_sensor.listen(lambda image_mini: CameraManager._parse_image(weak_self, image_mini, mini_flag))
            mini_calibration = np.identity(3)
            mini_calibration[0, 2] = mini_display_width / 2.0
            mini_calibration[1, 2] = mini_display_height / 2.0
            mini_calibration[0, 0] = mini_calibration[1, 1] = mini_display_width / (2.0 * np.tan(mini_display_fov * np.pi / 360.0))
            self.mini_sensor.calibration = mini_calibration
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.mini_index = index

    def toggle_camera(self): # NOT USED
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)
        self.set_mini_sensor(self.mini_index, notify=False, force_respawn=True)

    def next_sensor(self): # NOT USED
        self.set_sensor(self.index + 1)
        self.set_mini_sensor(self.index + 1)

    def toggle_recording(self): # NOT USED
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None and self.mini_surface is not None:
            display.blit(self.surface, (0, 0))
            display.blit(self.mini_surface, (main_display_width - mini_display_width,
                                             main_display_height - mini_display_height))

    @staticmethod
    def _parse_image(weak_self, image, mini_flag):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith('sensor.camera.dvs'):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            if mini_flag == 1:
                self.mini_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            if mini_flag == 0:
                self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            if mini_flag == 1:
                image.save_to_disk(self.path + '/mini_%08d.jpg' % image.frame)
            else:
                image.save_to_disk(self.path + '/%08d.jpg' % image.frame)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    Ts = 0.1

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        sim_world = client.get_world()
        original_settings = sim_world.get_settings()
        settings = sim_world.get_settings()
        if not settings.synchronous_mode:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = Ts
        sim_world.apply_settings(settings)

        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height, args.mini_width, args.mini_height)
        world = World(Ts, sim_world, hud, args)
        controller = MPCControl(traffic_manager, world, Ts, args.gp, args.method, args.scenario)
        sim_world.tick()
        t = 0
        clock = pygame.time.Clock()
        while t<400:
            sim_world.tick()
            t += 1
            clock.tick_busy_loop(60)
            if controller.parse_events(world, t):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

        controller.mpc.save()
        controller.mpc.render(world)
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.bmw*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    #Algorithm Parameters
    argparser.add_argument(
        '--with_obs',
        action='store_true',
        help='Add an obstacle')
    argparser.add_argument(
        '--gp',
        action='store_true',
        help='Use GP')
    argparser.add_argument(
        '--method',
        default='cvar',
        type=str,
        help='Risk Method')
    argparser.add_argument(
        '--record',
        action='store_true',
        help='Record the snapshots')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.mini_width, args.mini_height = int(args.width/3), int(args.height/3)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
