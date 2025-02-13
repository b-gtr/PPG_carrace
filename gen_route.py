import carla
import pygame
import json
import datetime
import os
import math
import cv2
import numpy as np

# Configuration
SAVE_FILENAME = "recorded_route.json"
WINDOW_SIZE = (800, 600)
FPS = 60
MIN_DISTANCE = 2.0

class RouteRecorder:
    def __init__(self):
        # Initialize attributes
        self.recording = True
        self.waypoints = []
        self.last_location = None
        self.camera_image = None
        
        # CARLA connection
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Setup pygame (single window)
        pygame.init()
        self.display = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("CARLA Controller")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        
        # Setup vehicle and camera
        self.vehicle = None
        self.camera = None
        self.setup_vehicle()
        self.setup_camera()

    def setup_vehicle(self):
        self.destroy_actors()
        blueprint = self.world.get_blueprint_library().filter('model3')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
        self.vehicle.set_autopilot(False)

    def setup_camera(self):
        # Bird's-eye view camera
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')

        transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=12.0),
            carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
        )

        self.camera = self.world.spawn_actor(
            camera_bp,
            transform,
            attach_to=self.vehicle
        )

        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            labels = array[:, :, 2].astype(np.float32)
            labels /= 22.0  # Normalize to [0,1]
            self.camera_image = labels

        self.camera.listen(camera_callback)

    def destroy_actors(self):
        for actor in [self.vehicle, self.camera]:
            if actor is not None:
                actor.destroy()

    def save_waypoints(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{SAVE_FILENAME}"
        
        duration = (self.waypoints[-1]['timestamp'] - self.waypoints[0]['timestamp']) if self.waypoints else 0
        distance = sum(math.sqrt(
            (self.waypoints[i+1]['x'] - self.waypoints[i]['x'])**2 +
            (self.waypoints[i+1]['y'] - self.waypoints[i]['y'])**2
        ) for i in range(len(self.waypoints)-1))
        
        metadata = {
            "timestamp": timestamp,
            "duration_seconds": round(duration, 2),
            "distance_meters": round(distance, 2),
            "waypoints_count": len(self.waypoints),
            "min_distance": MIN_DISTANCE,
            "waypoints": self.waypoints
        }
        
        if not os.path.exists('recorded_routes'):
            os.makedirs('recorded_routes')
        
        with open(f'recorded_routes/{filename}', 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Saved {len(self.waypoints)} waypoints to {filename}")

    def get_control(self):
        control = carla.VehicleControl()
        keys = pygame.key.get_pressed()
        
        # Improved steering logic
        control.steer = 0.0
        if keys[pygame.K_a]: control.steer -= 0.5
        if keys[pygame.K_d]: control.steer += 0.5
        
        # Throttle/brake logic
        control.throttle = 0.5 if keys[pygame.K_w] else 0.0
        control.brake = 0.5 if keys[pygame.K_s] else 0.0
        
        # Handbrake
        control.hand_brake = keys[pygame.K_SPACE]
        return control

    def draw_text(self, surface, text, pos):
        text_surface = self.font.render(text, True, (255, 255, 255))
        surface.blit(text_surface, pos)

    def run(self):
        try:
            while self.recording:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or \
                      (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        self.recording = False

                # Get vehicle data
                transform = self.vehicle.get_transform()
                location = transform.location
                velocity = self.vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                
                # Update waypoints
                current_time = pygame.time.get_ticks() / 1000.0
                if self.last_location:
                    dx = location.x - self.last_location.x
                    dy = location.y - self.last_location.y
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    if distance >= MIN_DISTANCE:
                        self.waypoints.append({
                            'x': location.x,
                            'y': location.y,
                            'z': location.z,
                            'yaw': transform.rotation.yaw,
                            'speed': speed,
                            'timestamp': current_time
                        })
                        self.last_location = location
                else:
                    self.waypoints.append({
                        'x': location.x,
                        'y': location.y,
                        'z': location.z,
                        'yaw': transform.rotation.yaw,
                        'speed': speed,
                        'timestamp': current_time
                    })
                    self.last_location = location

                # Update display
                self.display.fill((30, 30, 30))
                self.draw_text(self.display, f"Speed: {speed:.1f} km/h", (20, 20))
                self.draw_text(self.display, f"Waypoints: {len(self.waypoints)}", (20, 50))
                self.draw_text(self.display, "[W/S] Throttle/Brake | [A/D] Steer | [SPACE] Handbrake | [Q] Quit", (20, 80))
                pygame.display.flip()

                # Apply controls
                control = self.get_control()
                self.vehicle.apply_control(control)

                # Update camera view
                if self.camera_image is not None:
                    gray = (self.camera_image * 255).astype(np.uint8)
                    cv2.imshow("Bird's Eye View", gray)
                    cv2.waitKey(1)

                self.clock.tick(FPS)

        finally:
            cv2.destroyAllWindows()
            if self.waypoints:
                self.save_waypoints()
            self.destroy_actors()
            pygame.quit()

if __name__ == '__main__':
    recorder = RouteRecorder()
    print("CARLA window should be visible in the background")
    print("Focus THIS WINDOW for controls")
    print("Bird's eye view in separate OpenCV window")
    recorder.run()
