import argparse
import cv2
import os
import glob

class Annotator:
    def __init__(self, dataset_dir, class_filepath=None):
        self.dataset_dir = dataset_dir
        self.classes = self.load_classes(class_filepath)
        self.image_paths = self.load_images()
        self.current_image_index = 0
        self.boxes = []  # List of (track_id, class_id, x_min, y_min, x_max, y_max)
        self.current_class_id = 0
        self.drawing = False
        self.modifying = False
        self.modifying_track_id = None
        self.ix, self.iy = -1, -1
        self.img = None
        self.unsaved_box = None
        self.track_id_counter = 0
        self.colors = self.generate_colors()

    def load_classes(self, class_filepath):
        if class_filepath is None:
            classes_path = os.path.join(self.dataset_dir, "classes.txt")
        else:
            classes_path = class_filepath
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        with open(classes_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def load_images(self):
        img_extensions = ["*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        for ext in img_extensions:
            image_paths.extend(glob.glob(os.path.join(self.dataset_dir, "images", ext)))
        image_paths = sorted(image_paths)
        if not image_paths:
            raise FileNotFoundError("No images found in the dataset directory.")
        return image_paths

    def generate_colors(self):
        return [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (192, 192, 192), # Silver
            (128, 0, 0),    # Maroon
            (128, 128, 0),  # Olive
            (0, 128, 0)     # Dark Green
        ]

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.modifying:
                # Modify an existing box
                for i, (track_id, class_id, x_min, y_min, x_max, y_max) in enumerate(self.boxes):
                    if track_id == self.modifying_track_id:
                        self.boxes[i] = (track_id, class_id, min(self.ix, x), min(self.iy, y), max(self.ix, x), max(self.iy, y))
                        self.modifying = False
                        self.modifying_track_id = None
                        self.redraw_image()
                        return
            elif not self.drawing:
                # First click: set the first corner
                self.drawing = True
                self.ix, self.iy = x, y
            else:
                # Second click: set the opposite corner and finalize the box
                self.drawing = False
                x_min, y_min = min(self.ix, x), min(self.iy, y)
                x_max, y_max = max(self.ix, x), max(self.iy, y)
                self.unsaved_box = (self.track_id_counter, self.current_class_id, x_min, y_min, x_max, y_max)
                self.redraw_image()

    def load_labels(self, image_path):
        """Load bounding boxes from a YOLO-format labels file."""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        labels_path = os.path.join(self.dataset_dir, "labels", f"{base_name}.txt")
        if not os.path.exists(labels_path):
            return []

        h, w, _ = cv2.imread(image_path).shape
        boxes = []
        with open(labels_path, "r") as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x_min = int((x_center - width / 2) * w)
                y_min = int((y_center - height / 2) * h)
                x_max = int((x_center + width / 2) * w)
                y_max = int((y_center + height / 2) * h)
                track_id = self.track_id_counter
                self.track_id_counter += 1
                boxes.append((track_id, int(class_id), x_min, y_min, x_max, y_max))
        return boxes

    def redraw_image(self):
        """Redraw the image with all bounding boxes and the legend."""
        self.img = cv2.imread(self.image_paths[self.current_image_index])

        # Load existing labels if not already loaded
        if not self.boxes:
            self.boxes = self.load_labels(self.image_paths[self.current_image_index])

        img_h, img_w = self.img.shape[:2]

        # Draw all saved bounding boxes
        for track_id, class_id, x_min, y_min, x_max, y_max in self.boxes:
            color = self.colors[class_id]
            cv2.rectangle(self.img, (x_min, y_min), (x_max, y_max), color, 2)
            # Default position: above the box
            text_x, text_y = x_min + 5, y_min - 10
            # If above is outside, put inside the box
            text_edge_buffer = 20
            if text_y < text_edge_buffer:
                text_y = y_min + 15
            cv2.putText(self.img, f"{track_id}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw the unsaved box if it exists
        if self.unsaved_box:
            track_id, class_id, x_min, y_min, x_max, y_max = self.unsaved_box
            color = self.colors[class_id]
            cv2.rectangle(self.img, (x_min, y_min), (x_max, y_max), color, 2)
            text_x, text_y = x_min + 5, y_min - 10
            text_edge_buffer = 20
            if text_y < text_edge_buffer:
                text_y = y_min + 15
            cv2.putText(self.img, f"{track_id}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show the legend
        self.show_legend()
        cv2.imshow("Image", self.img)

    def show_legend(self):
        y_offset = 20
        for class_id, class_name in enumerate(self.classes):
            color = self.colors[class_id]
            cv2.putText(self.img, f"{class_id}: {class_name}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_offset += 20

    def save_yolo_format(self):
        if self.unsaved_box:
            self.boxes.append(self.unsaved_box)
            self.unsaved_box = None
            self.track_id_counter += 1

        h, w, _ = self.img.shape
        image_path = self.image_paths[self.current_image_index]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(self.dataset_dir, "labels")
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"{base_name}.txt")

        with open(txt_path, "w") as f:
            for track_id, class_id, x_min, y_min, x_max, y_max in self.boxes:
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def annotate(self):
        while 0 <= self.current_image_index < len(self.image_paths):
            image_path = self.image_paths[self.current_image_index]
            self.img = cv2.imread(image_path)
            self.boxes = []
            self.unsaved_box = None
            self.track_id_counter = 0

            cv2.namedWindow("Image")
            cv2.setMouseCallback("Image", self.draw_rectangle)
            self.redraw_image()

            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord('n'):  # Next image
                    self.current_image_index += 1
                    break
                elif key == ord('p'):  # Previous image
                    self.current_image_index = max(0, self.current_image_index - 1)
                    break
                elif key == ord('s'):  # Save annotations
                    self.save_yolo_format()
                    print(f"Annotations saved for {image_path}")
                elif key == ord('m'):  # Modify a box
                    self.modifying = True
                    self.modifying_track_id = int(input("Enter track ID to modify: "))
                elif key == ord('d'):  # Delete a box
                    del_id = int(input("Enter track ID to delete: "))
                    self.boxes = [box for box in self.boxes if box[0] != del_id]
                    self.redraw_image()
                    print(f"Deleted box with track ID {del_id}")
                elif key in map(ord, map(str, range(len(self.classes)))):  # Change class
                    self.current_class_id = int(chr(key))
                    print(f"Current class set to {self.current_class_id}: {self.classes[self.current_class_id]}")
                elif key == 27:  # ESC to exit
                    cv2.destroyAllWindows()
                    return

            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image annotation tool")
    parser.add_argument("directory_path", help="Dataset directory")
    parser.add_argument("-c", "--classes", default=None, help="Path to classes.txt file")
    args = parser.parse_args()

    dataset_dir = args.directory_path
    classes_path = args.classes

    annotator = Annotator(dataset_dir, class_filepath=classes_path)
    annotator.annotate()