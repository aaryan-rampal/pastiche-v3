import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from datetime import datetime


class DrawingApp:
    def __init__(self, canvas_size=(512, 512)):
        self.canvas_size = canvas_size
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, canvas_size[0])
        self.ax.set_ylim(0, canvas_size[1])
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()  # Make (0,0) top-left like image coordinates
        self.ax.set_title("Draw on the canvas - Close window when done")

        # Store drawing data
        self.lines = []
        self.current_line = []
        self.is_drawing = False

        # Connect mouse events
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.is_drawing = True
        self.current_line = [(event.xdata, event.ydata)]

    def on_motion(self, event):
        if not self.is_drawing or event.inaxes != self.ax:
            return

        self.current_line.append((event.xdata, event.ydata))

        # Draw the current line segment
        if len(self.current_line) >= 2:
            x_coords = [p[0] for p in self.current_line[-2:]]
            y_coords = [p[1] for p in self.current_line[-2:]]
            self.ax.plot(x_coords, y_coords, "k-", linewidth=2)
            self.fig.canvas.draw_idle()

    def on_release(self, event):
        if not self.is_drawing:
            return
        self.is_drawing = False
        if self.current_line:
            self.lines.append(self.current_line.copy())
        self.current_line = []

    def start_drawing(self):
        """Start the interactive drawing session"""
        print("Instructions:")
        print("- Click and drag to draw")
        print("- Close the window when you're done drawing")
        plt.show()

    def create_image_array(self):
        """Convert the drawing to a numpy array"""
        # Create a new figure with the exact canvas size
        fig, ax = plt.subplots(
            figsize=(self.canvas_size[0] / 100, self.canvas_size[1] / 100), dpi=100
        )
        ax.set_xlim(0, self.canvas_size[0])
        ax.set_ylim(0, self.canvas_size[1])
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

        # Set white background
        ax.set_facecolor("white")

        # Draw all the lines
        for line in self.lines:
            if len(line) >= 2:
                x_coords = [p[0] for p in line]
                y_coords = [p[1] for p in line]
                ax.plot(x_coords, y_coords, "k-", linewidth=2)

        # Convert to array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        buf = buf[:, :, :3]

        plt.close(fig)
        return buf

    def save_drawing(self, base_filename=None):
        """Save the drawing as both image and numpy array"""
        if not self.lines:
            print("No drawing to save!")
            return None, None

        # Generate filename if not provided
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"drawing_{timestamp}"

        # Create the image array
        img_array = self.create_image_array()

        # Save as image (PNG)
        img = Image.fromarray(img_array)
        img_filename = f"{base_filename}.png"
        img.save(img_filename)
        print(f"Image saved as: {img_filename}")

        # Save as numpy array
        npy_filename = f"{base_filename}.npy"
        np.save(npy_filename, img_array)
        print(f"Numpy array saved as: {npy_filename}")

        return img_filename, npy_filename

    def clear_drawing(self):
        """Clear the current drawing"""
        self.lines = []
        self.ax.clear()
        self.ax.set_xlim(0, self.canvas_size[0])
        self.ax.set_ylim(0, self.canvas_size[1])
        self.ax.set_aspect("equal")
        self.ax.invert_yaxis()
        self.ax.set_title("Draw on the canvas - Close window when done")
        self.fig.canvas.draw()


def main():
    """Main function to run the drawing app"""
    print("Starting Drawing App...")

    # Create the drawing app
    app = DrawingApp(canvas_size=(512, 512))

    # Start drawing
    app.start_drawing()

    # After the window is closed, save the drawing
    if app.lines:
        print("\nDrawing session completed!")

        # Ask user for filename
        filename = input(
            "Enter filename (without extension, press Enter for auto-generated): "
        ).strip()
        if not filename:
            filename = None

        # Save the drawing
        img_file, npy_file = app.save_drawing(filename)

        if img_file and npy_file:
            print(f"\nFiles saved successfully!")
            print(f"Image: {img_file}")
            print(f"NumPy array: {npy_file}")

            # Show some info about the saved array
            array = np.load(npy_file)
            print(f"Array shape: {array.shape}")
            print(f"Array dtype: {array.dtype}")
    else:
        print("No drawing was made.")


if __name__ == "__main__":
    main()
