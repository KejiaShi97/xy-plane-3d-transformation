import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd


class PlaneTransformGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Plane Transformation Tool")
        self.root.geometry("800x600")

        # Create main frames
        self.input_frame = ttk.LabelFrame(root, text="Input Data", padding="10")
        self.input_frame.pack(fill="x", padx=10, pady=5)

        self.output_frame = ttk.LabelFrame(root, text="Output", padding="10")
        self.output_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Create input widgets
        self.create_input_widgets()
        self.create_output_widgets()

    def create_input_widgets(self):
        # Plane points input
        ttk.Label(self.input_frame, text="Plane Definition Points (x,y,z per line):").pack(anchor="w")
        self.plane_points_text = tk.Text(self.input_frame, height=5, width=50)
        self.plane_points_text.pack(pady=5)

        # All points input
        ttk.Label(self.input_frame, text="All Points to Transform (x,y,z per line):").pack(anchor="w")
        self.all_points_text = tk.Text(self.input_frame, height=5, width=50)
        self.all_points_text.pack(pady=5)

        # Buttons frame
        button_frame = ttk.Frame(self.input_frame)
        button_frame.pack(fill="x", pady=5)

        ttk.Button(button_frame, text="Load Plane Points", command=self.load_plane_points).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Load All Points", command=self.load_all_points).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Transform", command=self.transform_points).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side="left", padx=5)

    def create_output_widgets(self):
        # Output text widget
        self.output_text = tk.Text(self.output_frame, height=15, width=70)
        self.output_text.pack(fill="both", expand=True, pady=5)

    def parse_points(self, text_input):
        try:
            lines = text_input.strip().split('\n')
            points = []
            for line in lines:
                if line.strip():
                    coords = [float(x) for x in line.replace(',', ' ').split()]
                    if len(coords) != 3:
                        raise ValueError("Each line must contain exactly 3 coordinates (x,y,z)")
                    points.append(coords)
            return np.array(points)
        except Exception as e:
            messagebox.showerror("Error", f"Error parsing coordinates: {str(e)}")
            return None

    def transform_points(self):
        # Get input points
        plane_points = self.parse_points(self.plane_points_text.get("1.0", tk.END))
        all_points = self.parse_points(self.all_points_text.get("1.0", tk.END))

        if plane_points is None or all_points is None:
            return

        try:
            # Center the points
            centered_points = plane_points - np.mean(plane_points, axis=0)

            # Compute the covariance matrix
            cov_matrix = np.cov(centered_points, rowvar=False)

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # The eigenvector corresponding to the smallest eigenvalue is the normal vector
            normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

            # Define the translation vector
            translation_vector = np.mean(plane_points, axis=0)

            # Compute the rotation matrix
            z_axis = np.array([0, 0, 1])
            rotation_vector = np.cross(normal_vector, z_axis)
            rotation_angle = np.arccos(np.dot(normal_vector, z_axis) / np.linalg.norm(normal_vector))

            # Handle small angle case
            if np.linalg.norm(rotation_vector) > 1e-10:
                rotation_vector /= np.linalg.norm(rotation_vector)
            else:
                rotation_vector = np.array([1, 0, 0])

            rotation = R.from_rotvec(rotation_angle * rotation_vector)
            rotation_matrix = rotation.as_matrix()

            # Transform all points
            transformed_points = np.array([
                rotation_matrix @ (point - translation_vector)
                for point in all_points
            ])

            # Create output DataFrame
            transformed_points_df = pd.DataFrame(transformed_points, columns=["x'", "y'", "z'"])

            # Display results
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", str(transformed_points_df))

            # Store results for saving
            self.transformed_points_df = transformed_points_df

            messagebox.showinfo("Success", "Transformation completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Error during transformation: {str(e)}")

    def load_plane_points(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.plane_points_text.delete("1.0", tk.END)
                    self.plane_points_text.insert("1.0", file.read())
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def load_all_points(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.all_points_text.delete("1.0", tk.END)
                    self.all_points_text.insert("1.0", file.read())
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")

    def save_results(self):
        if not hasattr(self, 'transformed_points_df'):
            messagebox.showerror("Error", "No results to save. Please perform transformation first.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.transformed_points_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", "Results saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving file: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PlaneTransformGUI(root)
    root.mainloop()